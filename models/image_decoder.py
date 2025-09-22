# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------


from timm.layers import Mlp
from timm.models.vision_transformer import Block

from models import *
from .pos_embed import get_2d_sincos_pos_embed


class CrossAttentionBlock(Block):
    # devised from timm.models.vision_transformer import Block, to supporting cross-attention
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__(
            dim, num_heads, mlp_ratio, qkv_bias, qk_norm, proj_drop, attn_drop,
            init_values, drop_path, act_layer, norm_layer, mlp_layer
        )

        # cross-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=0.0,
            bias=qkv_bias,
            batch_first=True
        )

        self.norm_memory = norm_layer(dim)

    def forward(self, x, memory, key_padding_mask=None):
        """
        x: image features [B, N, D]
        memory: text features [B, L, D]
        key_padding_mask: padding mask [B, L]
        """
        # cross-attention
        attn_output, _ = self.attn(
            query=self.norm1(x),
            key=self.norm_memory(memory),
            value=self.norm_memory(memory),
            key_padding_mask=key_padding_mask
        )
        x = x + self.drop_path1(self.ls1(attn_output))

        # Feed-forward network
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class MAEDecoder(nn.Module):  # Masked Autoencoder with ViT backbone
    """
    MAE Decoder:
    - patch_size: size of each image patch
    - embed_dim: encoder hidden dimension
    - decoder_embed_dim: decoder hidden dimension
    - decoder_num_heads: attention heads in decoder
    - decoder_depth: number of transformer blocks
    - in_chans: number of input channels
    - norm_pix_loss: whether to normalize pixels before MSE loss
    - mask_ratio: fraction of patches to mask
    - use_learnable_pos_emb: if True, use learnable position embeddings
    """

    def __init__(
            self,
            patch_size=16,
            num_patches=196,
            embed_dim=768,
            decoder_embed_dim=512,
            decoder_num_heads=16,
            mlp_ratio=4.,
            norm_layer=nn.LayerNorm,
            decoder_depth=8,
            in_chans=3,
            norm_pix_loss=False,
            mask_ratio=0.75,
            use_learnable_pos_emb=True
    ):
        super().__init__()

        # Decoder projection and mask token
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token_img = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Fixed position embeddings
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim),
            requires_grad=False
        )

        # Transformer blocks for decoding
        self.decoder_blocks = nn.ModuleList([
            CrossAttentionBlock(
                decoder_embed_dim,
                decoder_num_heads,
                mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer
            ) for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)

        # Prediction heads
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)
        self.decoder_image = nn.Linear(self.num_patches, 1, bias=True)
        self.norm_pix_loss = norm_pix_loss
        self.use_learnable_pos_emb = use_learnable_pos_emb

        # Initialize weights and position embeddings
        self.initialize_weights()

    # 初始化权重
    def initialize_weights(self):
        # Initialize fixed position embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.num_patches ** 0.5),
            cls_token=True
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

        # Initialize mask and cls tokens
        nn.init.normal_(self.mask_token, std=.02)
        nn.init.normal_(self.mask_token_img, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """Convert images to patch sequences"""
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        return x.reshape(imgs.shape[0], h * w, p ** 2 * 3)

    def unpatchify(self, x):
        """Reconstruct images from patch sequences"""
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, 3)
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(x.shape[0], 3, h * p, h * p)

    def random_masking(self, x, mask_ratio):
        """Randomly mask a fraction of patches"""
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )

        mask = torch.ones(N, L, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_masked, mask, ids_restore

    def forward_decoder(self, x, text_embeds, ids_restore, key_padding_mask=None):
        """Decode masked embeddings with cross-attention"""
        # Project embeddings and insert mask tokens
        x = self.decoder_embed(x)
        if ids_restore is not None:

            mask_tokens = self.mask_token.repeat(
                x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
            )
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
            x_ = torch.gather(
                x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2])
            )
            x = torch.cat([x[:, :1, :], x_], dim=1)
        else:
            pass

        # Add position embeddings
        x = x + self.decoder_pos_embed

        # Transformer decoding
        for blk in self.decoder_blocks:
            x = blk(x, memory=text_embeds, key_padding_mask=key_padding_mask)
        x = self.decoder_norm(x)

        # Compute features and predictions
        features = x
        clip_features = self.decoder_image(
            features[:, 1:, :].transpose(1, 2)
        )
        pred = self.decoder_pred(x)[:, 1:, :]

        # (B, L-1, D), (B, L, decoder_embed_dim), (B, 1, decoder_embed_dim)
        return pred, features, clip_features

    def forward_loss(self, imgs, pred, mask):  # 平方差loss
        """
        Compute pixel-wise MSE loss for masked patches
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(-1, keepdim=True)
            var = target.var(-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5

        loss = (pred - target) ** 2
        return loss.mean(dim=-1)

    def del_tensor_ele_n(self, arr, index, n):
        """
        arr: 输入tensor
        index: 需要删除位置的索引
        n: 从index开始，需要删除的行数
        """
        arr1 = arr[0:index]
        arr2 = arr[index + n:]
        return torch.cat((arr1, arr2), dim=0)

    @torch.jit.ignore
    def no_weight_decay(self):
        # Exclude these parameters from weight decay
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, image_features, text_embeds, ids_restore, key_padding_mask=None):
        return self.forward_decoder(image_features, text_embeds, ids_restore, key_padding_mask)

# # 三种mae，mae_vit_base，mae_vit_large，mae_vit_huge
# def mae_vit_base_patch16_dec512d8b(**kwargs):
#     model = MaskedAutoencoderViT(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12,
#         decoder_embed_dim=512, decoder_num_heads=16,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model
#
#
# # decoder_depth=8,
# def mae_vit_large_patch16_dec512d8b(**kwargs):
#     model = MaskedAutoencoderViT(
#         patch_size=16, embed_dim=1024, depth=24, num_heads=16,
#         decoder_embed_dim=512, decoder_num_heads=16,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model
#
#
# def mae_vit_huge_patch14_dec512d8b(**kwargs):
#     model = MaskedAutoencoderViT(
#         patch_size=14, embed_dim=1280, depth=32, num_heads=16,
#         decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model
#
#
# # set recommended archs
# mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
# mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
# mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
