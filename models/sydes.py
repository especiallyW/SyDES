from functools import partial

from safetensors import safe_open
from transformers import CLIPTextModel

from configs.config import ModelConfig
from models import *
from .encoder import CustomCLIPVisionModel
from .image_decoder import MAEDecoder
from .text_decoder import MultimodalTransformer
from .utils import CombinedAggregator, EmbedToLatents, MLPLayer, GateKV
from .visualization import visualize_mae_batch, visualize_attention_batch


class MultimodalModel(nn.Module):
    """
    Multimodal architecture combining CLIP encoders, MAE-style image decoder,
    and cross-modal text decoder for multimodal sentiment classification tasks.
    """

    def __init__(self, is_pretrain: bool, config: ModelConfig):
        super().__init__()
        self.is_pretrain = is_pretrain
        self.config = config

        # Initialize vision and text encoders from pretrained CLIP
        self.image_encoder = CustomCLIPVisionModel.from_pretrained(config.image_encoder_path)
        # self.text_encoder = BertModel.from_pretrained(config.text_encoder_path)
        # for param in self.text_encoder.parameters(): param.data = param.data.contiguous()
        self.text_encoder = CLIPTextModel.from_pretrained(config.text_encoder_path)

        # Compute dimensions and patch counts
        patch_size = self.image_encoder.vision_model.config.patch_size
        num_patches = (self.image_encoder.vision_model.config.image_size // patch_size) ** 2
        self.img_dim = self.image_encoder.vision_model.config.hidden_size
        self.text_dim = self.text_encoder.text_model.config.hidden_size
        # self.text_dim = self.text_encoder.base_model.config.hidden_size

        # MAE-style image decoder for masked reconstruction
        img_nhead = config.image_decoder['nhead']
        img_embed_dim = config.image_decoder['embed_dim']
        self.text_kv_proj = nn.Linear(self.text_dim, img_embed_dim)
        self.img_kv_proj = nn.Linear(self.img_dim, img_embed_dim)
        self.text_kv_ln = nn.LayerNorm(img_embed_dim)
        self.img_kv_ln = nn.LayerNorm(img_embed_dim)
        self.gate_kv = GateKV(n_heads=img_nhead, dec_dim=img_embed_dim, init_bias=-3.0)
        # create L_img learnable queries for mapping text->patch-length features
        self.text_to_patch_queries = nn.Parameter(torch.randn(1, num_patches, img_embed_dim) * 0.02)  # [1, L_img, D]
        # single cross-attention block: use simple multihead attention
        self.text2patch_attn = nn.MultiheadAttention(embed_dim=img_embed_dim, num_heads=img_nhead, batch_first=True)

        self.image_decoder = MAEDecoder(
            patch_size=patch_size,
            num_patches=num_patches,
            embed_dim=self.img_dim,
            decoder_embed_dim=img_embed_dim,
            decoder_num_heads=img_nhead,
            mlp_ratio=config.image_decoder['mlp_ratio'],
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

        # Aggregator merges multiple view embeddings into one
        self.aggregator = CombinedAggregator(
            feature_dim=img_embed_dim,
            hidden_dim=img_embed_dim // 2,
            num_views=4,
        )

        # Projections from encoder outputs to latent space
        self.text_to_latents = EmbedToLatents(self.text_dim, config.latent_dim)
        self.img_to_latents = EmbedToLatents(self.img_dim, config.latent_dim)

        # Text decoder for cross-modal fusion
        text_embed_dim = config.text_decoder['embed_dim']
        output_dim = config.text_decoder['output_dim']
        self.text_proj = nn.Linear(self.text_dim, text_embed_dim)
        self.ln1_post = nn.LayerNorm(self.img_dim)
        self.ln2_post = nn.LayerNorm(text_embed_dim)
        self.text_decoder = MultimodalTransformer(
            width=text_embed_dim,
            layers=config.text_decoder['layers'],
            heads=config.text_decoder['nhead'],
            context_length=config.text_decoder['context_length'],
            mlp_ratio=config.text_decoder['mlp_ratio'],
            ls_init_value=None,
            act_layer=nn.GELU,
            output_dim=output_dim,
        )
        self.ln3_post = nn.LayerNorm(text_embed_dim)
        self.ln4_post = nn.LayerNorm(output_dim)

        # MultiPool after attain text-decoder output to attain mlp head
        # learnable query for attention pooling
        self.pool_attn_q = nn.Parameter(torch.zeros(1, output_dim))
        # projection head after concat of [mean, max, attn]
        self.multipool_proj = nn.Sequential(
            nn.LayerNorm(output_dim * 3),
            nn.Linear(output_dim * 3, output_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Initialize weights for all modules
        self.initialize_weights()

    def initialize_weights(self):

        # Initialize weights for all modules
        nn.init.normal_(self.pool_attn_q, std=0.02)
        for module in [self.image_decoder, self.text_decoder, self.aggregator,
                       self.text_to_latents, self.img_to_latents,
                       self.ln1_post, self.ln2_post, self.ln3_post, self.ln4_post,
                       self.text_kv_proj, self.text_kv_ln, self.img_kv_proj, self.img_kv_ln,
                       self.gate_kv, self.text2patch_attn, self.text_proj, self.multipool_proj]:
            self._init_weights(module)

        # Load pretrained text decoder weights if provided
        if self.config.text_decoder.get('pretrained_path', None) is not None:
            self._load_text_decoder_weights(self.config.text_decoder['pretrained_path'])

        # Optionally freeze text encoder parameters
        self._freeze_modules()

    def _init_weights(self, module):
        """
        Xavier initialization for weights and zero for biases.
        LayerNorm layers set weight=1 and bias=0.
        """
        for name, param in module.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        if isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def _freeze_modules(self):

        if self.config.freeze_text_encoder:
            self.text_encoder.requires_grad_(False)

        if self.config.freeze_text_decoder:
            self.text_decoder.requires_grad_(False)

        if self.config.freeze_image_encoder:
            self.image_encoder.requires_grad_(False)

        if self.config.freeze_image_decoder:
            self.image_decoder.requires_grad_(False)

        if self.config.freeze_aggregator:
            self.aggregator.requires_grad_(False)

        if not self.is_pretrain:
            self.text_to_patch_queries.requires_grad_(False)
            self.text2patch_attn.requires_grad_(False)

    def _load_text_decoder_weights(self, weight_path):
        """
        Load state dict for text decoder from checkpoint.
        Filters keys prefixed with 'module.text_decoder.'.
        """
        state = torch.load(weight_path, map_location='cpu')
        filtered = {
            k.replace('module.text_decoder.', ''): v
            for k, v in state.items()
            if k.startswith('module.text_decoder.') and 'text_projection' not in k
        }
        self.text_decoder.load_state_dict(filtered, strict=False)
        print(f"[Info] Loaded pretrained text_decoder weights from {weight_path}")

    def _visualization(self, img_path, img_224, raw_img_448, img_448, masks, decoder_preds):
        if self.config.is_attn_plot:
            visualize_attention_batch(
                self.image_encoder,
                img_224,
                img_path,
                save_dir=f"output/attn_results/",
            )

        if self.is_pretrain and self.config.is_mae_plot:
            visualize_mae_batch(self.image_decoder, img_path, raw_img_448, img_448, masks, decoder_preds,
                                save_dir="./output/mae_results")

    def encode_images(self, images, mask_ratio=0.0):
        """
        Encode images via CLIP vision encoder.
        Supports optional MAE-style random masking for masked reconstruction.

        Args:
            images: torch.Tensor, shape [B, C, H, W]
            mask_ratio: float, proportion of patches to mask

        Returns:
            outputs: torch.Tensor, shape [B*(1 or 4), N_patches+1, D_img]
            masks: mask tensor if masked, else None
            ids_restore: indices to restore masked patches
        """
        masks, ids_restore = None, None

        if mask_ratio > 0:
            # Process 448-version sub-images with masking
            img_448_embs = self.image_encoder.forward_embeddings(images)
            img_448_mask, masks, ids_restore = self.image_decoder.random_masking(
                img_448_embs[:, 1:, :], mask_ratio=mask_ratio
            )  # img_mask: [B, N_visible, D_img]
            img_448_embs = torch.cat([torch.unsqueeze(img_448_embs[:, 0, :], 1), img_448_mask],
                                     dim=1)  # [B*4, N_vis+1, D_img]
            outputs = self.image_encoder.forward_encoder(img_448_embs).last_hidden_state
        else:
            # Process 224-version to encode
            img_224_embs = self.image_encoder.forward_embeddings(images)
            outputs = self.image_encoder.forward_encoder(img_224_embs).last_hidden_state  # [B, N_patches+1, D_img]

        outputs = self.ln1_post(outputs)
        return outputs, masks, ids_restore

    def encode_texts(self, input_ids, attention_mask, return_dict):
        """
        Encode text input via CLIP text encoder and remove EOS token from token embeddings.

        Returns:
            text_cls: [B, D_text], pooled EOS embedding
            text_embeds_76: [B, L-1, D_text], token embeddings without EOS
            new_mask: [B, L-1], attention mask without EOS position
        """
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict
        )
        text_embeds = text_outputs.last_hidden_state  # [B, L, D_text]
        # text_cls = text_outputs.pooler_output  # EOS token [B, D_text]

        # Identify EOS token position per sequence
        B, L, D = text_embeds.shape
        eos_positions = (attention_mask.long().sum(dim=1) - 1).clamp(min=0)
        text_cls = text_embeds[torch.arange(B, device=text_embeds.device), eos_positions]  # [B, D]

        # Ensure index in range
        base_indices = torch.arange(L, device=text_embeds.device).unsqueeze(0).expand(B, -1)
        mask = torch.ones_like(base_indices, dtype=torch.bool)
        mask[torch.arange(B), eos_positions] = False  # mark EOS as False
        indices = base_indices[mask].view(B, L - 1)  # [B, L-1]
        indices = indices.unsqueeze(2).expand(-1, -1, D)  # [B, L-1, D]
        text_embeds_wo_eos = torch.gather(text_embeds, dim=1, index=indices)  # [B, L-1, D]

        # update attention mask (exclude eos)
        new_mask = attention_mask.clone()
        new_mask[torch.arange(B), eos_positions] = 0
        new_mask = new_mask[:, :L - 1]

        return text_cls, text_embeds_wo_eos, new_mask

    def forward_image_decoder(self, outputs_448, full_448_tokens, text_tokens, ids_restore):
        """
        MAE-style decoding with text-conditioned reconstruction
        Args:
            outputs_448: Encoded image features [B*4, N_vis+1, D_img]
            full_448_tokens: Masked image features with padding masking B*4, N_patches, D_text]
            text_tokens: Encoded text features [B*4, L, D_text]
            ids_restore: Restoration indices for masked patches [B*4, N]
        Returns:
            preds: Reconstructed patches [B*4, N_patches, patch_size**2 * 3]
            features: Intermediate features [B*4, N_patches, D]
            align_features: aligned features [B*4, D]
        """
        B = text_tokens.size(0)
        BV = outputs_448.size(0)

        # gate mechanism to fusion text and image feature in order to decrease recon loss
        text_tokens_proj = self.text_kv_ln(self.text_kv_proj(text_tokens))  # [B, L_text, dec_dim]
        text_tokens_proj_bv = text_tokens_proj.repeat_interleave(4, dim=0)  # [B*4, L_text, dec_dim]
        img_tokens_proj = self.img_kv_ln(self.img_kv_proj(full_448_tokens))  # [BV, N_patches, dec_dim]

        # multihead attn: queries as Q, keys/values are text tokens, to attain new L_text with same L_img
        queries = self.text_to_patch_queries.expand(BV, -1, -1)  # [B*4, L_img, D]
        text_kv_for_merge, attn_w = self.text2patch_attn(queries, text_tokens_proj_bv,
                                                         text_tokens_proj_bv)  # [Bv, L_img, D]
        kv_combined, gate_alpha = self.gate_kv(img_tokens_proj,
                                               text_kv_for_merge)  # [B*4, L_img, dec_dim], gate_alpha: [n_heads]

        # Cross-attention decoding
        decoder_preds, features, align_features = self.image_decoder(
            image_features=outputs_448,  # [B*4, L, D_img]
            text_embeds=kv_combined,  # .repeat_interleave(4, dim=0),  # [B*4, L, D_text]
            ids_restore=ids_restore,
        )  # decoder_preds: [B*4, N_patches, D], clip_features: [B*4, D, 1]
        align_features = align_features.squeeze(-1).view(B, 4, -1)
        align_features = self.aggregator(align_features)  # [B, D]

        return decoder_preds, features, align_features

    def post_processing(self, text_decoder_out, attention_mask):
        # text_decoder_out: [B, L, D], attention_mask: [B, L] (1 for real tokens)
        token_embs = text_decoder_out  # [B, L, D]
        mask = attention_mask  # [B, L]
        mask_bool = mask.to(dtype=torch.bool)  # ensure bool

        # mean pool (masked)
        mask_f = mask.unsqueeze(-1)  # [B, L, 1]
        sum_emb = (token_embs * mask_f).sum(dim=1)  # [B, D]
        counts = mask_f.sum(dim=1).clamp(min=1.0)  # [B, 1]
        mean_pool = sum_emb / counts  # [B, D]

        # max pool (masked) -- set masked positions to large negative
        neg_inf = torch.finfo(token_embs.dtype).min
        masked_for_max = token_embs.masked_fill(~mask_bool.unsqueeze(-1), neg_inf)
        max_pool = masked_for_max.max(dim=1).values  # [B, D]

        # attention pooling using learnable query
        # pool_attn_q: [1, D]  -> expand to [B, 1, D]
        q = self.pool_attn_q.unsqueeze(0).expand(token_embs.size(0), -1, -1)  # [B,1,D]
        attn_scores = torch.bmm(q, token_embs.transpose(1, 2))  # [B,1,L]
        attn_scores = attn_scores.masked_fill(~mask_bool.unsqueeze(1), neg_inf)
        attn_w = torch.softmax(attn_scores, dim=-1)  # [B,1,L]
        attn_pool = torch.bmm(attn_w, token_embs).squeeze(1)  # [B, D]

        # concat
        pooled_concat = torch.cat([mean_pool, max_pool, attn_pool], dim=-1)  # [B, 3D]
        pooled = self.multipool_proj(pooled_concat)  # [B, hidden_dim]

        return pooled

    def forward(self, img_path, img_224, raw_img_448, img_448, input_ids, attention_mask):
        """
        Forward pass through multimodal architecture
          1) Encode images (single and multi-crop)
          2) Encode texts
          3) MAE reconstruction (training only)
          4) Cross-modal fusion & classification

        Args:
            img_224: Input image (224x224) [B, C, 224, 224]
            img_448: Input image (448x448) [B*4, C, 224, 224]
            input_ids: Text token IDs [B, L]
            attention_mask: Text attention mask [B, L]
        Returns:
            dict of:
              img_latents: [B, D_latent]
              text_latents: [B, D_latent]
              masks, image_decoder_preds,
              image_decoder_latents, pooled outputs
        """
        # --- Encode primary image (224x224) ---
        outputs_224, _, _ = self.encode_images(img_224)  # [B, N_patches+1, D_img]
        img_224_embs, img_224_tokens = outputs_224[:, 0, :], outputs_224[:, 1:, :]

        # if self.training:
        # --- Encode multi-view images (448x448) for MAE ---
        outputs_448, masks, ids_restore = self.encode_images(
            img_448,
            mask_ratio=self.config.image_decoder[
                'mask_ratio'] if self.is_pretrain else 0.0)  # [B*4, N_visible+1, D_img]
        img_448_embs, img_448_tokens = outputs_448[:, 0, :], outputs_448[:, 1:, :]

        # Build padding mask to masked-img -> same patches with full-img
        B = img_224_embs.size(0)
        N_patches = img_224_tokens.size(1)
        BV, N_visible, D = img_448_tokens.shape
        if self.is_pretrain:
            mask_tokens_img = self.image_decoder.mask_token_img.repeat(BV, N_patches - N_visible,
                                                                       1)  # [B*4, N_visible, D]
            x_unshuffled = torch.cat([img_448_tokens, mask_tokens_img], dim=1)  # [B*4, N_patches, D]
            full_448_tokens = torch.gather(
                x_unshuffled,
                dim=1,
                index=ids_restore.unsqueeze(-1).expand(-1, -1, D)
            )  # [B*4, N_patches, D_img]
            prefix = torch.zeros(B, N_patches, dtype=torch.bool, device=img_448_tokens.device)
            img_key_padding = torch.cat([prefix, masks.bool().view(B, -1)], dim=1)  # [B, N_patches+4*N_patches]
        else:
            full_448_tokens = img_448_tokens  # [B*4, N_patches, D_img]
            img_key_padding = torch.zeros((B, 5 * N_patches), dtype=torch.bool,
                                          device=img_448_tokens.device)  # [B, N_patches+4*N_patches]

        # --- Encode text inputs ---
        text_embs, text_tokens, attention_mask = self.encode_texts(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )  # text_embs: [B, D_text], text_tokens: [B, L-1, D_text]

        # Project to latent spaces
        text_latents = self.text_to_latents(text_embs)  # [B, D_text] -> [B, D_lat]
        img_224_latents = self.img_to_latents(img_224_embs)  # [B, D_img] -> [B, D_lat]

        # --- MAE Reconstruction during training ---
        decoder_preds, align_features = None, None
        if self.is_pretrain:
            decoder_preds, features, align_features = (
                self.forward_image_decoder(outputs_448, full_448_tokens, text_tokens, ids_restore))

        # --- Cross-Modal Fusion & Pooling ---
        # Concatenate tokens from 224 and 448 views
        image_embs_for_decoder = torch.cat([img_224_tokens, full_448_tokens.reshape(B, -1, D)], dim=1)
        text_decoder_out = self.text_decoder(
            self.ln2_post(image_embs_for_decoder),
            self.ln3_post(self.text_proj(text_tokens)),  # [B, L-1, D_text] -> [B, L-1, D]
            # attn_mask=attention_mask,
            image_key_padding_mask=img_key_padding
        )  # [B, L-1, D]

        # Masked pooling over text positions to attain mlp head
        pooled = self.post_processing(text_decoder_out, attention_mask)
        pooled = self.ln4_post(pooled)

        self._visualization(img_path, img_224, raw_img_448, img_448, masks, decoder_preds)

        return {
            "img_224_latents": img_224_latents,  # CLS token for original img
            "text_latents": text_latents,  # CLS token for text
            "masks": masks,  # Mask map for mae
            "img_dec_preds": decoder_preds,  # Reconstructed patches features
            "img_dec_latents": align_features,  # CLS token for reconstructed patch
            "pooled": pooled,  # class features [B, D]
        }


class MultimodalModelForClassification(nn.Module):
    """Wrapper for multimodal sentiment model with classification head"""

    def __init__(self, is_pretrain: bool, cfg: ModelConfig):
        super().__init__()
        self.config = cfg
        self.base_model = MultimodalModel(is_pretrain, cfg)
        self.temperature = nn.Parameter(torch.Tensor([1.]))
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / 0.07))

        # Classification head (multimodal sentiment analysis)
        intput_dim = self.config.text_decoder['output_dim']
        self.proj1 = MLPLayer(intput_dim, intput_dim, intput_dim)
        self.proj2 = MLPLayer(intput_dim, intput_dim // 2, self.config.num_labels)

        self.base_model._init_weights(self.proj1)
        self.base_model._init_weights(self.proj2)

        if is_pretrain:
            self.proj1.requires_grad_(False)
            self.proj2.requires_grad_(False)

    def forward(
            self,
            img_path: torch.Tensor,
            img_224: torch.Tensor,
            img_448: torch.Tensor,
            raw_img_448: torch.tensor,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            labels: torch.Tensor
    ):
        # Forward pass through base model
        outputs = self.base_model(
            img_path=img_path,
            img_224=img_224,
            raw_img_448=raw_img_448,
            img_448=img_448,
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Forward for Classification
        proj1 = self.proj1(outputs['pooled'])
        proj1 += outputs['pooled']
        logits = self.proj2(proj1)  # [B, num_labels]
        if self.training or self.base_model.is_pretrain:
            return {
                **outputs,
                "logits": logits,
                "temperature": self.temperature,
                "logit_scale": self.logit_scale,
            }
        return {
            "img_path": img_path,
            "logits": logits
        }

    @classmethod
    def from_pretrained(cls, cfg: ModelConfig, is_pretrain: bool, mode: str, save_directory: str):
        # cfg: your Config object to re-create base model
        model = cls(is_pretrain, cfg)

        if save_directory.endswith('.safetensors'):
            state_dict = {}
            with safe_open(save_directory, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if not is_pretrain and mode == 'train' and ('proj2.mlp' in key):
                        pass
                    else:
                        state_dict[key] = f.get_tensor(key)
        else:
            state_dict = torch.load(save_directory, map_location="cpu")

        model.load_state_dict(state_dict, strict=False)
        return model
