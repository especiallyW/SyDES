import torch
import torch.nn.functional as F
from torch import einsum, nn
from transformers import Trainer


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        alpha: None or list/array of shape [C] for class weights
        gamma: focusing parameter
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        # logits: [B, C]
        ce_loss = F.cross_entropy(logits, targets, reduction='none')  # per-sample CE
        pt = torch.exp(-ce_loss)  # p_t
        loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)
            at = self.alpha.gather(0, targets)
            loss = at * loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SyMSATrainer(Trainer):
    """
    Custom Trainer for the SyMSA model, integrating multiple loss components:
      - Image-Text Contrastive (ITC)
      - Similarity & Contrastive KL (CS & CF)
      - Masked Reconstruction (MAE)
      - Classification (CLS)
    Loss weights are provided via `loss_weights` dict.
    """

    def __init__(self, *args, loss_weights, cls_type, **kwargs):
        super().__init__(*args, **kwargs)
        # Weights for each loss term
        self.loss_weight = loss_weights
        self.current_lr, self.current_grad_norm = 0.0, 0.0
        if cls_type == 'focal_loss':
            self.cls = FocalLoss(alpha=[0.115, 0.658, 0.227])
        else:
            self.cls = F.cross_entropy

    def patchify(self, imgs: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def loss_cs_cf(
            self,
            img_latents: torch.Tensor,
            text_latents: torch.Tensor,
            decoder_latents: torch.Tensor,
            logit_scale: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        """
        Similarity (MSE) loss between teacher (img) and student (MAE),
        plus cross-modal KL divergence regularization.
        Args:
            img_latents: Tensor of shape (B, D)
            text_latents: Tensor of shape (B, D)
            decoder_latents: Tensor of shape (B, *) before reshape
            logit_scale: Scalar parameter for scaling logits
        Returns:
            similarity_loss, kl_distance_loss
        """
        # Normalize features
        # img_norm = F.normalize(img_latents, dim=1, eps=1e-12)
        # text_norm = F.normalize(text_latents, dim=1, eps=1e-12)
        img_norm = img_latents
        text_norm = text_latents

        # Flatten decoder latents
        decoder_flat = decoder_latents.reshape(len(img_latents), -1)
        decoder_norm = F.normalize(decoder_flat, dim=1, eps=1e-12)

        # # MSE similarity loss
        # sim_loss = 1.0 - F.cosine_similarity(decoder_norm, img_norm, dim=-1).mean()
        sim_loss = F.mse_loss(decoder_norm, img_norm)

        # Scaled logits for KL
        logits_clip = logit_scale * (img_norm @ text_norm.t())
        logits_mae = logit_scale * (decoder_norm @ text_norm.t())

        # KLDiv + entropy regularization
        kl = F.kl_div(
            F.log_softmax(logits_mae, dim=-1),
            F.softmax(logits_clip, dim=-1),
            reduction='batchmean'
        )
        entropy_term = - (F.softmax(logits_mae, dim=0) * F.log_softmax(logits_mae, dim=0)).mean()

        return sim_loss, kl + entropy_term

    def loss_itc(self, img_latents, text_latents, temperature):
        """
        Image-Text contrastive loss (Symmetric cross entropy).
        Args:
            img_latents: Tensor of shape (B, D)
            text_latents: Tensor of shape (B, D)
            temperature: Learned temperature parameter
        Returns:
            contrastive_loss
        """
        batch = img_latents.size(0)
        # similarity matrix
        sim = einsum('i d, j d -> i j', text_latents, img_latents)
        sim = sim * temperature.exp()
        labels = torch.arange(batch, device=sim.device)
        ce = F.cross_entropy
        return 0.5 * (ce(sim, labels) + ce(sim.t(), labels))

    def loss_cls(self, preds, targets):
        """
        Standard cross-entropy classification loss.
        """
        return self.cls(preds, targets)

    def loss_mae(
            self,
            imgs_448: torch.Tensor,
            decoder_preds: torch.Tensor,
            masks: torch.Tensor,
            patch_size: int = 16
    ) -> torch.Tensor:
        """
        Pixel reconstruction loss for masked patches (MAE).
        Args:
            imgs_448: Original images (N, 3, H, W)
            decoder_preds: Predicted patches (N, L, p*p*3)
            masks: Binary mask (N, L), 1 indicates masked
        Returns:
            recon_loss
        """
        target_patches = self.patchify(imgs_448, patch_size)

        # Original MSE:
        loss = (decoder_preds - target_patches) ** 2
        per_patch = loss.mean(dim=-1)
        # average only masked patches
        return (per_patch * masks).sum() / masks.sum()

        # With SmoothL1:
        # recon_criterion = torch.nn.SmoothL1Loss(reduction='none')
        # per_elem = recon_criterion(decoder_preds, target_patches)  # [B, N, 768]
        # per_patch = per_elem.mean(dim=-1)  # [B, N]
        # recon_loss = (per_patch * masks).sum() / (masks.sum() + 1e-12)

        # # SyCoca L1 loss:
        # per_patch = masks * torch.abs(target_patches - decoder_preds).sum(dim=-1)
        # recon_loss = per_patch.sum()
        # return recon_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override compute_loss to combine multiple losses.
        total = w_itc * ITC + w_cs * CS + w_cf * CF + w_recon * MAE + w_cls * CLS
        """
        # Forward pass
        outputs = model(**inputs)

        img_448 = inputs.get('img_448')
        labels = inputs.get('labels')
        logits = outputs.get('logits')

        rec_loss = torch.tensor(0.0, device=logits.device)
        itc_loss = torch.tensor(0.0, device=logits.device)
        cs_loss = torch.tensor(0.0, device=logits.device)
        cf_loss = torch.tensor(0.0, device=logits.device)

        # Classification loss always applied
        cls_loss = self.loss_cls(logits, labels)

        if model.base_model.is_pretrain or model.training:  # true = pretrain, false = finetuning
            temp = outputs['temperature']
            masks = outputs['masks']
            logit_scale = outputs['logit_scale']
            text_latents = outputs['text_latents']
            img_latents = outputs['img_224_latents']
            dec_latents = outputs['img_dec_latents']
            dec_preds = outputs['img_dec_preds']

            # Compute individual losses
            itc_loss = self.loss_itc(img_latents, text_latents, temp)
            if dec_latents is not None and masks is not None:
                cs_loss, cf_loss = self.loss_cs_cf(img_latents, text_latents, dec_latents, logit_scale)
                rec_loss = self.loss_mae(img_448, dec_preds, masks)

            # Aggregate weighted losses (normalized combine)
            total_loss = self.loss_weight['itc'] * itc_loss \
                         + self.loss_weight['cs'] * cs_loss \
                         + self.loss_weight['cf'] * cf_loss \
                         + self.loss_weight['recon'] * rec_loss \
                         + self.loss_weight['cls'] * cls_loss
        else:
            total_loss = self.loss_weight['cls'] * cls_loss

        if ((model.base_model.is_pretrain and model.training)
                or (not model.base_model.is_pretrain and model.training)):
            # Unpack training-specific outputs
            # Log intermediate metrics at each logging step
            step = self.state.global_step
            if step % self.args.logging_steps == 0:
                # step will default to current global_step
                self.log({
                    'loss_itc': self.loss_weight['itc'] * itc_loss.item(),
                    'loss_cs': self.loss_weight['cs'] * cs_loss.item(),
                    'loss_cf': self.loss_weight['cf'] * cf_loss.item(),
                    'loss_rec': self.loss_weight['recon'] * rec_loss.item(),
                    'loss_cls': self.loss_weight['cls'] * cls_loss.item(),
                    'loss_total': total_loss.item(),
                    'step': step,
                    'epoch': self.state.epoch
                })

        if model.base_model.is_pretrain and not model.training:
            self.log({
                "eval_loss_itc": self.loss_weight['itc'] * itc_loss.detach().cpu().item(),
                "eval_cs_loss": self.loss_weight['cs'] * cs_loss.detach().cpu().item(),
                "eval_cf_loss": self.loss_weight['cf'] * cf_loss.detach().cpu().item(),
                "eval_loss_rec": self.loss_weight['recon'] * rec_loss.detach().cpu().item(),
                "eval_loss_total": total_loss.detach().cpu().item()
            })

        if return_outputs:
            return total_loss, outputs
        return total_loss
