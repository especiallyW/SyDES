import os

import cv2
import numpy as np
from torchvision import transforms

from models import *


def transform_image(raw_img, mean, std):
    """Transform image tensor: transpose, denormalize, clip, transpose back"""
    raw_img = torch.einsum('chw->hwc', raw_img)
    full_img = torch.clip((raw_img * std + mean), 0, 1)
    return torch.einsum('hwc->chw', full_img)


def visualize_mae_batch(model, img_paths, raw_img_batch, img_patches_batch, masks_batch,
                        decoder_preds_batch, save_dir="./mae_visualizations", num_samples=None):
    """
    Batch visualization of MAE results: original, masked and reconstructed images

    Args:
        raw_img_batch: Original 448x448 images [B, 3, 448, 448]
        img_patches_batch: Normalized image patches [B*4, 3, 224, 224]
        masks_batch: Mask information [B*4, N_patches]
        decoder_preds_batch: Decoder predictions [B*4, N_patches, D]
    """
    # Create output directories
    for subdir in ["original", "masked", "reconstructed", "pasted"]:
        os.makedirs(f"{save_dir}/{subdir}", exist_ok=True)

    B = raw_img_batch.size(0)
    num_samples = min(num_samples or B, B)

    with torch.no_grad():
        for i in range(num_samples):
            # Process each sample individually
            visualize_mae_single(
                model,
                img_paths[i],
                raw_img_batch[i],
                img_patches_batch[i * 4:(i + 1) * 4],
                masks_batch[i * 4:(i + 1) * 4],
                decoder_preds_batch[i * 4:(i + 1) * 4],
                save_dir
            )


def visualize_mae_single(model, filename, raw_img, img_patches, masks, decoder_preds, save_dir):
    """Visualize MAE results for single sample"""

    # def visualize_mae_results_single(model, filename, raw_img_448, img_448_patches, masks, decoder_preds,
    #                                  save_dir="./mae_visualizations"):
    def create_masked_images(model, patches, masks, preds):
        """Generate masked, reconstructed and pasted images"""
        is_sub = True if len(patches.shape) == 4 else False
        if not is_sub:
            patches = patches.unsqueeze(dim=0)
            decoder_preds = preds.unsqueeze(dim=0)

        # visualize the mask
        mask = masks.detach().unsqueeze(-1).repeat(1, 1, 16 ** 2 * 3)  # (N, H*W, p*p*3)
        mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping

        # masked image
        masked_img = patches * (1 - mask)
        # reconstructed image
        reconstructed_img = model.unpatchify(decoder_preds)
        # MAE reconstruction pasted with visible patches
        pasted_img = patches * (1 - mask) + decoder_preds * mask

        if is_sub:
            masked_img = stitch_images(masked_img)
            reconstructed_img = stitch_images(reconstructed_img)
            pasted_img = stitch_images(pasted_img)

        return masked_img.squeeze(), reconstructed_img.squeeze(), pasted_img.squeeze()

    def stitch_images(images):
        """Combine 4 sub-images into full 448x448 image"""
        top = torch.cat([images[0], images[1]], dim=2)
        bottom = torch.cat([images[2], images[3]], dim=2)
        return torch.cat([top, bottom], dim=1)

    # Normalization parameters
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device='cuda')
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device='cuda')

    # Save original image
    orig_img = transform_image(raw_img, mean, std)
    orig_pil = transforms.ToPILImage()(orig_img)
    orig_pil.save(f"{save_dir}/original_448_075/{filename}.png")

    # Generate and save processed images
    masked_img, reconstructed_img, pasted_img = create_masked_images(model, img_patches, masks, decoder_preds)
    masked_img = transform_image(masked_img, mean, std)
    masked_pil = transforms.ToPILImage()(masked_img)
    masked_pil.save(f"{save_dir}/masked_448_075/{filename}.png")

    reconstructed_img = transform_image(reconstructed_img, mean, std)
    reconstructed_pil = transforms.ToPILImage()(reconstructed_img)
    reconstructed_pil.save(f"{save_dir}/reconstructed_448_075/{filename}.png")

    pasted_img = transform_image(pasted_img, mean, std)
    pasted_pil = transforms.ToPILImage()(pasted_img)
    pasted_pil.save(f"{save_dir}/pasted_448_075/{filename}.png")


def visualize_attention_batch(model, img_batch, img_paths, save_dir, num_samples=None):
    """Batch visualization of attention heatmaps"""
    # Create output directories
    for subdir in ["original", "attention_heatmap"]:
        os.makedirs(f"{save_dir}/{subdir}", exist_ok=True)

    # Normalization parameters
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device='cuda')
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device='cuda')

    B = img_batch.size(0)
    num_samples = min(num_samples or B, B)

    with torch.no_grad():
        # Batch process attention maps
        embeddings = model.forward_embeddings(img_batch)
        outputs = model.forward_encoder(embeddings, output_attentions=True)

        # Extract and average attention weights
        attn_weights = outputs.attentions[-1].mean(dim=1)
        cls_attention = attn_weights[:, 1:, 1:].mean(dim=1)

        # Reshape and interpolate attention maps
        grid_size = int(np.sqrt(cls_attention.shape[-1]))
        attn_maps = cls_attention.reshape(B, grid_size, grid_size).unsqueeze(1)
        attn_maps = F.interpolate(attn_maps, size=224, mode='bilinear')

    # Process each sample
    for i in range(num_samples):
        filename = img_paths[i]
        attn_map = attn_maps[i].squeeze().cpu().numpy()

        # Normalize and colorize attention map
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-5)
        heatmap = cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255

        # Overlay heatmap on original image
        orig_img = transform_image(img_batch[i], mean, std).permute(1, 2, 0).cpu().numpy()
        alpha = 0.5
        overlay = orig_img * (1 - alpha) + heatmap * alpha

        # Save results
        orig_pil = transforms.ToPILImage()(overlay)
        orig_pil.save(f"{save_dir}/original/{filename}.png")
        overlayed_pil = transforms.ToPILImage()(np.uint8(overlay * 255))
        overlayed_pil.save(f"{save_dir}/attention_heatmap/{filename}.png")
