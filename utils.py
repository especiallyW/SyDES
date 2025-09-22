import dataclasses
import json
import os

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from configs.config import DataConfig
from datasets.mseddataset import MSEDDataset


# --- Utility functions ---
def print_trainable_params(model: nn.Module):
    """
    Print which parameters are frozen vs trainable and summary.
    """
    total = 0
    trainable = 0
    print("Parameter status:")
    for name, param in model.named_parameters():
        num = param.numel()
        total += num
        if param.requires_grad:
            trainable += num
            status = "trainable"
        else:
            status = "frozen"
        print(f"  {name}: {status}, shape={tuple(param.shape)}")
    print(f"Total params: {total}")
    print(f"Trainable params: {trainable} ({trainable / total:.2%})")


def write_logging(logging_dir, cfg):
    """Saving config yaml to json file"""
    os.makedirs(logging_dir, exist_ok=True)
    with open(f"{logging_dir}/config.json", "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)


def multimodal_collator(batch):
    """Collate function for multimodal batch processing"""
    img_448 = torch.stack([item['img_448'] for item in batch])
    B, N, C, H, W = img_448.shape
    img_448 = img_448.view(B * N, C, H, W)  # [B*4, C, H, W]
    return {
        'img_path': torch.stack([torch.tensor(item['img_path']) for item in batch]),
        'img_224': torch.stack([torch.tensor(item['img_224']) for item in batch]),
        'img_448': img_448,
        'raw_img_448': torch.stack([item['raw_img_448'].clone().detach() for item in batch]),
        'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in batch]),
        'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in batch]),
        'labels': torch.tensor([item['labels'] for item in batch], dtype=torch.long)
    }


def compute_metrics(p):
    """Calculate classification metrics"""
    preds = p.predictions[-1].argmax(-1)
    labels = p.label_ids
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, average='macro', zero_division=0),
        'recall': recall_score(labels, preds, average='macro', zero_division=0),
        'f1_micro': f1_score(labels, preds, average='micro'),
        'f1_macro': f1_score(labels, preds, average='macro'),
        'f1_weighted': f1_score(labels, preds, average='weighted')
    }


# Prepare datasets
def load_multimodal_dataset(cfg: DataConfig, split: str, img_process, tokenizer):
    if cfg.data_name in ['MVSA-Single', 'MVSA-Multiple', 'MSED']:
        return MSEDDataset(
            root_path=cfg.root_path,
            img_process=img_process,
            tokenizer=tokenizer,
            train_type=split,
            label_type=cfg.label_type
        )
    else:
        pass
