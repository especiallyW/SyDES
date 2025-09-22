from dataclasses import dataclass
from typing import Optional


# --- Configuration dataclasses ---
@dataclass
class TextDecoderConfig:
    """Configuration for text decoder component"""
    pretrained_path: str  # Path to pretrained weights or model identifier
    layers: int  # Number of transformer layers
    nhead: int  # Number of attention heads
    embed_dim: int  # Embedding dimension size
    mlp_ratio: float  # MLP expansion ratio
    context_length: int  # Maximum input sequence length
    output_dim: int  # Dimension of output embeddings


@dataclass
class ImageDecoderConfig:
    """Configuration for image decoder component"""
    layers: int  # Number of transformer layers
    nhead: int  # Number of attention heads
    embed_dim: int  # Embedding dimension size
    mask_ratio: float  # Ratio of masked image patches
    mlp_ratio: float  # MLP expansion ratio


@dataclass
class ModelConfig:
    """Main model configuration parameters"""
    image_encoder_path: str  # Pretrained image encoder identifier
    text_encoder_path: str  # Pretrained text encoder identifier
    image_decoder: ImageDecoderConfig  # Image decoder settings
    text_decoder: TextDecoderConfig  # Text decoder settings
    num_labels: int  # Number of output classes
    latent_dim: int  # Dimension of latent space
    freeze_text_encoder: bool  # Freeze text encoder weights
    freeze_text_decoder: bool  # Freeze text decoder weights
    freeze_image_encoder: bool  # Freeze image encoder weights
    freeze_image_decoder: bool  # Freeze image decoder weights
    freeze_aggregator: bool  # Freeze feature aggregator weights
    is_mae_plot: bool = False  # Enable masked autoencoder visualization
    is_attn_plot: bool = False  # Enable attention visualization


@dataclass
class DataConfig:
    """Dataset configuration parameters"""
    data_name: str  # Dataset identifier
    root_path: str  # Root directory path
    label_type: str  # Type of labels (e.g., sentiment, emotion)


@dataclass
class LossWeightConfig:
    """Loss weighting coefficients"""
    itc: float  # Image-text contrastive loss weight
    cs: float  # Consistency loss weight
    cf: float  # Cross-modal feature loss weight
    recon: float  # Reconstruction loss weight
    cls: float  # Classification loss weight


@dataclass
class LRSchedulerConfig:
    """Learning rate configuration per component"""
    image_encoder: float = 5e-6  # Image encoder learning rate
    image_decoder: float = 1e-4  # Image decoder learning rate
    text_decoder: float = 1e-4  # Text decoder learning rate
    classifier: float = 1e-4  # Classifier learning rate
    others: float = 1e-4  # Other components learning rate


@dataclass
class TrainingConfig:
    """Training process configuration"""
    stage: str  # Training stage (e.g., pretrain, finetune)
    loss_weight: LossWeightConfig  # Loss weighting configuration
    output_dir: str  # Output directory for checkpoints
    logging_dir: Optional[str] = None  # Logging directory
    result_dir: Optional[str] = None  # Results directory
    cls_type: str = "cross_entropy"  # Classification loss type
    num_epochs: int = 10  # Total training epochs
    batch_size: int = 32  # Training batch size
    eval_batch_size: Optional[int] = None  # Evaluation batch size
    learning_rate_map: LRSchedulerConfig = None  # Component-specific learning rates
    lr_scheduler_type: str = "cosine"  # Learning rate scheduler type
    warmup_ratio: float = 0.1  # Warmup steps ratio
    weight_decay: float = 0.01  # Weight decay coefficient
    gradient_accumulation: int = 2  # Gradient accumulation steps
    max_grad_norm: float = 1.0  # Gradient clipping norm
    logging_steps: int = 50  # Logging frequency
    logging_strategy: str = "steps"  # Logging strategy
    eval_strategy: str = "steps"  # Evaluation strategy
    save_strategy: str = "steps"  # Checkpoint saving strategy
    eval_steps: Optional[int] = None  # Evaluation frequency (if step-based)
    save_steps: Optional[int] = None  # Saving frequency (if step-based)
    save_total_limit: int = 3  # Maximum checkpoints to keep
    load_best_model_at_end: bool = True  # Load best model at end of training
    metric_for_best_model: str = "f1_weighted"  # Metric for model selection
    greater_is_better: bool = True  # Whether higher metric is better
    fp16: bool = True  # Use mixed-precision training
    seed: int = 42  # Random seed
    max_length: int = 50  # Maximum text sequence length
    image_size: int = 224  # Input image resolution


@dataclass
class Config:
    """Top-level configuration container"""
    model: ModelConfig  # Model architecture configuration
    data: DataConfig  # Dataset configuration
    training: TrainingConfig  # Training procedure configuration
