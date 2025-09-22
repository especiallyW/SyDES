import argparse
from os.path import dirname

import math
import mmcv
import pandas as pd
import yaml
from transformers import TrainingArguments, CLIPImageProcessor, CLIPTokenizer

from configs.config import Config, ModelConfig, TrainingConfig
from models.sydes import MultimodalModelForClassification
from trainer import SyMSATrainer
from utils import *


def load_config(path: str) -> Config:
    """Load YAML config and construct nested Config objects"""
    with open(path, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    model_cfg = ModelConfig(**cfg_dict.get('model', {}))
    data_cfg = DataConfig(**cfg_dict.get('data', {}))
    training_cfg = TrainingConfig(**cfg_dict.get('training', {}))

    # default eval_batch_size if not set
    if training_cfg.eval_batch_size is None:
        training_cfg.eval_batch_size = training_cfg.batch_size * 2
    # set default logging_dir if not set
    if not training_cfg.logging_dir:
        training_cfg.logging_dir = f"{training_cfg.output_dir}/logs"
    # default eval_steps and save_steps
    if training_cfg.eval_steps is None:
        training_cfg.eval_steps = training_cfg.logging_steps
    if training_cfg.save_steps is None:
        training_cfg.save_steps = training_cfg.logging_steps

    return Config(model=model_cfg, data=data_cfg, training=training_cfg)


def set_lr_scheduler(cfg, model, train_ds_len):
    from transformers import get_cosine_schedule_with_warmup
    from torch.optim import AdamW

    # --- build lr_map from cfg (supports either single float or dict in config) ---
    lr_map = {}
    if isinstance(cfg.training.learning_rate_map, dict):
        lr_map = cfg.training.learning_rate_map
    else:
        # single lr provided -> use as default for all groups
        lr_map = {"others": float(cfg.training.learning_rate_map)}

    # Helper: build param groups by substring match
    def build_param_groups_from_model(model, lr_map, weight_decay):
        matched = set()
        all_named = list(model.named_parameters())
        groups = []
        # match explicit keys first (order matters: more specific first)
        for key, lr in lr_map.items():
            if key == "others":
                continue
            params = [p for n, p in all_named if (key in n) and p.requires_grad]
            if params:
                groups.append({"params": params, "lr": float(lr), "weight_decay": weight_decay})
                matched.update({id(p) for p in params})
        # remaining
        remaining = [p for n, p in all_named if p.requires_grad and id(p) not in matched]
        if remaining:
            groups.append({"params": remaining, "lr": float(lr_map.get("others", 1e-4)), "weight_decay": weight_decay})
        return groups

    # Build param groups
    param_groups = build_param_groups_from_model(model, lr_map, cfg.training.weight_decay)

    # Create optimizer
    optimizer = AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)

    # Compute total training steps for scheduler
    # train_ds_len = len(train_dataset)
    effective_batch = cfg.training.batch_size * cfg.training.gradient_accumulation
    steps_per_epoch = math.ceil(train_ds_len / effective_batch)
    total_steps = steps_per_epoch * cfg.training.num_epochs
    warmup_steps = int(
        cfg.training.warmup_ratio * total_steps) if cfg.training.warmup_ratio is not None else cfg.training.warmup_steps

    # Cosine scheduler with warmup (transformers util)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                   num_training_steps=total_steps)

    return optimizer, lr_scheduler


def main(args):
    # Load nested configuration
    cfg = load_config(args.config)
    is_pretrain = True if cfg.training.stage == 'pretrain' else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set random seed
    torch.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed_all(cfg.training.seed)

    # Prepare data loaders
    img_process = CLIPImageProcessor()
    tokenizer = CLIPTokenizer.from_pretrained(cfg.model.text_encoder_path)
    # tokenizer = BertTokenizer.from_pretrained(cfg.model.text_encoder_path)
    train_dataset = load_multimodal_dataset(cfg.data, 'train', img_process, tokenizer)
    val_dataset = load_multimodal_dataset(cfg.data, 'dev', img_process, tokenizer)
    test_dataset = load_multimodal_dataset(cfg.data, 'dev' if is_pretrain else 'test', img_process, tokenizer)

    # Initialize model using model config
    if is_pretrain:
        model = MultimodalModelForClassification(is_pretrain, cfg.model)
    else:
        # load best checkpoint
        model = MultimodalModelForClassification.from_pretrained(cfg.model, is_pretrain, args.mode, args.model_path)
        model.to(device)

    # Configure training
    print_trainable_params(model)
    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        logging_dir=cfg.training.logging_dir,

        num_train_epochs=cfg.training.num_epochs,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation,
        max_grad_norm=cfg.training.max_grad_norm,

        logging_steps=cfg.training.logging_steps,
        eval_strategy=cfg.training.eval_strategy,
        eval_steps=cfg.training.eval_steps,

        save_strategy=cfg.training.save_strategy,
        save_steps=cfg.training.save_steps,

        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model=cfg.training.metric_for_best_model,
        greater_is_better=cfg.training.greater_is_better,

        fp16=cfg.training.fp16,
        seed=cfg.training.seed,
        save_total_limit=cfg.training.save_total_limit,

        remove_unused_columns=False,
        # report_to="tensorboard"
    )

    # Choose mode
    optimizer, lr_scheduler = set_lr_scheduler(cfg, model, len(train_dataset))
    if args.mode == "train":
        write_logging(cfg.training.logging_dir, cfg)
        # Initialize Trainer
        trainer = SyMSATrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=multimodal_collator,
            compute_metrics=compute_metrics if not is_pretrain else None,
            loss_weights=cfg.training.loss_weight,
            optimizers=(optimizer, lr_scheduler),
            cls_type=cfg.training.cls_type,
            # callbacks=[EvalLossAggregatorCallback()]
        )
        # Start training
        trainer.train()
        # best model is loaded automatically
        trainer.save_model()  # saves to output_dir

    # test mode (or after training)
    if args.mode == "test":
        print('-' * 100, '\nTesting phase\n', '-' * 100)
        # load best checkpoint
        model = MultimodalModelForClassification.from_pretrained(cfg.model, is_pretrain, args.mode, args.model_path)
        model.to(device)

    # Initialize Trainer
    test_trainer = SyMSATrainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        data_collator=multimodal_collator,
        compute_metrics=compute_metrics if not is_pretrain else None,
        loss_weights=cfg.training.loss_weight,
        cls_type=cfg.training.cls_type
    )
    preds = test_trainer.predict(test_dataset)
    print("Test metrics:", preds.metrics)

    # save per-sample results with pandas
    if not is_pretrain:
        df = pd.DataFrame({
            'pred': preds.predictions[1].argmax(-1),
            'label': preds.label_ids,
            'probabilities': torch.softmax(torch.tensor(preds.predictions[1]), dim=1).round(decimals=4).tolist(),
            # torch.softmax(torch.tensor(preds.predictions[1]), dim=1).max(dim=1).values,
            'img_path': preds.predictions[0],
        })
        scores_output_file = (
            f"{cfg.training.result_dir}/test_results_{test_dataset.label_type}.csv")
        mmcv.mkdir_or_exist(dirname(scores_output_file))
        df.to_csv(scores_output_file, index=False)
        print(f"Saved test results to {scores_output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Sentiment Classification")
    parser.add_argument(
        "--config", type=str, required=False,
        default="configs/pretrain_msed.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train", help="Run mode")
    parser.add_argument("--model_path", type=str, default=None, help="Path to pretrained model for finetune phase")
    args = parser.parse_args()
    main(args)
