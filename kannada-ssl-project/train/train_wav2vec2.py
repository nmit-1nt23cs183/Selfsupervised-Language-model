"""
wav2vec2 Self-Supervised Pre-training on Kannada Audio
This script fine-tunes facebook/wav2vec2-base on Kannada speech data
using contrastive self-supervised learning.
"""

import os
import sys
import logging
import json
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from train.prepare_data import prepare_audio_data, KannadaAudioDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Wav2Vec2TrainingArgs:
    output_dir: str = "checkpoints/wav2vec2_kannada"
    base_model: str = "facebook/wav2vec2-base"
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_steps: int = 500
    logging_steps: int = 50
    save_steps: int = 200
    eval_steps: int = 200
    fp16: bool = False
    max_duration_sec: float = 10.0
    min_duration_sec: float = 1.0
    mask_time_prob: float = 0.065
    mask_feature_prob: float = 0.004


def collate_fn(batch: List[dict], processor) -> dict:
    """Collate audio samples into batches."""
    input_values = [item["input_values"] for item in batch if "error" not in item]
    if not input_values:
        return {}

    encoded = processor(
        input_values,
        sampling_rate=16000,
        padding=True,
        return_tensors="pt",
    )
    return {"input_values": encoded.input_values}


def train_wav2vec2(args: Wav2Vec2TrainingArgs, config: Config, progress_callback=None):
    """
    Main training function for wav2vec2 on Kannada audio.

    Args:
        args: Training arguments
        config: App config
        progress_callback: Optional function(epoch, loss) for progress updates
    """
    try:
        from transformers import (
            Wav2Vec2Config,
            Wav2Vec2ForPreTraining,
            Wav2Vec2Processor,
        )
        from torch.utils.data import DataLoader
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import OneCycleLR
    except ImportError as e:
        logger.error(f"Missing dependency: {e}. Run: pip install transformers torch")
        return {"success": False, "error": str(e)}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training device: {device}")

    # Load dataset
    audio_dataset = prepare_audio_data(config)
    if audio_dataset is None or len(audio_dataset) == 0:
        return {
            "success": False,
            "error": "No Kannada audio data found. Place WAV files in data/dataset_kannada/",
        }

    logger.info(f"Training on {len(audio_dataset)} Kannada audio samples")

    # Load model and processor
    logger.info(f"Loading base model: {args.base_model}")
    processor = Wav2Vec2Processor.from_pretrained(args.base_model)

    model_config = Wav2Vec2Config.from_pretrained(
        args.base_model,
        mask_time_prob=args.mask_time_prob,
        mask_feature_prob=args.mask_feature_prob,
    )
    model = Wav2Vec2ForPreTraining(model_config)
    model = model.to(device)

    # DataLoader
    dataloader = DataLoader(
        audio_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: collate_fn(batch, processor),
        drop_last=True,
    )

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(dataloader) * args.num_train_epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=total_steps,
        pct_start=0.1,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    training_history = []

    logger.info(f"Starting training for {args.num_train_epochs} epochs")

    for epoch in range(1, args.num_train_epochs + 1):
        model.train()
        epoch_losses = []

        for step, batch in enumerate(dataloader):
            if not batch or "input_values" not in batch:
                continue

            input_values = batch["input_values"].to(device)

            optimizer.zero_grad()
            outputs = model(input_values=input_values)
            loss = outputs.loss

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            epoch_losses.append(loss.item())

            if step % args.logging_steps == 0:
                avg_loss = np.mean(epoch_losses[-args.logging_steps:])
                logger.info(f"Epoch {epoch}/{args.num_train_epochs} | Step {step} | Loss: {avg_loss:.4f}")

        avg_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        training_history.append({"epoch": epoch, "loss": avg_epoch_loss})
        logger.info(f"Epoch {epoch} complete. Avg Loss: {avg_epoch_loss:.4f}")

        if progress_callback:
            progress_callback(epoch, avg_epoch_loss)

        # Save checkpoint
        if epoch % max(1, args.num_train_epochs // 5) == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}")
            model.save_pretrained(ckpt_path)
            processor.save_pretrained(ckpt_path)
            logger.info(f"Checkpoint saved: {ckpt_path}")

    # Save final model
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    logger.info(f"Training complete. Model saved to {args.output_dir}")

    # Save training history
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)

    return {
        "success": True,
        "output_dir": args.output_dir,
        "epochs": args.num_train_epochs,
        "final_loss": training_history[-1]["loss"] if training_history else None,
        "history": training_history,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train wav2vec2 on Kannada audio")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="checkpoints/wav2vec2_kannada")
    cli_args = parser.parse_args()

    config = Config()
    train_args = Wav2Vec2TrainingArgs(
        output_dir=cli_args.output_dir,
        num_train_epochs=cli_args.epochs,
        per_device_train_batch_size=cli_args.batch_size,
        learning_rate=cli_args.lr,
    )

    logger.info("Starting Kannada wav2vec2 training...")
    result = train_wav2vec2(train_args, config)

    if result["success"]:
        logger.info(f"Training finished! Final loss: {result['final_loss']:.4f}")
    else:
        logger.error(f"Training failed: {result['error']}")
