"""
Kannada Text Classifier Training
Fine-tunes a multilingual BERT on the Kannada classification dataset.
"""

import os
import sys
import json
import logging
import torch
import numpy as np
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from train.prepare_data import prepare_text_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BASE_MODEL = "bert-base-multilingual-cased"


def train_text_classifier(
    config: Config,
    num_epochs: int = 10,
    batch_size: int = 16,
    lr: float = 2e-5,
    output_dir: str = "checkpoints/text_classifier",
    progress_callback=None,
) -> dict:
    """
    Fine-tune multilingual BERT on Kannada text classification.
    """
    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            get_linear_schedule_with_warmup,
        )
        from torch.utils.data import DataLoader
        from torch.optim import AdamW
        from sklearn.metrics import accuracy_score, classification_report
    except ImportError as e:
        return {"success": False, "error": f"Missing dependency: {e}"}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Text classifier training on device: {device}")

    # Prepare datasets
    datasets = prepare_text_data(config)
    if datasets is None:
        return {
            "success": False,
            "error": "No text dataset found. Place 'KannadaKasturi_dataset_300_5label.xlsx' in data/",
        }

    train_ds = datasets["train"]
    val_ds = datasets["val"]
    num_labels = len(train_ds.label2id)

    logger.info(f"Classes: {train_ds.id2label}")
    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_ds.tokenizer = tokenizer
    val_ds.tokenizer = tokenizer

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    ).to(device)

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    os.makedirs(output_dir, exist_ok=True)
    history = []
    best_val_acc = 0.0
    best_model_path = os.path.join(output_dir, "best_model")

    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_losses = []
        for batch in train_loader:
            if "input_ids" not in batch:
                continue
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                if "input_ids" not in batch:
                    continue
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_losses.append(outputs.loss.item())
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
        avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        avg_val_loss = float(np.mean(val_losses)) if val_losses else 0.0

        epoch_stats = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_acc,
        }
        history.append(epoch_stats)

        logger.info(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if progress_callback:
            progress_callback(epoch, avg_train_loss, val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(best_model_path, exist_ok=True)
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            logger.info(f"New best model saved (val_acc={val_acc:.4f})")

    # Save final model
    final_path = os.path.join(output_dir, "final_model")
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    # Save label map
    with open(os.path.join(output_dir, "label_map.json"), "w") as f:
        json.dump({"label2id": train_ds.label2id, "id2label": train_ds.id2label}, f, ensure_ascii=False)

    # Save history
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Classification report
    if all_labels:
        label_names = [train_ds.id2label.get(i, str(i)) for i in sorted(set(all_labels))]
        report = classification_report(all_labels, all_preds, target_names=label_names)
        logger.info(f"Final Classification Report:\n{report}")

    return {
        "success": True,
        "output_dir": output_dir,
        "best_model_path": best_model_path,
        "epochs": num_epochs,
        "best_val_accuracy": best_val_acc,
        "history": history,
        "label2id": train_ds.label2id,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Kannada text classifier")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output-dir", type=str, default="checkpoints/text_classifier")
    cli_args = parser.parse_args()

    config = Config()
    result = train_text_classifier(
        config,
        num_epochs=cli_args.epochs,
        batch_size=cli_args.batch_size,
        lr=cli_args.lr,
        output_dir=cli_args.output_dir,
    )

    if result["success"]:
        logger.info(f"Done! Best val accuracy: {result['best_val_accuracy']:.4f}")
    else:
        logger.error(f"Failed: {result['error']}")
