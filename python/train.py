"""Entrenamiento de GLiNER2 usando JSONL.

Formato recomendado (una linea por muestra):
{
  "input": "Ada Lovelace trabajó con Charles Babbage en Londres",
  "output": {
    "entities": {
      "persona": ["Ada Lovelace", "Charles Babbage"],
      "ubicacion": ["Londres"]
    }
  }
}
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gliner2 import Extractor
from gliner2.old_trainer import (
    ExtractorDataCollator,
    ExtractorDataset,
    ExtractorTrainer,
    TrainingArguments,
)
from gliner2.training.lora import LoRAConfig, apply_lora_to_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento para GLiNER2")
    parser.add_argument("--train-file", required=True, help="Dataset train en JSONL")
    parser.add_argument("--val-file", required=False, help="Dataset validacion en JSONL")
    parser.add_argument(
        "--base-model",
        default="fastino/gliner2-multi-v1",
        help="Modelo base de Hugging Face",
    )
    parser.add_argument(
        "--output-dir",
        default="./models/gliner2-finetuned",
        help="Directorio de salida",
    )
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser.parse_args()


def ensure_file(path_str: str | None, kind: str) -> str | None:
    if path_str is None:
        return None
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"No existe archivo de {kind}: {path}")
    return str(path)


def main() -> None:
    args = parse_args()
    train_file = ensure_file(args.train_file, "train")
    _ = ensure_file(args.val_file, "validacion")

    model = Extractor.from_pretrained(args.base_model)
    lora_config = LoRAConfig(
        enabled=True,
        r=args.lora_r,
        alpha=float(args.lora_alpha),
        dropout=args.lora_dropout,
    )
    model, lora_layers = apply_lora_to_model(model, lora_config)
    model._lora_layers = lora_layers

    processor = model.processor
    train_dataset = ExtractorDataset(train_file)
    data_collator = ExtractorDataCollator(processor, is_training=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=float(args.num_epochs),
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="no",
        logging_steps=max(1, min(10, args.save_steps)),
        save_total_limit=2,
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer = ExtractorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train()

    # Fusiona LoRA y guarda un modelo normal compatible con inferencia.
    model.save_pretrained(args.output_dir)
    print(f"Modelo entrenado y guardado en: {args.output_dir}")


if __name__ == "__main__":
    main()
