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

from gliner2 import GLiNER2Trainer
from gliner2.config import TrainingConfig


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
    val_file = ensure_file(args.val_file, "validacion")

    config = TrainingConfig(
        num_epochs=args.num_epochs,
        train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    trainer = GLiNER2Trainer(
        model_name=args.base_model,
        config=config,
    )

    trainer.train(
        train_data=train_file,
        val_data=val_file,
    )

    trainer.save_model(args.output_dir)
    print(f"Modelo entrenado y guardado en: {args.output_dir}")


if __name__ == "__main__":
    main()
