"""Test/evaluacion para GLiNER2 sobre JSONL.

Formatos soportados por muestra:
1) Formato GLiNER2 recomendado:
{
  "input": "texto",
  "output": {"entities": {"persona": ["Ada"]}}
}

2) Formato alternativo:
{
  "input": {"text": "texto"},
  "output": {"entities": [{"label": "persona", "text": "Ada"}]},
  "entity_descriptions": {"persona": "Nombre de una persona"}
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gliner2 import GLiNER2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test para GLiNER2")
    parser.add_argument("--model", default="fastino/gliner2-multi-v1", help="Modelo HF o ruta local")
    parser.add_argument("--test-file", required=True, help="Dataset de prueba en JSONL")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0, help="Maximo de muestras (0 = todas)")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return str(value).strip().lower()


def parse_text(sample: dict[str, Any]) -> str:
    input_field = sample.get("input", "")
    if isinstance(input_field, str):
        return input_field
    if isinstance(input_field, dict):
        return str(input_field.get("text", ""))
    return ""


def parse_expected_entities(sample: dict[str, Any]) -> set[tuple[str, str]]:
    entities = sample.get("output", {}).get("entities", {})
    normalized: set[tuple[str, str]] = set()

    if isinstance(entities, dict):
        for label, values in entities.items():
            for value in values:
                text = value.get("text") if isinstance(value, dict) else value
                t = normalize_text(text)
                l = normalize_text(label)
                if t and l:
                    normalized.add((l, t))
        return normalized

    if isinstance(entities, list):
        for item in entities:
            if not isinstance(item, dict):
                continue
            t = normalize_text(item.get("text"))
            l = normalize_text(item.get("label"))
            if t and l:
                normalized.add((l, t))

    return normalized


def parse_schema(sample: dict[str, Any]) -> dict[str, str]:
    descriptions = sample.get("entity_descriptions")
    if isinstance(descriptions, dict) and descriptions:
        return {str(k): str(v) for k, v in descriptions.items()}

    entities = sample.get("output", {}).get("entities", {})
    if isinstance(entities, dict) and entities:
        return {str(label): f"Entidad de tipo {label}" for label in entities.keys()}

    return {}


def parse_predicted_entities(predicted: dict[str, Any]) -> set[tuple[str, str]]:
    entities = predicted.get("entities", {})
    normalized: set[tuple[str, str]] = set()

    for label, values in entities.items():
        for value in values:
            text = value.get("text") if isinstance(value, dict) else value
            t = normalize_text(text)
            l = normalize_text(label)
            if t and l:
                normalized.add((l, t))

    return normalized


def evaluate(model: GLiNER2, test_file: str, threshold: float, limit: int = 0) -> None:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    processed = 0

    lines = Path(test_file).read_text(encoding="utf-8").splitlines()
    if limit > 0:
        lines = lines[:limit]

    for line in lines:
        line = line.strip()
        if not line:
            continue

        row = json.loads(line)
        text = parse_text(row)
        if not text:
            continue

        schema = parse_schema(row)
        if not schema:
            continue

        expected_set = parse_expected_entities(row)
        predicted = model.extract_entities(
            text=text,
            schema=schema,
            threshold=threshold,
        )
        predicted_set = parse_predicted_entities(predicted)

        tp = len(expected_set & predicted_set)
        fp = len(predicted_set - expected_set)
        fn = len(expected_set - predicted_set)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        processed += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    print("=== Resultado de test ===")
    print(f"Muestras procesadas: {processed}")
    print(f"TP: {total_tp} | FP: {total_fp} | FN: {total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")


def main() -> None:
    args = parse_args()
    model = GLiNER2.from_pretrained(args.model)
    evaluate(model=model, test_file=args.test_file, threshold=args.threshold, limit=args.limit)


if __name__ == "__main__":
    main()
