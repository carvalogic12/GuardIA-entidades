"""Inferencia interactiva con GLiNER2 leyendo schema desde JSON.

Uso:
  python3 infer.py
  python3 infer.py --schema-file data/entity_descriptions.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.schemas import EntityDefinition, ExtractedEntity
from app.service import GLiNER2Service
from config_loader import get_config


def parse_args() -> argparse.Namespace:
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Inferencia interactiva con GLiNER2")
    parser.add_argument("--model", default=cfg.model_name, help="Modelo HF o ruta local")
    parser.add_argument(
        "--schema-file",
        default="data/entity_descriptions.json",
        help="Archivo JSON con definiciones de entidades",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Umbral de confianza [0,1]")
    parser.add_argument("--no-confidence", action="store_true", help="No incluir score")
    parser.add_argument("--no-spans", action="store_true", help="No incluir start/end")
    return parser.parse_args()


def load_schema(path_str: str) -> dict[str, str]:
    path = Path(path_str)
    if not path.exists():
        # Compatibilidad con nombres alternativos que suelen usarse por error tipografico.
        fallbacks = [
            Path("entity_Descripttion.json"),
            Path("entity_descriptions.json"),
            Path("data/entity_Descripttion.json"),
        ]
        for candidate in fallbacks:
            if candidate.exists():
                path = candidate
                break

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo de entidades: {path_str}. "
            "Prueba con --schema-file data/entity_descriptions.json"
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError("El schema debe ser un JSON objeto no vacio: {\"entidad\": \"definicion\"}")

    schema: dict[str, str] = {}
    for key, value in data.items():
        k = str(key).strip()
        v = str(value).strip()
        if k and v:
            schema[k] = v

    if not schema:
        raise ValueError("No hay entidades validas en el schema.")

    return schema


def print_entities(entities: list[ExtractedEntity]) -> None:
    if not entities:
        print("- (sin entidades detectadas)")
        return

    for entity in entities:
        print(
            f"- {entity.label}: '{entity.text}'"
            f"{'' if entity.score is None else f' | score={entity.score:.4f}'}"
            f"{'' if entity.start is None or entity.end is None else f' | span=({entity.start},{entity.end})'}"
        )


class GLiNER2ServiceCompat(GLiNER2Service):
    def extract(
        self,
        text: str,
        entities: list[EntityDefinition],
        threshold: float = 0.5,
        include_confidence: bool = True,
        include_spans: bool = True,
    ) -> list[ExtractedEntity]:
        model = self.load_model()
        schema = self._build_schema(entities)
        try:
            raw_entities = model.extract_entities(
                text=text,
                schema=schema,
                threshold=threshold,
                include_confidence=include_confidence,
                include_spans=include_spans,
            )
        except TypeError:
            raw_entities = model.extract_entities(
                text=text,
                entity_types=list(schema.keys()),
                threshold=threshold,
                include_confidence=include_confidence,
                include_spans=include_spans,
            )

        normalized: list[ExtractedEntity] = []
        entities_by_label = raw_entities.get("entities", {})
        for label, values in entities_by_label.items():
            for value in values:
                if isinstance(value, dict):
                    normalized.append(
                        ExtractedEntity(
                            text=value.get("text", ""),
                            label=label,
                            score=value.get("confidence"),
                            start=value.get("start"),
                            end=value.get("end"),
                        )
                    )
                else:
                    normalized.append(ExtractedEntity(text=str(value), label=label))
        return normalized


def main() -> None:
    args = parse_args()
    schema = load_schema(args.schema_file)
    entity_defs = [EntityDefinition(name=key, definition=value) for key, value in schema.items()]

    print("Cargando modelo...")
    service = GLiNER2ServiceCompat(model_name=args.model)
    service.load_model()
    print(f"Modelo: {args.model}")
    print(f"Entidades cargadas: {', '.join(schema.keys())}")
    print("Escribe un texto y pulsa Enter. Escribe 'salir' para terminar.\n")

    include_confidence = not args.no_confidence
    include_spans = not args.no_spans

    while True:
        try:
            text = input("Texto> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaliendo.")
            break

        if not text:
            continue
        if text.lower() in {"salir", "exit", "quit"}:
            print("Saliendo.")
            break

        t0 = time.perf_counter()
        result = service.extract(
            text=text,
            entities=entity_defs,
            threshold=args.threshold,
            include_confidence=include_confidence,
            include_spans=include_spans,
        )
        elapsed = time.perf_counter() - t0
        print_entities(result)
        print(f"Tiempo de inferencia: {elapsed:.3f}s")
        print()


if __name__ == "__main__":
    main()
