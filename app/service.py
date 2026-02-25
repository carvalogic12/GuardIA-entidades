from __future__ import annotations

from collections.abc import Iterable
from threading import Lock

from gliner2 import GLiNER2

from app.schemas import EntityDefinition, ExtractedEntity


class GLiNER2Service:
    def __init__(self, model_name: str = "fastino/gliner2-multi-v1") -> None:
        self.model_name = model_name
        self._model: GLiNER2 | None = None
        self._lock = Lock()

    def load_model(self) -> GLiNER2:
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is None:
                self._model = GLiNER2.from_pretrained(self.model_name)

        return self._model

    @staticmethod
    def _build_schema(entities: Iterable[EntityDefinition]) -> dict[str, str]:
        return {entity.name: entity.definition for entity in entities}

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
        raw_entities = model.extract_entities(
            text=text,
            schema=schema,
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
                    normalized.append(
                        ExtractedEntity(
                            text=str(value),
                            label=label,
                        )
                    )
        return normalized
