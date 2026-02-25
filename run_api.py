from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from gliner2 import GLiNER2

from app.schemas import EntityDefinition, ExtractRequest, ExtractResponse, ExtractedEntity
from config_loader import get_config

APP_TITLE = "GLiNER2 NER API"
APP_VERSION = "1.0.0"

cfg = get_config()


class GLiNER2CompatService:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: GLiNER2 | None = None

    def _load_model(self) -> GLiNER2:
        if self._model is None:
            self._model = GLiNER2.from_pretrained(self.model_name)
        return self._model

    @staticmethod
    def _build_schema(entities: list[EntityDefinition]) -> dict[str, str]:
        return {entity.name: entity.definition for entity in entities}

    def _extract_raw(
        self,
        text: str,
        schema: dict[str, str],
        threshold: float,
        include_confidence: bool,
        include_spans: bool,
    ) -> dict:
        model = self._load_model()
        try:
            return model.extract_entities(
                text=text,
                schema=schema,
                threshold=threshold,
                include_confidence=include_confidence,
                include_spans=include_spans,
            )
        except TypeError:
            return model.extract_entities(
                text=text,
                entity_types=list(schema.keys()),
                threshold=threshold,
                include_confidence=include_confidence,
                include_spans=include_spans,
            )

    def extract(
        self,
        text: str,
        entities: list[EntityDefinition],
        threshold: float,
        include_confidence: bool,
        include_spans: bool,
    ) -> list[ExtractedEntity]:
        schema = self._build_schema(entities)
        raw_entities = self._extract_raw(
            text=text,
            schema=schema,
            threshold=threshold,
            include_confidence=include_confidence,
            include_spans=include_spans,
        )

        normalized: list[ExtractedEntity] = []
        by_label = raw_entities.get("entities", {})
        for label, values in by_label.items():
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


service = GLiNER2CompatService(model_name=cfg.model_name)

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=(
        "API para extraccion de entidades nominales con GLiNER2. "
        "Recibe entidades con definiciones y devuelve entidades detectadas."
    ),
)


@app.get("/health")
def health() -> dict[str, str | int]:
    return {"status": "ok", "model": cfg.model_name, "port": cfg.port}


@app.post("/extract", response_model=ExtractResponse)
def extract_entities(payload: ExtractRequest) -> ExtractResponse:
    entities = service.extract(
        text=payload.text,
        entities=payload.entities,
        threshold=payload.threshold,
        include_confidence=payload.include_confidence,
        include_spans=payload.include_spans,
    )
    return ExtractResponse(model=cfg.model_name, entities=entities)


if __name__ == "__main__":
    uvicorn.run("run_api:app", host="0.0.0.0", port=cfg.port)
