from fastapi import FastAPI

from app.schemas import ExtractRequest, ExtractResponse
from app.service import GLiNER2Service

APP_TITLE = "GLiNER2 NER API"
APP_VERSION = "1.0.0"
MODEL_NAME = "fastino/gliner2-multi-v1"

service = GLiNER2Service(model_name=MODEL_NAME)

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=(
        "API para extraccion de entidades nominales con GLiNER2. "
        "Recibe entidades con definiciones y devuelve entidades detectadas."
    ),
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/extract", response_model=ExtractResponse)
def extract_entities(payload: ExtractRequest) -> ExtractResponse:
    entities = service.extract(
        text=payload.text,
        entities=payload.entities,
        threshold=payload.threshold,
        include_confidence=payload.include_confidence,
        include_spans=payload.include_spans,
    )
    return ExtractResponse(model=MODEL_NAME, entities=entities)
