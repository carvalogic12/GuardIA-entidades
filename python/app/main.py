from contextlib import asynccontextmanager
import json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from app.schemas import (
    DocumentCheckResponse,
    EntityDefinition,
    ExtractRequest,
    ExtractResponse,
    TrainResponse,
)
from app.service import DEFAULT_MODEL_NAME, GLiNER2Service

APP_TITLE = "GLiNER2 NER API"
APP_VERSION = "1.0.0"
MODEL_NAME = DEFAULT_MODEL_NAME

service = GLiNER2Service(model_name=MODEL_NAME)


@asynccontextmanager
async def lifespan(_: FastAPI):
    service.load_model()
    yield


app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    lifespan=lifespan,
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


@app.post(
    "/comprobar-documento",
    response_model=DocumentCheckResponse,
    summary="Comprobar documento PDF por bloques de 500 palabras",
    description=(
        "Recibe un PDF y una lista de entidades con sus definiciones. "
        "Extrae el texto con pdfplumber, lo divide en bloques de 500 palabras "
        "y analiza las entidades nominales en cada bloque."
    ),
)
async def comprobar_documento(
    entities: str = Form(
        ...,
        description='JSON con lista de entidades y definiciones. Ej: [{"name":"empresa","definition":"Organizacion comercial"}]',
    ),
    pdf: UploadFile = File(..., description="Documento PDF a analizar"),
    threshold: float = Form(0.5),
    include_confidence: bool = Form(True),
    include_spans: bool = Form(True),
) -> DocumentCheckResponse:
    if pdf.content_type and pdf.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="El archivo debe ser un PDF.")

    try:
        entities_payload = json.loads(entities)
        entity_definitions = [EntityDefinition.model_validate(item) for item in entities_payload]
    except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as exc:
        raise HTTPException(
            status_code=400,
            detail="El campo entities debe ser un JSON valido con name y definition.",
        ) from exc

    if not entity_definitions:
        raise HTTPException(
            status_code=400,
            detail="Debes enviar al menos una entidad con su definicion.",
        )

    pdf_bytes = await pdf.read()
    try:
        chunk_results, entities_found, total_words = service.comprobar_documento(
            pdf_bytes=pdf_bytes,
            entities=entity_definitions,
            threshold=threshold,
            include_confidence=include_confidence,
            include_spans=include_spans,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=f"No se pudo procesar el PDF: {exc}",
        ) from exc

    return DocumentCheckResponse(
        model=MODEL_NAME,
        total_words=total_words,
        total_chunks=len(chunk_results),
        chunks=chunk_results,
        entities=entities_found,
    )


@app.post(
    "/entrenar",
    response_model=TrainResponse,
    summary="Entrenar GLiNER2 desde archivo",
    description=(
        "Entrena GLiNER2 con estrategia LoRA, siempre partiendo del modelo base "
        "`fastino/gliner2-multi-v1`, y guarda el resultado como `gliner_entrenado`."
    ),
)
async def entrenar(
    train_file: UploadFile = File(..., description="Archivo de entrenamiento (JSONL)"),
    num_epochs: int = Form(3),
    batch_size: int = Form(8),
    learning_rate: float = Form(2e-5),
    warmup_ratio: float = Form(0.1),
    max_length: int = Form(384),
    gradient_accumulation_steps: int = Form(1),
    eval_steps: int = Form(50),
    save_steps: int = Form(50),
    seed: int = Form(42),
    lora_r: int = Form(16),
    lora_alpha: int = Form(32),
    lora_dropout: float = Form(0.05),
) -> TrainResponse:
    train_file_bytes = await train_file.read()
    if not train_file_bytes:
        raise HTTPException(status_code=400, detail="El archivo de entrenamiento esta vacio.")

    try:
        output_model_dir = service.entrenar(
            train_file_bytes=train_file_bytes,
            train_filename=train_file.filename or "train.jsonl",
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            max_length=max_length,
            gradient_accumulation_steps=gradient_accumulation_steps,
            eval_steps=eval_steps,
            save_steps=save_steps,
            seed=seed,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Error durante el entrenamiento: {exc}") from exc

    return TrainResponse(
        status="ok",
        base_model="fastino/gliner2-multi-v1",
        output_model_dir=output_model_dir,
    )
