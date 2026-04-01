from pydantic import BaseModel, Field


class EntityDefinition(BaseModel):
    name: str = Field(..., description="Nombre del tipo de entidad, ej: empresa")
    definition: str = Field(..., description="Definicion de la entidad para guiar al modelo")


class ExtractRequest(BaseModel):
    text: str = Field(..., description="Texto sobre el cual extraer entidades")
    entities: list[EntityDefinition] = Field(
        ...,
        min_length=1,
        description="Lista de entidades objetivo con su definicion",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Umbral minimo de confianza para aceptar entidades",
    )
    include_confidence: bool = Field(
        default=True,
        description="Si es true, pide confianza por entidad",
    )
    include_spans: bool = Field(
        default=True,
        description="Si es true, pide posiciones start/end por entidad",
    )


class ExtractedEntity(BaseModel):
    text: str
    label: str
    score: float | None = None
    start: int | None = None
    end: int | None = None


class ExtractResponse(BaseModel):
    model: str
    entities: list[ExtractedEntity]


class DocumentChunkResult(BaseModel):
    chunk_index: int
    entities: list[ExtractedEntity]


class DocumentCheckResponse(BaseModel):
    model: str
    total_words: int
    total_chunks: int
    chunks: list[DocumentChunkResult]
    entities: list[ExtractedEntity]


class TrainResponse(BaseModel):
    status: str
    base_model: str
    output_model_dir: str
