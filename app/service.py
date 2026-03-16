from __future__ import annotations

from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
import tempfile
from threading import Lock

import pdfplumber
import torch
from gliner2 import GLiNER2
from huggingface_hub import snapshot_download

from app.schemas import DocumentChunkResult, EntityDefinition, ExtractedEntity

DEFAULT_MODEL_DIR = Path("/modelo")
DEFAULT_MODEL_NAME = "fastino/gliner2-multi-v1"
TRAINED_MODEL_NAME = "gliner_entrenado"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GLiNER2Service:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: GLiNER2 | None = None
        self._lock = Lock()

    @staticmethod
    def _is_model_dir(path: Path) -> bool:
        return path.is_dir() and (path / "config.json").exists()

    @staticmethod
    def _is_path_like(value: str) -> bool:
        return value.startswith("/") or value.startswith(".") or "/" in value and value.count("/") > 1

    @classmethod
    def resolve_model_path(cls, model_name: str) -> str:
        configured_path = Path(model_name)
        if cls._is_model_dir(configured_path):
            return str(configured_path)
        if configured_path.is_dir():
            entries = [entry for entry in configured_path.iterdir() if cls._is_model_dir(entry)]
            if len(entries) == 1:
                return str(entries[0])

        if DEFAULT_MODEL_DIR.is_dir():
            if cls._is_model_dir(DEFAULT_MODEL_DIR):
                return str(DEFAULT_MODEL_DIR)
            entries = [entry for entry in DEFAULT_MODEL_DIR.iterdir() if cls._is_model_dir(entry)]
            if len(entries) == 1:
                return str(entries[0])

        if cls._is_path_like(model_name):
            raise FileNotFoundError(
                f"No se encontro un modelo local en '{model_name}' ni en '{DEFAULT_MODEL_DIR}'."
            )

        downloaded_path = snapshot_download(
            repo_id=model_name,
            local_dir=DEFAULT_MODEL_DIR,
            local_files_only=False,
        )
        downloaded_model_path = Path(downloaded_path)
        if cls._is_model_dir(downloaded_model_path):
            return str(downloaded_model_path)
        entries = [entry for entry in downloaded_model_path.iterdir() if cls._is_model_dir(entry)]
        if len(entries) == 1:
            return str(entries[0])

        raise FileNotFoundError(
            f"No se encontro un modelo local en '{model_name}' ni en '{DEFAULT_MODEL_DIR}'."
        )

    def load_model(self) -> GLiNER2:
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is None:
                model_path = self.resolve_model_path(self.model_name)
                model = GLiNER2.from_pretrained(model_path, local_files_only=True)
                self._model = model.to(DEFAULT_DEVICE)

        return self._model

    @staticmethod
    def _build_schema(entities: Iterable[EntityDefinition]) -> dict[str, str]:
        return {entity.name: entity.definition for entity in entities}

    @staticmethod
    def _split_text_in_chunks(text: str, words_per_chunk: int = 500) -> list[str]:
        words = text.split()
        if not words:
            return []
        return [
            " ".join(words[index : index + words_per_chunk])
            for index in range(0, len(words), words_per_chunk)
        ]

    @staticmethod
    def _extract_text_from_pdf(pdf_bytes: bytes) -> str:
        pages_text: list[str] = []
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages_text.append(page_text)
        return "\n".join(pages_text)

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

    def comprobar_documento(
        self,
        pdf_bytes: bytes,
        entities: list[EntityDefinition],
        threshold: float = 0.5,
        include_confidence: bool = True,
        include_spans: bool = True,
    ) -> tuple[list[DocumentChunkResult], list[ExtractedEntity], int]:
        text = self._extract_text_from_pdf(pdf_bytes)
        chunks = self._split_text_in_chunks(text=text, words_per_chunk=500)
        chunk_results: list[DocumentChunkResult] = []
        all_entities: list[ExtractedEntity] = []

        for chunk_index, chunk_text in enumerate(chunks, start=1):
            chunk_entities = self.extract(
                text=chunk_text,
                entities=entities,
                threshold=threshold,
                include_confidence=include_confidence,
                include_spans=include_spans,
            )
            chunk_results.append(
                DocumentChunkResult(chunk_index=chunk_index, entities=chunk_entities)
            )
            all_entities.extend(chunk_entities)

        total_words = len(text.split())
        return chunk_results, all_entities, total_words

    def entrenar(
        self,
        train_file_bytes: bytes,
        train_filename: str = "train.jsonl",
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        max_length: int = 384,
        gradient_accumulation_steps: int = 1,
        eval_steps: int = 50,
        save_steps: int = 50,
        seed: int = 42,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ) -> str:
        try:
            from gliner2 import Extractor
            from gliner2.old_trainer import (
                ExtractorDataCollator,
                ExtractorDataset,
                ExtractorTrainer,
                TrainingArguments,
            )
            from gliner2.training.lora import LoRAConfig, apply_lora_to_model
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "No se pudo importar los componentes de entrenamiento de gliner2."
            ) from exc

        output_dir = DEFAULT_MODEL_DIR / TRAINED_MODEL_NAME
        output_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(train_filename).suffix or ".jsonl"
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=suffix,
            prefix="train_gliner2_",
            delete=False,
        ) as train_tmp:
            train_tmp.write(train_file_bytes)
            train_tmp_path = Path(train_tmp.name)

        model = Extractor.from_pretrained(DEFAULT_MODEL_NAME)
        lora_config = LoRAConfig(
            enabled=True,
            r=lora_r,
            alpha=float(lora_alpha),
            dropout=lora_dropout,
        )
        model, lora_layers = apply_lora_to_model(model, lora_config)
        model._lora_layers = lora_layers

        processor = model.processor
        train_dataset = ExtractorDataset(str(train_tmp_path))
        data_collator = ExtractorDataCollator(processor, is_training=True)
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            do_train=True,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            num_train_epochs=float(num_epochs),
            save_steps=save_steps,
            eval_steps=eval_steps,
            eval_strategy="no",
            logging_steps=max(1, min(10, save_steps)),
            save_total_limit=2,
            remove_unused_columns=False,
            seed=seed,
        )
        trainer = ExtractorTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        try:
            trainer.train()
            model.save_pretrained(str(output_dir))
        finally:
            train_tmp_path.unlink(missing_ok=True)

        # Actualiza el servicio para poder usar el modelo entrenado en inferencia posterior.
        self.model_name = str(output_dir)
        self._model = None
        return str(output_dir)
