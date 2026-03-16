# GuardIA-entidades

## Swagger UI

- Documentacion interactiva: `/docs`
- OpenAPI JSON: `/openapi.json`

### Endpoint nuevo

- `POST /comprobar-documento`
- `multipart/form-data`:
  - `pdf`: archivo PDF
  - `entities`: JSON string con `name` y `definition`
  - `threshold` (opcional)
  - `include_confidence` (opcional)
  - `include_spans` (opcional)

- `POST /entrenar`
- `multipart/form-data`:
  - `train_file`: archivo JSONL de entrenamiento
  - `num_epochs`, `batch_size`, `learning_rate`, `warmup_ratio`
  - `max_length`, `gradient_accumulation_steps`, `eval_steps`, `save_steps`, `seed`
  - `lora_r`, `lora_alpha`, `lora_dropout`
  - El entrenamiento usa LoRA, siempre parte de `fastino/gliner2-multi-v1` y se guarda en `/modelo/gliner_entrenado`

## Dataset de ejemplo

- Archivo: `data/ejemplo_de_entrenamiento.jsonl`
- Incluye varias muestras para detectar entidades de tipo `calle`.
