# GLiNER2 API Project

Proyecto en Python para extraer entidades nominales con **GLiNER2** usando el modelo de Hugging Face **`fastino/gliner2-multi-v1`**.

## Estructura

- `app/main.py`: API FastAPI (Swagger en `/docs`)
- `app/service.py`: carga del modelo y lógica de inferencia
- `app/schemas.py`: contratos de request/response
- `scripts/train.py`: entrenamiento/finetuning
- `scripts/test.py`: test/evaluación sobre JSONL

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Ejecutar la API (Swagger)

```bash
python3 run_api.py
```

Configuracion por archivo (`config/app_config.json`) y override por variables del SO:

```bash
export APP_MODEL_NAME="fastino/gliner2-multi-v1"
export APP_PORT=9000
python3 run_api.py
```

Abrir:

- Swagger UI: `http://localhost:<PUERTO>/docs`
- OpenAPI JSON: `http://localhost:<PUERTO>/openapi.json`

## Request de ejemplo

```json
{
  "text": "OpenAI contrató a Ana García en Madrid y firmó con Iberia.",
  "entities": [
    {
      "name": "persona",
      "definition": "Nombre completo de una persona"
    },
    {
      "name": "organizacion",
      "definition": "Nombre de una empresa o institución"
    },
    {
      "name": "ubicacion",
      "definition": "Ciudad, país o lugar geográfico"
    }
  ],
  "threshold": 0.5,
  "include_confidence": true,
  "include_spans": true
}
```

## Respuesta de ejemplo

```json
{
  "model": "fastino/gliner2-multi-v1",
  "entities": [
    {
      "text": "OpenAI",
      "label": "organizacion",
      "score": 0.98,
      "start": 0,
      "end": 6
    }
  ]
}
```

## Entrenamiento

Dataset en JSONL (una muestra por línea, formato recomendado por GLiNER2):

```json
{
  "input": "Ada Lovelace trabajó con Charles Babbage",
  "output": {
    "entities": {
      "persona": ["Ada Lovelace", "Charles Babbage"]
    }
  }
}
```

Ejecutar entrenamiento:

```bash
python scripts/train.py \
  --train-file data/train.jsonl \
  --val-file data/val.jsonl \
  --base-model fastino/gliner2-multi-v1 \
  --output-dir models/gliner2-finetuned \
  --num-epochs 3 \
  --batch-size 8
```

## Test / Evaluación

```bash
python scripts/test.py \
  --model fastino/gliner2-multi-v1 \
  --test-file data/test.jsonl \
  --threshold 0.5
```

Muestra métricas micro: `Precision`, `Recall` y `F1`.
