# tile_rag (VAMA 3.5)

Asistente de cotización para materiales y acabados con **Flask + ChromaDB + Ollama**, optimizado para conversación natural, memoria de usuario y generación de PDF.

## ¿Qué cambió recientemente?

Esta versión documenta los cambios más recientes del branch (`happy-path-complete`, `requirements-update` e `ingest-update`):

- **Flujo conversacional “happy path” reforzado** en `vama-agent2.py`:
  - Validaciones de conexión al arrancar (Ollama, Chroma y embeddings).
  - Memoria conversacional persistente con historial.
  - Interceptores para cierres de compra/pago/agradecimientos (evita “volver a vender” cuando el cliente ya está cerrando).
  - Manejo de cotización con resumen total y registro de cotizaciones cerradas.
- **Operación y observabilidad**:
  - Logs diarios de conversación en `logs/`.
  - Dataset incremental para fine-tuning en `dataset/conversaciones_completas.jsonl`.
  - Dashboard simple en `/dashboard` y healthcheck en `/health`.
- **Salida comercial**:
  - Generación de PDF por cliente en `cotizaciones/` y descarga por endpoint.
- **Dependencias actualizadas** en `requirements.txt`.
- **Ingesta** con ajuste reciente en `ingest_tiles_catalog.py`.

---

## Estructura del proyecto

- `vama-agent2.py`: servidor principal (VAMA 3.5) y lógica conversacional.
- `vama-agent.py`: versión alternativa previa/corregida.
- `ingest_tiles_catalog.py`: ingesta de CSVs a Chroma en colecciones por categoría.
- `catalog.py`: utilidades de búsqueda rápida por código, CSV y fallback de Chroma.
- `revisar_db.py`: inspección rápida de documentos/metadatos en Chroma.
- `test_vama.py`: prueba de conectividad directa contra Ollama API.
- `tests/router_tests.sh`: smoke tests históricos del endpoint `/webhook`.
- `catalog_work/codes_index.json`: índice de códigos para lookup.
- `dataset/`, `logs/`, `cotizaciones/`: artefactos generados en runtime.

---

## Requisitos

- Python 3.10+
- Ollama levantado localmente en `http://127.0.0.1:11434`
- Modelo Ollama disponible (default del proyecto: `qwen2.5:14b`)

Instalación de dependencias:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## ChromaDB y datos de catálogo

La base de datos de Chroma (`chroma_db_v3`) **se genera localmente** con la ingesta y **no debe versionarse** en el repositorio.

Los CSV de entrada para la ingesta se esperan en un directorio local `data/` (por ejemplo `data/nacionales.csv`, `data/importados.csv`, etc.).

---

## Ingesta de catálogo

Ingesta completa (si están los CSV esperados en `data/`):

```bash
python ingest_tiles_catalog.py --tipo todo
```

Ingesta por categoría (ejemplo):

```bash
python ingest_tiles_catalog.py --tipo nacionales --file data/nacionales.csv
python ingest_tiles_catalog.py --tipo importados --file data/importados.csv
python ingest_tiles_catalog.py --tipo promos --file data/promo.csv
```

Tipos soportados por `--tipo`:

`nacionales`, `importados`, `griferia`, `lavabos`, `sanitarios`, `muebles`, `tinacos`, `espejos`, `tarjas`, `herramientas`, `polvos`, `otras`, `promos`, `todo`.

---

## Ejecutar el agente principal

```bash
python vama-agent2.py
```

El servicio corre en **`0.0.0.0:5001`**.

### Endpoints (puerto 5001)

- `POST /webhook`: entrada principal de conversación.
- `GET /pdf/<telefono>`: descarga el último PDF generado para el cliente.
- `GET /dashboard`: tablero simple de operación (usuarios, cotizaciones, logs).
- `GET /health`: estado básico del servicio.

Ejemplo de request:

```bash
curl -X POST "http://127.0.0.1:5001/webhook" \
  -H "Content-Type: application/json" \
  -d '{"telefono":"5215512345678","nombre":"Cliente","mensaje":"Necesito piso blanco para baño"}'
```

---

## Pruebas y utilidades

Prueba de conectividad Ollama (sin librería intermedia):

```bash
python test_vama.py
```

Inspección rápida de Chroma:

```bash
python revisar_db.py
```

Prueba rápida del router en puerto 5001:

```bash
curl -s -X POST "http://127.0.0.1:5001/webhook" \
  -H "Content-Type: application/json" \
  -d '{"telefono":"demo","nombre":"Cliente","mensaje":"CMX01"}'
```

> Nota: `tests/router_tests.sh` es un script histórico; valida/ajusta su puerto antes de usarlo si tu entorno está en `5001`.

---

## Notas operativas

- El modelo configurado en código es `qwen2.5:14b`. Si usarás otro modelo, actualiza la constante `MODELO`.
- El sistema persiste memoria de usuarios en `memoria_vama.pkl`.
- El dataset para fine-tuning crece con conversaciones útiles; considera rotación/limpieza periódica.
- El endpoint `/dashboard` usa lecturas directas de archivos de logs/dataset para monitoreo rápido.
