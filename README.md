# VAMA Agent 3.0 - Asistente de Ventas Inteligente 🤖

Sistema de cotización conversacional con memoria persistente para VAMA (materiales de construcción). 
Combina RAG (Retrieval-Augmented Generation) con flujo de ventas guiado por opciones numéricas.

## 🎯 ¿Qué hace?

- **Busca productos** en catálogo (pisos, azulejos, grifería, pegamentos) usando embeddings semánticos
- **Cotiza automáticamente** calculando cajas, sacos y totales
- **Mantiene conversación** con contexto (agrega complementos: pegamento + piso)
- **Guarda historial** de cotizaciones para consultas futuras
- **Multi-usuario** con sesiones independientes

## 🧠 Arquitectura Técnica

| Componente | Tecnología |
|------------|-----------|
| Embeddings | BAAI/bge-m3 (multilingüe, 1024 dims) |
| Vector DB | ChromaDB con distancia coseno |
| LLM Local | Qwen 3 30B-A3B via Ollama (opcional) |
| Memoria Corto Plazo | Sesión en RAM (2h) |
| Memoria Largo Plazo | Pickle persistente (disco) |
| Backend API | Flask (para N8N/WhatsApp) |

## 🚀 Instalación

```bash
# Clonar
git clone https://github.com/aquariustech12/tile_rag.git
cd vama-agent

# Dependencias
pip install chromadb ollama flask numpy

# Verificar Ollama (opcional, para modo LLM)
ollama list

📋 Uso
Modo Consola (pruebas)
python3 vama-agent.py
# ¿Usar LLM? (s/n): n  <- recomendado para pruebas

# Ejecutar
python3 vama-agent.py
Flujo típico:
Selecciona usuario (1-4)
Escribe qué buscas: "pisos blancos para baño"
Elige por número: 2
Indica metros: 20
Agrega complementos: "pegamento" → elige 3
Termina: "listo"
Modo API (producción con N8N/WhatsApp)

python3 vama-agent.py api
# Endpoint: POST http://localhost:5000/webhook
# Body: {"telefono": "5215512345678", "nombre": "Juan", "mensaje": "busco pisos"}

chroma_db_v3/          # Vector DB (no incluir en git)
├── nacionales/        # Pisos y azulejos nacionales
├── importados/        # Pisos importados
├── griferia/          # Llaves, regaderas, etc.
├── polvos/            # Pegamentos, adhesivos, boquillas
└── otras/             # Muebles, espejos, etc.

memoria_vama.pkl       # Historial de cotizaciones (no incluir en git)

🗂️ Cargar Catálogo
Si necesitas regenerar la base de datos vectorial:

python3 ingest_tiles_catalog.py

⚙️ Configuración
Variables de entorno (opcional):

export OLLAMA_MODEL="qwen3:30b-a3b-fp16"
export MEMORIA_PATH="./memoria_vama.pkl"
export CHROMA_PATH="./chroma_db_v3"

🐛 Troubleshooting
| Problema                 | Solución                                                                                                                                                  |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| "No encuentra productos" | Verificar que ChromaDB tenga datos: `python3 -c "import chromadb; print(chromadb.PersistentClient('chroma_db_v3').get_collection('nacionales').count())"` |
| Ollama no responde       | `sudo systemctl start ollama` o ejecutar sin LLM (modo determinista)                                                                                      |
| Memoria corrupta         | Borrar `memoria_vama.pkl` y reiniciar                                                                                                                     |

📁 Archivos Principales
vama-agent.py - Agente conversacional principal
ingest_tiles_catalog.py - ETL para vectorizar Excel
query_tiles.py - Scripts de consulta directa (debug)
👥 Equipo
Desarrollo: Julian Lugo
Infraestructura: SCTI
Datos: VAMA catalogo 2024
🏢 vama.com.mx - Materiales de construcción
