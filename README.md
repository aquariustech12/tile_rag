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

# Ejecutar
python3 vama-agent.py
