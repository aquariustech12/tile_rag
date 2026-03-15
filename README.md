-# VAMA Agent 2.0 - Asistente de Ventas Inteligente 🤖
+# VAMA Agent 3.0 (Flask + Chroma + Ollama)
 
-Sistema de cotización conversacional con memoria persistente para VAMA (materiales de construcción). 
-Combina RAG (Retrieval-Augmented Generation) con flujo de ventas guiado por opciones numéricas.
+Asistente conversacional para cotización de materiales de construcción.
 
-## 🎯 ¿Qué hace?
+Este proyecto implementa un agente de ventas por mensajes: recibe texto del cliente, busca productos en una base vectorial y responde con un flujo de cotización guiado.
 
-- **Busca productos** en catálogo (pisos, azulejos, grifería, pegamentos) usando embeddings semánticos
-- **Cotiza automáticamente** calculando cajas, sacos y totales
-- **Mantiene conversación** con contexto (agrega complementos: pegamento + piso)
-- **Guarda historial** de cotizaciones para consultas futuras
-- **Multi-usuario** con sesiones independientes
+---
 
-## 🧠 Arquitectura Técnica
+## 1) Resumen funcional
 
-| Componente | Tecnología |
-|------------|-----------|
-| Embeddings | BAAI/bge-m3 (multilingüe, 1024 dims) |
-| Vector DB | ChromaDB con distancia coseno |
-| LLM Local | Qwen 3 30B-A3B via Ollama (opcional) |
-| Memoria Corto Plazo | Sesión en RAM (2h) |
-| Memoria Largo Plazo | Pickle persistente (disco) |
-| Backend API | Flask (para N8N/WhatsApp) |
+El agente:
 
-## 🚀 Instalación
+- Busca productos en ChromaDB usando embeddings semánticos.
+- Detecta intención por palabras clave (`piso`, `azulejo`, `grifo`, `lavabo`, etc.).
+- Detecta color en el mensaje (`blanco`, `gris`, `negro`, etc.).
+- Detecta metros cuadrados cuando el cliente escribe `m2`, `m²`, `metros` o `mts`.
+- Muestra hasta 3 productos y permite selección por número (`1`, `2`, `3`) o por código de producto.
+- Calcula cantidad sugerida con `ceil(m2_proyecto / m2_caja)`.
+- Mantiene memoria por usuario (carrito, últimos productos y m2).
+- Entrega resumen y total cuando el cliente solicita cierre.
+
+---
+
+## 2) Arquitectura real (según `vama-agent.py`)
+
+- **API**: Flask (`POST /webhook`).
+- **LLM local**: Ollama en `http://127.0.0.1:11434`.
+- **Modelo**: `qwen2.5:3b`.
+- **Vector DB**: ChromaDB persistente en `chroma_db_v3`.
+- **Embeddings**: `BAAI/bge-m3` en CPU.
+- **Memoria**: archivo pickle `memoria_vama.pkl`.
+
+> Diseño de control: el flujo de negocio está en Python (determinista) y el LLM solo “redacta” respuestas.
+
+---
+
+## 3) Colecciones de Chroma
+
+El agente usa 12 colecciones:
+
+1. `nacionales`
+2. `importados`
+3. `griferia`
+4. `lavabos`
+5. `sanitarios`
+6. `muebles`
+7. `tinacos`
+8. `espejos`
+9. `tarjas`
+10. `herramientas`
+11. `polvos`
+12. `otras`
+
+### Importante
+
+- `nacionales` e `importados` se cargan con `get_collection(...)`; deben existir previamente en Chroma.
+- Las demás colecciones se crean si no existen (`get_or_create_collection(...)`).
+
+---
+
+## 4) Mapeo de intención → colección
+
+Ejemplos relevantes del mapeo interno:
+
+- `piso`, `azulejo`, `porcelanato`, `muro`, `pared` → `nacionales`, `importados`
+- `grifo`, `monomando`, `llave`, `regadera` → `griferia`
+- `lavabo` → `lavabos`
+- `wc`, `inodoro`, `taza`, `sanitario` → `sanitarios`
+- `mueble`, `gabinete` → `muebles`
+- `tinaco`, `cisterna` → `tinacos`
+- `espejo` → `espejos`
+- `tarja`, `fregadero` → `tarjas`
+- `herramienta` → `herramientas`
+- `pegamento`, `pega`, `adesivo`, `cemix` → `polvos`, `otras`
+
+Si no detecta intención, usa `piso` por defecto.
+
+---
+
+## 5) Requisitos
+
+- Python 3.10+
+- Dependencias de `requirements.txt`
+- Ollama activo en `127.0.0.1:11434`
+- ChromaDB persistente en `./chroma_db_v3`
+
+Instalación:
 
 ```bash
-# Clonar
-git clone https://github.com/aquariustech12/tile_rag.git
-cd vama-agent
+pip install -r requirements.txt
+```
 
-# Dependencias
-pip install chromadb ollama flask numpy
+---
 
-# Verificar Ollama (opcional, para modo LLM)
-ollama list
+## 6) Ejecución
 
-📋 Uso
-Modo Consola (pruebas)
+```bash
 python3 vama-agent.py
-# ¿Usar LLM? (s/n): n  <- recomendado para pruebas
+```
 
-# Ejecutar
-python3 vama-agent.py
-Flujo típico:
-Selecciona usuario (1-4)
-Escribe qué buscas: "pisos blancos para baño"
-Elige por número: 2
-Indica metros: 20
-Agrega complementos: "pegamento" → elige 3
-Termina: "listo"
-Modo API (producción con N8N/WhatsApp)
-
-python3 vama-agent.py api
-# Endpoint: POST http://localhost:5000/webhook
-# Body: {"telefono": "5215512345678", "nombre": "Juan", "mensaje": "busco pisos"}
-
-chroma_db_v3/          # Vector DB (no incluir en git)
-├── nacionales/        # Pisos y azulejos nacionales
-├── importados/        # Pisos importados
-├── griferia/          # Llaves, regaderas, etc.
-├── polvos/            # Pegamentos, adhesivos, boquillas
-└── otras/             # Muebles, espejos, etc.
-
-memoria_vama.pkl       # Historial de cotizaciones (no incluir en git)
-
-🗂️ Cargar Catálogo
-Si necesitas regenerar la base de datos vectorial:
-
-python3 ingest_tiles_catalog.py
-
-⚙️ Configuración
-Variables de entorno (opcional):
-
-export OLLAMA_MODEL="qwen3:30b-a3b-fp16"
-export MEMORIA_PATH="./memoria_vama.pkl"
-export CHROMA_PATH="./chroma_db_v3"
-
-🐛 Troubleshooting
-| Problema                 | Solución                                                                                                                                                  |
-| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
-| "No encuentra productos" | Verificar que ChromaDB tenga datos: `python3 -c "import chromadb; print(chromadb.PersistentClient('chroma_db_v3').get_collection('nacionales').count())"` |
-| Ollama no responde       | `sudo systemctl start ollama` o ejecutar sin LLM (modo determinista)                                                                                      |
-| Memoria corrupta         | Borrar `memoria_vama.pkl` y reiniciar                                                                                                                     |
-
-📁 Archivos Principales
-vama-agent.py - Agente conversacional principal
-ingest_tiles_catalog.py - ETL para vectorizar Excel
-query_tiles.py - Scripts de consulta directa (debug)
-👥 Equipo
-Desarrollo: Julian Lugo
-Infraestructura: SCTI
-Datos: VAMA catalogo 2024
-🏢 vama.com.mx - Materiales de construcción
+Al iniciar imprime algo como:
+
+- `🚀 VAMA 3.0 | 12 colecciones | Modelo: qwen2.5:3b`
+- Lista de colecciones detectadas
+- Servidor Flask en `0.0.0.0:5000`
+
+---
+
+## 7) API
+
+### Endpoint
+
+`POST /webhook`
+
+### Request
+
+```json
+{
+  "telefono": "5215512345678",
+  "nombre": "Juan",
+  "mensaje": "busco piso gris para 20 m2"
+}
+```
+
+### Response
+
+```json
+{
+  "respuesta": "...respuesta del agente..."
+}
+```
+
+---
+
+## 8) Flujo conversacional
+
+1. Llega mensaje de usuario.
+2. Se recupera/crea memoria por `telefono`.
+3. Se evalúan comandos de control:
+   - Reinicio: `nueva cotizacion`, `reiniciar`, `empezar de nuevo`, `borrar todo`.
+   - Cierre: `total`, `eso es todo`, `terminamos`, `cuanto es`, `cerrar`.
+4. Si hay lista previa y cliente elige `1/2/3` o código, agrega al carrito.
+5. Si no, detecta intención/color/m2 y ejecuta búsqueda semántica.
+6. Devuelve opciones o mensaje de no encontrado.
+7. Guarda memoria en disco después de cada request.
+
+---
+
+## 9) Estructura de datos en memoria
+
+Por usuario se guarda:
+
+- `nombre`
+- `carrito`
+- `ultimos_productos`
+- `m2`
+- `ultimo_mensaje`
+- `contador`
+
+Archivo persistente: `memoria_vama.pkl`.
+
+---
+
+## 10) Archivos clave
+
+- `vama-agent.py`: agente, búsqueda, memoria y endpoint.
+- `requirements.txt`: dependencias.
+- `chroma_db_v3/`: almacenamiento local de Chroma.
+- `memoria_vama.pkl`: memoria persistida de conversaciones.
+
+---
+
+## 11) Troubleshooting rápido
+
+- **Error al arrancar por colección faltante** (`nacionales`/`importados`):
+  - Debes crear/cargar esas colecciones primero en `chroma_db_v3`.
+- **No devuelve productos relevantes**:
+  - Verifica metadata de catálogo (`descripcion`, `color`, `metraje_caja`, precios).
+- **Ollama no responde**:
+  - Revisa que el servicio esté activo en `127.0.0.1:11434`.
+  - El agente tiene fallback de texto en errores de generación.
+- **Necesitas limpiar conversaciones**:
+  - Elimina `memoria_vama.pkl` y reinicia el servicio.
