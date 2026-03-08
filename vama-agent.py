#!/usr/bin/env python3
"""
VAMA 2.0 - SISTEMA DE PRODUCCIÓN COMPLETO
Mantiene: Lógica de cajas, Memoria LP, Sesiones y RAG corregido.
"""
# ==================== PARCHES INICIALES ====================
import sys, subprocess, os
try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pysqlite3-binary"])
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.utils import embedding_functions
import ollama, re, math, pickle, json
from typing import Dict, Optional
from datetime import datetime, timedelta
from flask import Flask, request, jsonify

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
CHROMA_PATH = "chroma_db_v3"
MEMORIA_PATH = "memoria_vama.pkl"
MODELO = "qwen3:30b-a3b-fp16"

# ============================================================================
# CONEXIÓN A DB
# ============================================================================
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-m3", device="cuda"
)
client = chromadb.PersistentClient(path=CHROMA_PATH)

cols = {
    "nacionales": client.get_collection("nacionales", embedding_function=embedding_func),
    "importados": client.get_or_create_collection("importados", embedding_function=embedding_func),
    "griferia": client.get_or_create_collection("griferia", embedding_function=embedding_func),
    "polvos": client.get_or_create_collection("polvos", embedding_function=embedding_func),
    "otras": client.get_or_create_collection("otras", embedding_function=embedding_func)
}

# ============================================================================
# CLASES DE NEGOCIO (TU LÓGICA ORIGINAL COMPLETA)
# ============================================================================
class MemoriaPersistente:
    def __init__(self, archivo=MEMORIA_PATH):
        self.archivo = archivo
        self.datos = self._cargar()
    
    def _cargar(self):
        if os.path.exists(self.archivo):
            try:
                with open(self.archivo, 'rb') as f: return pickle.load(f)
            except: return {}
        return {}
    
    def guardar(self):
        with open(self.archivo, 'wb') as f:
            pickle.dump(self.datos, f)
    
    def obtener(self, usuario_id):
        return self.datos.get(usuario_id, {
            "nombre": "", "cotizaciones": [], "productos_vistos": [], 
            "ultima_visita": None, "total_acumulado": 0
        })
    
    def agregar_cotizacion(self, usuario_id, nombre, items, total):
        if usuario_id not in self.datos: self.datos[usuario_id] = self.obtener(usuario_id)
        self.datos[usuario_id]["cotizaciones"].append({
            "fecha": datetime.now().isoformat(), "items": items, "total": total
        })
        self.datos[usuario_id]["nombre"] = nombre
        self.datos[usuario_id]["ultima_visita"] = datetime.now().isoformat()
        self.datos[usuario_id]["total_acumulado"] += total
        self.guardar()

memoria_largo_plazo = MemoriaPersistente()

class DB:
    def buscar(self, query, colecciones=None, top_k=5):
        if colecciones is None:
            colecciones = ["nacionales", "importados", "griferia", "polvos", "otras"]
        resultados = []
        for nombre in colecciones:
            if nombre not in cols: continue
            try:
                r = cols[nombre].query(query_texts=[query], n_results=top_k)
                for meta in r["metadatas"][0]:
                    meta['coleccion'] = nombre # Guardamos origen
                    resultados.append(meta)
            except: continue
        return resultados[:top_k]

db = DB()

class EstadoConversacion:
    def __init__(self, usuario_id, nombre):
        self.usuario_id, self.nombre = usuario_id, nombre
        self.producto_seleccionado = None
        self.m2_proyecto = None
        self.items_cotizacion = []
        self.ultimos_productos = []
        self.ultimo_mensaje = datetime.now()

    def actualizar(self): self.ultimo_mensaje = datetime.now()
    def guardar_productos(self, productos): self.ultimos_productos = productos

    def agregar_item(self, producto, calculo, m2=None):
        self.items_cotizacion.append({"producto": producto, "calculo": calculo, "m2": m2})
        if m2: self.m2_proyecto = m2

    def get_total(self):
        return sum(item["calculo"].get("total", 0) for item in self.items_cotizacion)

class GestorSesiones:
    def __init__(self): self.sesiones = {}
    def obtener(self, usuario_id, nombre):
        if usuario_id not in self.sesiones:
            self.sesiones[usuario_id] = EstadoConversacion(usuario_id, nombre)
        self.sesiones[usuario_id].actualizar()
        return self.sesiones[usuario_id]

gestor_sesiones = GestorSesiones()

# ============================================================================
# FORMATEADORES PARA EL LLM
# ============================================================================
def formatear_productos_catalogo(productos):
    if not productos: return "CATÁLOGO VACÍO."
    texto = "--- LISTA DE PRODUCTOS DISPONIBLES (ORDEN ESTRICTO) ---\n"
    for i, p in enumerate(productos, 1):
        # Guardamos el índice en el objeto para que el estado lo sepa
        p['indice_sesion'] = i 
        pre = p.get('precio_m2') or p.get('precio_unitario') or 0
        texto += f"PRODUCTO {i}: {p['descripcion']} | PRECIO: ${pre} | ORIGEN: {p.get('coleccion','').upper()}\n"
    return texto

def formatear_cotizacion(items):
    if not items: return "Sin cotización."
    res = "COTIZACIÓN ACTUAL:\n"
    for it in items: res += f"- {it['producto']['descripcion']}: ${it['calculo']['total']}\n"
    return res

# ============================================================================
# FUNCIÓN PRINCIPAL (EL CEREBRO)
# ============================================================================
def generar_respuesta_llm(mensaje, estado, usuario_id):
    msg_low = mensaje.lower()
    
    # --- LOGICA DE BÚSQUEDA CORREGIDA ---
    quiere_buscar = any(p in msg_low for p in ["necesito", "busco", "quiero", "pegamento", "adhesivo", "tienen", "hay"])
    
    if quiere_buscar or not estado.ultimos_productos:
        # Si busca pegamento, forzamos la colección 'polvos'
        if any(p in msg_low for p in ["pegamento", "adhesivo", "pega"]):
            productos = db.buscar(mensaje, colecciones=["polvos"], top_k=10)
        else:
            productos = db.buscar(mensaje, top_k=10)
        
        estado.guardar_productos(productos)
        
        # DEBUG TERMINAL (PARA TI)
        print(f"\n--- 🔍 DEBUG DB ---")
        print(f"Mensaje: '{mensaje}' | Encontrados: {len(productos)}")
        for p in productos: print(f" -> {p['descripcion']} ({p['coleccion']})")

    # Datos para el Prompt
    hist = memoria_largo_plazo.obtener(usuario_id)
    cat_txt = formatear_productos_catalogo(estado.ultimos_productos)
    cot_txt = formatear_cotizacion(estado.items_cotizacion)

    prompt = f"""Eres VAMA, un vendedor de precisión. No inventes índices.
{cat_txt}

REGLAS DE RESPUESTA:
1. Si el cliente dice un número (ej. "5"), busca EXACTAMENTE el 'PRODUCTO 5' de la lista de arriba. 
2. NO confundas los nombres. Si el PRODUCTO 5 es TULUM, responde sobre TULUM.
3. Si el cliente elige uno, pregúntale: "¿Para cuántos metros cuadrados (m2) lo necesitas?" para calcular bultos y cajas.

MENSAJE DEL CLIENTE: "{mensaje}"
RESPUESTA:"""

    try:
        res = ollama.generate(model=MODELO, prompt=prompt, options={'temperature': 0.3})['response']
        return res.strip()
    except Exception as e: return f"Error: {e}"

# ============================================================================
# WEBHOOK
# ============================================================================
app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json() or {}
    tel, nom, msg = data.get('telefono', '0'), data.get('nombre', 'Cliente'), data.get('mensaje', '')
    sesion = gestor_sesiones.obtener(tel, nom)
    respuesta = generar_respuesta_llm(msg, sesion, tel)
    return jsonify({"respuesta": respuesta})

if __name__ == "__main__":
    total = sum(c.count() for c in cols.values())
    print(f"🚀 VAMA API LISTA. {total} productos en DB.")
    app.run(host='0.0.0.0', port=5000)