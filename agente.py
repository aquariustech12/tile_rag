
#!/usr/bin/env python3
"""
VAMA PRO v2.2 - WhatsApp Integration
Listo para producción con familia (4 usuarios)
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import chromadb
from chromadb.utils import embedding_functions
import ollama
import re
import json
import math
import pickle
import threading
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from flask import Flask, request, jsonify

# Para WhatsApp usarás una de estas opciones:
# Opción A: CallMeBot (gratis, web-based)
# Opción B: WhatsApp Business API (Meta oficial, pago)
# Opción C: whatsapp-web.js vía Node puente (recomendado)

CHROMA_PATH = "chroma_db_v3"
MEMORIA_PATH = "memoria_vama.pkl"
WHATSAPP_MODE = "callmebot"  # Cambia a "api" o "bridge" según evolucione

app = Flask(__name__)

# ============================================================================
# DATABASE (igual que antes)
# ============================================================================

print("🔌 Iniciando VAMA WhatsApp Bot...")
print("🔌 Conectando a ChromaDB...")

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-m3",
    device="cuda"
)
client = chromadb.PersistentClient(path=CHROMA_PATH)

cols = {
    "nacionales": client.get_collection("nacionales", embedding_function=embedding_func),
    "importados": client.get_or_create_collection("importados", embedding_function=embedding_func),
    "griferia": client.get_or_create_collection("griferia", embedding_function=embedding_func),
    "polvos": client.get_or_create_collection("polvos", embedding_function=embedding_func),
    "otras": client.get_or_create_collection("otras", embedding_function=embedding_func)
}

try:
    ollama.list()
    print("✅ Ollama disponible")
except:
    print("⚠️ Ollama no responde")

total_productos = sum(c.count() for c in cols.values())
print(f"✅ {total_productos} productos listos")
print(f"✅ Modo WhatsApp: {WHATSAPP_MODE}")

# ============================================================================
# MEMORIA (igual que antes, optimizado)
# ============================================================================

class MemoriaPersistente:
    def __init__(self, archivo=MEMORIA_PATH):
        self.archivo = archivo
        self.datos = self._cargar()
        self.lock = threading.Lock()
    
    def _cargar(self):
        if os.path.exists(self.archivo):
            try:
                with open(self.archivo, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}
    
    def _guardar(self):
        with self.lock:
            with open(self.archivo, 'wb') as f:
                pickle.dump(self.datos, f)
    
    def obtener(self, usuario_id):
        return self.datos.get(usuario_id, {
            "nombre": "",
            "cotizaciones": [],
            "productos_vistos": [],
            "ultima_visita": None,
            "total_gastado": 0
        })
    
    def guardar_cotizacion(self, usuario_id, nombre, items, total):
        with self.lock:
            if usuario_id not in self.datos:
                self.datos[usuario_id] = self.obtener(usuario_id)
            
            self.datos[usuario_id]["cotizaciones"].append({
                "fecha": datetime.now().isoformat(),
                "items": items,
                "total": total
            })
            self.datos[usuario_id]["nombre"] = nombre
            self.datos[usuario_id]["ultima_visita"] = datetime.now().isoformat()
            self.datos[usuario_id]["total_gastado"] += total
            
            for item in items:
                desc = item["producto"]["descripcion"]
                if desc not in self.datos[usuario_id]["productos_vistos"]:
                    self.datos[usuario_id]["productos_vistos"].append(desc)
            
            self._guardar()

memoria = MemoriaPersistente()

# ============================================================================
# SESIÓN ACTIVA (Corto plazo)
# ============================================================================

class SesionActiva:
    def __init__(self, telefono, nombre):
        self.telefono = telefono
        self.nombre = nombre
        self.producto_seleccionado = None
        self.m2_proyecto = None
        self.items = []
        self.ultimos_productos = []
        self.ultimo_mensaje = datetime.now()
    
    def actualizar(self):
        self.ultimo_mensaje = datetime.now()

sesiones_activas: Dict[str, SesionActiva] = {}
lock_sesiones = threading.Lock()

def obtener_sesion(telefono, nombre):
    with lock_sesiones:
        # Limpiar sesiones viejas (>2 horas)
        ahora = datetime.now()
        vencidos = [t for t, s in sesiones_activas.items() 
                   if (ahora - s.ultimo_mensaje).seconds > 7200]
        for t in vencidos:
            del sesiones_activas[t]
        
        if telefono not in sesiones_activas:
            sesiones_activas[telefono] = SesionActiva(telefono, nombre)
            print(f"🆕 Nueva sesión: {telefono[-10:]}")
        
        sesiones_activas[telefono].actualizar()
        return sesiones_activas[telefono]

# ============================================================================
# DB Y CÁLCULOS (Optimizados)
# ============================================================================

class DB:
    def __init__(self):
        self.cols = cols
    
    def buscar(self, query, colecciones=None, tipo=None, top_k=5):
        if colecciones is None:
            colecciones = ["nacionales", "importados"]
        
        resultados = []
        for nombre in colecciones:
            if nombre not in self.cols:
                continue
            
            try:
                col = self.cols[nombre]
                where = {"tipo": tipo} if tipo and nombre in ["nacionales", "importados"] else None
                
                if where:
                    r = col.query(query_texts=[query], n_results=top_k, where=where)
                else:
                    r = col.query(query_texts=[query], n_results=top_k)
                
                for meta, pid in zip(r["metadatas"][0], r["ids"][0]):
                    resultados.append({
                        "codigo": meta.get("codigo", ""),
                        "descripcion": meta.get("descripcion", ""),
                        "proveedor": meta.get("proveedor", ""),
                        "precio_m2": float(meta.get("precio_m2", 0)) or 0,
                        "precio_unitario": float(meta.get("precio_unitario", 0)) or 0,
                        "metraje_caja": float(meta.get("metraje_caja", 1.44)) or 1.44,
                        "formato": meta.get("formato", ""),
                        "coleccion": nombre
                    })
            except Exception as e:
                print(f"Error DB {nombre}: {e}")
                continue
        
        # Quitar duplicados por código
        unicos = {}
        for r in resultados:
            if r["codigo"] not in unicos:
                unicos[r["codigo"]] = r
        return list(unicos.values())[:top_k]

db = DB()

def calcular_piso(producto, m2):
    cajas = math.ceil(m2 / producto["metraje_caja"])
    total = cajas * producto["precio_m2"] * producto["metraje_caja"]
    return {"cajas": cajas, "total": total, "detalle": f"{cajas} cajas = ${total:.2f}"}

# ============================================================================
# CLASIFICADOR (Simplificado y robusto)
# ============================================================================

def clasificar(mensaje: str, sesion: SesionActiva):
    m = mensaje.lower().strip()
    
    # Número 1-5 = selección si hay productos previos
    if m.isdigit() and 1 <= int(m) <= 5 and sesion.ultimos_productos:
        return {"intencion": "seleccionar", "indice": int(m) - 1}
    
    # Recordar cotización anterior
    if any(x in m for x in ["ayer", "anterior", "otro día", "seguimos", "todavía"]):
        return {"intencion": "recordar"}
    
    # Despedida
    if any(x in m for x in ["gracias", "listo", "terminamos", "adiós", "chao"]):
        return {"intencion": "despedida"}
    
    # Cotizar (tiene números)
    if re.search(r'\d+', m) and any(x in m for x in ["m2", "metros", "precio", "cuesta"]):
        return {"intencion": "cotizar"}
    
    # Agregar complemento
    if sesion.items and any(x in m for x in ["también", "además", "otro", "más", "agrega"]):
        return {"intencion": "agregar"}
    
    # Por defecto: buscar
    return {"intencion": "buscar", "categoria": detectar_cat(m)}

def detectar_cat(mensaje: str):
    m = mensaje.lower()
    if any(x in m for x in ["piso", "porcelanato", "baño", "suelo"]):
        return "pisos"
    if any(x in m for x in ["muro", "azulejo", "pared"]):
        return "muros"
    if any(x in m for x in ["grifo", "llave", "regadera"]):
        return "griferia"
    if any(x in m for x in ["pega", "adhesivo", "boquilla", "cemento"]):
        return "polvos"
    return None

# ============================================================================
# RESPUESTAS (Compactas para WhatsApp)
# ============================================================================

def resp_buscar(sesion, clas):
    cat = clas.get("categoria")
    if not cat:
        return "¿Qué buscas? (pisos, azulejos, grifería, pegamento)"
    
    productos = db.buscar("baño" if cat == "pisos" else cat, tipo=cat[:-1] if cat in ["pisos", "muros"] else None, top_k=5)
    
    if not productos:
        return f"No encontré {cat}. Intenta con otras palabras."
    
    sesion.ultimos_productos = productos
    
    msg = f"*{cat.upper()}* - Opciones:\n\n"
    for i, p in enumerate(productos[:5], 1):
        if p["precio_m2"] > 0:
            msg += f"{i}. {p['descripcion'][:32]}\n"
            msg += f"   ${p['precio_m2']:.0f}/m² | {p['proveedor']}\n\n"
        else:
            msg += f"{i}. {p['descripcion'][:32]}\n"
            msg += f"   ${p['precio_unitario']:.0f}/u | {p['proveedor']}\n\n"
    
    msg += "Responde con el número (1-5):"
    return msg

def resp_seleccionar(sesion, clas):
    idx = clas.get("indice", 0)
    if not sesion.ultimos_productos or idx >= len(sesion.ultimos_productos):
        return "Primero busca productos. ¿Qué necesitas?"
    
    prod = sesion.ultimos_productos[idx]
    sesion.producto_seleccionado = prod
    
    if prod["precio_m2"] > 0:
        return f"✅ *{prod['descripcion'][:40]}*\n💵 ${prod['precio_m2']:.2f}/m²\n\n¿Cuántos m² necesitas?"
    else:
        return f"✅ *{prod['descripcion'][:40']}*\n💵 ${prod['precio_unitario']:.2f}/u\n\n¿Cuántas unidades?"

def resp_cotizar(sesion, mensaje):
    # Extraer número
    match = re.search(r'(\d+(?:\.\d+)?)', mensaje.replace(",", "."))
    if not match:
        return "¿Para cuántos m²? (ejemplo: 25)"
    
    m2 = float(match.group(1))
    sesion.m2_proyecto = m2
    
    if not sesion.producto_seleccionado:
        if sesion.ultimos_productos:
            sesion.producto_seleccionado = sesion.ultimos_productos[0]
        else:
            return "Primero selecciona un producto (1-5)"
    
    prod = sesion.producto_seleccionado
    calc = calcular_piso(prod, m2)
    
    sesion.items.append({
        "producto": prod,
        "calculo": calc,
        "m2": m2
    })
    
    total = sum(i["calculo"]["total"] for i in sesion.items)
    
    msg = f"💰 *COTIZACIÓN*\n\n"
    msg += f"*{prod['descripcion'][:35]}*\n"
    msg += f"📐 {m2}m² = {calc['detalle']}\n\n"
    msg += f"💵 *TOTAL: ${total:.2f}*\n\n"
    msg += "¿Algo más? (pegamento, grifería) o 'listo'"
    
    return msg

def resp_recordar(telefono, nombre):
    hist = memoria.obtener(telefono)
    
    if not hist["cotizaciones"]:
        return "No tengo registro anterior. Empecemos de nuevo. ¿Qué buscas?"
    
    ultima = hist["cotizaciones"][-1]
    dias = (datetime.now() - datetime.fromisoformat(ultima["fecha"])).days
    
    msg = f"📚 *BIENVENIDO DE VUELTA {nombre}*\n\n"
    if dias == 0:
        msg += "Hoy ya cotizaste con nosotros.\n"
    elif dias == 1:
        msg += "Ayer estuviste aquí.\n"
    else:
        msg += f"Última vez: hace {dias} días.\n"
    
    msg += f"💰 Total histórico: ${hist['total_gastado']:.2f}\n\n"
    msg += "*TU ÚLTIMA COTIZACIÓN:*\n"
    
    for i, item in enumerate(ultima["items"][-2:], 1):
        p = item["producto"]
        msg += f"{i}. {p['descripcion'][:30]}... = ${item['calculo']['total']:.0f}\n"
    
    msg += f"\n¿Repetir esta cotización o nueva búsqueda?"
    return msg

def resp_despedida(telefono, sesion):
    if not sesion.items:
        return "¡Gracias! VAMA https://vama.com.mx 👋"
    
    total = sum(i["calculo"]["total"] for i in sesion.items)
    memoria.guardar_cotizacion(telefono, sesion.nombre, sesion.items, total)
    
    msg = f"🎉 *COTIZACIÓN GUARDADA*\n\n"
    msg += f"💵 *TOTAL: ${total:.2f}*\n"
    msg += f"📋 {len(sesion.items)} productos\n\n"
    msg += "Tu cotización quedó guardada. Puedes consultarla escribiendo '¿te acuerdas?'\n\n"
    msg += "VAMA https://vama.com.mx 👋"
    
    # Limpiar sesión
    with lock_sesiones:
        if telefono in sesiones_activas:
            del sesiones_activas[telefono]
    
    return msg

def resp_agregar(sesion):
    if not sesion.items:
        return "Primero cotiza un piso. ¿Qué producto necesitas?"
    return "🔧 *COMPLEMENTOS*\n\n¿Qué agregamos?\n• Pegamento\n• Boquilla\n• Grifería\n• Otro piso\n\nEspecifica cuál:"

# ============================================================================
# PROCESADOR PRINCIPAL
# ============================================================================

def procesar(telefono: str, nombre: str, mensaje: str) -> str:
    sesion = obtener_sesion(telefono, nombre)
    clas = clasificar(mensaje, sesion)
    
    print(f"[{telefono[-10:]}] {clas['intencion']}: {mensaje[:40]}...")
    
    if clas["intencion"] == "recordar":
        return resp_recordar(telefono, nombre)
    elif clas["intencion"] == "buscar":
        return resp_buscar(sesion, clas)
    elif clas["intencion"] == "seleccionar":
        return resp_seleccionar(sesion, clas)
    elif clas["intencion"] == "cotizar":
        return resp_cotizar(sesion, mensaje)
    elif clas["intencion"] == "agregar":
        return resp_agregar(sesion)
    elif clas["intencion"] == "despedida":
        return resp_despedida(telefono, sesion)
    
    return "No entendí. Intenta: 'busco pisos' o 'cotizar 25m2'"

# ============================================================================
# WHATSAPP INTEGRATION
# ============================================================================

# Opción 1: CallMeBot (Gratis, fácil, para empezar hoy)
def enviar_whatsapp_callmebot(telefono, mensaje):
    """Usa CallMeBot API gratuita"""
    import urllib.request
    import urllib.parse
    
    # Necesitas obtener API key de http://www.callmebot.com/blog/free-api-whatsapp-messages/
    api_key = os.getenv("CALLMEBOT_KEY", "TU_API_KEY_AQUI")
    
    mensaje_codificado = urllib.parse.quote(mensaje)
    url = f"https://api.callmebot.com/whatsapp.php?phone={telefono}&text={mensaje_codificado}&apikey={api_key}"
    
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return response.read()
    except Exception as e:
        print(f"Error enviando WhatsApp: {e}")
        return None

# Opción 2: Webhook para recibir mensajes (usarás esto)
@app.route('/webhook', methods=['POST'])
def webhook():
    """Recibe mensajes de WhatsApp (via CallMeBot, Twilio, o tu puente)"""
    data = request.json or request.form
    
    # Extraer datos según el proveedor
    telefono = data.get('phone') or data.get('From', '').replace('whatsapp:', '')
    mensaje = data.get('message') or data.get('Body', '')
    nombre = data.get('name') or 'Cliente'
    
    if not telefono or not mensaje:
        return jsonify({"error": "Faltan datos"}), 400
    
    # Procesar
    respuesta = procesar(telefono, nombre, mensaje)
    
    # Enviar respuesta de vuelta
    if WHATSAPP_MODE == "callmebot":
        enviar_whatsapp_callmebot(telefono, respuesta)
    
    return jsonify({"status": "ok", "respuesta": respuesta})

@app.route('/test', methods=['GET'])
def test():
    return "VAMA Bot activo"

# ============================================================================
# MODO CONSOLA (Para probar hoy con tu familia sin WhatsApp aún)
# ============================================================================

def modo_consola_familia():
    """Simula 4 familiares para probar lógica antes de WhatsApp"""
    
    familia = {
        "5215512345678": ("Papá", "busca pisos para la cocina"),
        "5215598765432": ("Mamá", "azulejos para el baño"),
        "5215522223333": ("Hermano", "grifería"),
        "5215544445555": ("Tú", "cotizar 45m2")
    }
    
    print("\n" + "="*60)
    print("MODO PRUEBA FAMILIAR (Simulando WhatsApp)")
    print("="*60)
    print("Usuarios configurados:")
    for num, (nombre, _) in familia.items():
        print(f"  {nombre}: {num}")
    print("-"*60)
    print("Escribe mensajes como si fueran de WhatsApp")
    print("Comandos: 'reset' = limpiar todo | 'salir' = salir")
    print("="*60 + "\n")
    
    telefono_actual = "5215512345678"
    nombre_actual = "Papá"
    
    while True:
        print(f"\n👤 {nombre_actual} ({telefono_actual[-6:]}): ")
        entrada = input("Mensaje (o 'cambiar'/'reset'/'salir'): ").strip()
        
        if entrada.lower() == 'salir':
            break
        elif entrada.lower() == 'reset':
            sesiones_activas.clear()
            print("🧹 Todo reiniciado")
            continue
        elif entrada.lower() == 'cambiar':
            print("\nElige:")
            for i, (num, (nom, _)) in enumerate(familia.items(), 1):
                print(f"  {i}. {nom}")
            try:
                opt = int(input("Número: ")) - 1
                telefono_actual = list(familia.keys())[opt]
                nombre_actual = familia[telefono_actual][0]
            except:
                print("❌ Inválido")
            continue
        
        # Procesar como si viniera de WhatsApp
        respuesta = procesar(telefono_actual, nombre_actual, entrada)
        
        print(f"\n🤖 VAMA:\n{respuesta}\n")
        print("-"*40)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "whatsapp":
        # Modo producción WhatsApp
        print("\n🚀 Iniciando servidor WhatsApp...")
        print("Configura tu webhook en:")
        print("  - CallMeBot: http://www.callmebot.com/")
        print("  - O usa ngrok para pruebas: ngrok http 5000")
        print("\nEsperando mensajes en http://0.0.0.0:5000/webhook\n")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        # Modo prueba familiar (consola)
        modo_consola_familia()