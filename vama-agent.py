#!/usr/bin/env python3
"""
VAMA 2.0 - Sistema de Cotizaciones con RAG y LLM local
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import chromadb
from chromadb.utils import embedding_functions
import ollama
import re
import math
import pickle
from typing import Dict, Optional
from datetime import datetime, timedelta

CHROMA_PATH = "chroma_db_v3"
MEMORIA_PATH = "memoria_vama.pkl"
MODELO = "qwen3:30b-a3b-fp16"

# ============================================================================
# CONEXIÓN A DB
# ============================================================================

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
print(f"✅ {total_productos} productos en catálogo\n")

# ============================================================================
# MEMORIA PERSISTENTE
# ============================================================================

class MemoriaPersistente:
    def __init__(self, archivo=MEMORIA_PATH):
        self.archivo = archivo
        self.datos = self._cargar()
    
    def _cargar(self):
        if os.path.exists(self.archivo):
            try:
                with open(self.archivo, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}
    
    def guardar(self):
        with open(self.archivo, 'wb') as f:
            pickle.dump(self.datos, f)
    
    def obtener(self, usuario_id):
        return self.datos.get(usuario_id, {
            "nombre": "",
            "cotizaciones": [],
            "productos_vistos": [],
            "ultima_visita": None,
            "total_acumulado": 0
        })
    
    def agregar_cotizacion(self, usuario_id, nombre, items, total):
        if usuario_id not in self.datos:
            self.datos[usuario_id] = self.obtener(usuario_id)
        
        self.datos[usuario_id]["cotizaciones"].append({
            "fecha": datetime.now().isoformat(),
            "items": items,
            "total": total
        })
        self.datos[usuario_id]["nombre"] = nombre
        self.datos[usuario_id]["ultima_visita"] = datetime.now().isoformat()
        self.datos[usuario_id]["total_acumulado"] += total
        
        for item in items:
            desc = item["producto"]["descripcion"]
            if desc not in self.datos[usuario_id]["productos_vistos"]:
                self.datos[usuario_id]["productos_vistos"].append(desc)
        
        self.guardar()

memoria_largo_plazo = MemoriaPersistente()

# ============================================================================
# DB
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
            
            col = self.cols[nombre]
            where_filter = {"tipo": tipo} if tipo and nombre in ["nacionales", "importados"] else None
            
            try:
                if where_filter:
                    r = col.query(query_texts=[query], n_results=top_k, where=where_filter)
                else:
                    r = col.query(query_texts=[query], n_results=top_k)
                
                for meta, dist, pid in zip(r["metadatas"][0], r["distances"][0], r["ids"][0]):
                    resultados.append({
                        "codigo": meta.get("codigo", ""),
                        "descripcion": meta.get("descripcion", ""),
                        "proveedor": meta.get("proveedor", ""),
                        "precio_m2": float(meta.get("precio_m2", 0)) if meta.get("precio_m2") else 0,
                        "precio_unitario": float(meta.get("precio_unitario", 0)) if meta.get("precio_unitario") else 0,
                        "metraje_caja": float(meta.get("metraje_caja", 1.44)) if meta.get("metraje_caja") else 1.44,
                        "formato": meta.get("formato", ""),
                        "color": meta.get("color", ""),
                        "coleccion": nombre
                    })
            except Exception as e:
                print(f"⚠️ Error en búsqueda {nombre}: {e}")
                continue
        
        vistos = set()
        unicos = []
        for r in resultados:
            if r["codigo"] not in vistos:
                vistos.add(r["codigo"])
                unicos.append(r)
        
        return unicos[:top_k]

db = DB()

# ============================================================================
# ESTADO DE CONVERSACIÓN
# ============================================================================

class EstadoConversacion:
    def __init__(self, usuario_id, nombre):
        self.usuario_id = usuario_id
        self.nombre = nombre
        self.producto_seleccionado = None
        self.m2_proyecto = None
        self.items_cotizacion = []
        self.ultimos_productos = []
        self.categoria_activa = None
        self.ultimo_mensaje = datetime.now()
        self.esperando_respuesta = None
        self.saludo_enviado = False  # ← NUEVA BANDERA

    def actualizar(self):
        self.ultimo_mensaje = datetime.now()

    def guardar_productos(self, productos):
        self.ultimos_productos = productos

    def seleccionar(self, idx):
        if 0 <= idx < len(self.ultimos_productos):
            self.producto_seleccionado = self.ultimos_productos[idx]
            return self.producto_seleccionado
        return None

    def agregar_item(self, producto, calculo, m2=None):
        item = {
            "producto": producto,
            "calculo": calculo,
            "m2": m2
        }
        self.items_cotizacion.append(item)
        if m2:
            self.m2_proyecto = m2

    def get_total(self):
        return sum(item["calculo"].get("total", 0) for item in self.items_cotizacion)

    def reset(self):
        self.producto_seleccionado = None
        self.m2_proyecto = None
        self.items_cotizacion = []
        self.ultimos_productos = []
        self.categoria_activa = None
        self.esperando_respuesta = None
        # NO RESETEAR saludo_enviado - así no se repite el saludo

# ============================================================================
# GESTOR DE SESIONES
# ============================================================================

class GestorSesiones:
    def __init__(self):
        self.sesiones = {}
        self.expiracion = timedelta(hours=2)
    
    def obtener(self, usuario_id, nombre):
        ahora = datetime.now()
        
        vencidos = [uid for uid, ses in self.sesiones.items() 
                   if (ahora - ses.ultimo_mensaje) > self.expiracion]
        for uid in vencidos:
            del self.sesiones[uid]
        
        if usuario_id not in self.sesiones:
            self.sesiones[usuario_id] = EstadoConversacion(usuario_id, nombre)
            print(f"🆕 Sesión: {usuario_id}")
        
        self.sesiones[usuario_id].actualizar()
        return self.sesiones[usuario_id]
    
    def guardar_cotizacion(self, usuario_id):
        if usuario_id in self.sesiones:
            sesion = self.sesiones[usuario_id]
            if sesion.items_cotizacion:
                total = sesion.get_total()
                memoria_largo_plazo.agregar_cotizacion(
                    usuario_id, sesion.nombre, sesion.items_cotizacion, total
                )

gestor_sesiones = GestorSesiones()

# ============================================================================
# CÁLCULOS Y UTILERÍAS
# ============================================================================

def calcular_piso(producto, m2):
    cajas = math.ceil(m2 / producto["metraje_caja"])
    total = cajas * producto["precio_m2"] * producto["metraje_caja"]
    return {"cajas": cajas, "total": total, "detalle": f"{cajas} cajas = ${total:.2f}"}

def extraer_numero(mensaje):
    m = mensaje.lower().strip()
    
    palabras = {
        "una": 1, "un": 1, "uno": 1,
        "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
        "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10
    }
    
    if m in palabras:
        return palabras[m]
    
    match = re.search(r'(\d+(?:\.\d+)?)', m.replace(",", "."))
    if match:
        return float(match.group(1))
    
    return None

# ============================================================================
# INTERPRETE QWEN
# ============================================================================

def qwen_interpretar_categoria(mensaje: str) -> str:
    """Qwen decide la categoría - VERSIÓN FORZADA"""
    
    # Si el mensaje es muy corto o obvio, ni preguntamos
    m = mensaje.lower().strip()
    if "piso" in m or "baño" in m or "suelo" in m:
        return "pisos"
    if "muro" in m or "pared" in m or "azulejo" in m:
        return "muros"
    if "grifo" in m or "llave" in m or "monomando" in m:
        return "griferia"
    if "pega" in m or "adhesivo" in m or "cemento" in m:
        return "polvos"
    
    # Si no es obvio, preguntamos a Qwen pero con un prompt MUY específico
    prompt = f"""El cliente escribe: "{mensaje}"

Tu tarea es clasificar su mensaje en UNA de estas categorías:
- pisos (si habla de pisos, suelos, porcelanatos, cerámicas para el suelo)
- muros (si habla de paredes, azulejos, recubrimientos para pared)
- grifería (si habla de llaves, grifos, monomandos, regaderas)
- polvos (si habla de pegamentos, adhesivos, boquillas, cemento)

IMPORTANTE: SIEMPRE responde con UNA SOLA PALABRA de la lista.
NO expliques, NO añadas texto, SOLO la palabra.

Respuesta:"""
    
    try:
        respuesta = ollama.generate(
            model=MODELO,
            prompt=prompt,
            options={'temperature': 0.1, 'num_predict': 10}
        )['response'].strip().lower()
        
        # Limpiar la respuesta (quitar puntos, espacios, etc)
        respuesta = re.sub(r'[^a-z]', '', respuesta)
        
        if respuesta in ["pisos", "muros", "griferia", "polvos"]:
            print(f"   🧠 Qwen interpretó: {respuesta}")
            return respuesta
        else:
            print(f"   ⚠️ Qwen respondió '{respuesta}', usando reglas locales")
            # Reglas locales de respaldo
            if "piso" in m or "baño" in m:
                return "pisos"
            if "muro" in m or "pared" in m:
                return "muros"
            if "grifo" in m or "llave" in m:
                return "griferia"
            if "pega" in m:
                return "polvos"
            return "pisos"  # Default
    except Exception as e:
        print(f"⚠️ Error en Qwen: {e}")
        return "pisos"

def qwen_responder_con_catalogo(mensaje: str, productos: list, contexto: str = "") -> Optional[str]:
    if not productos:
        return None
    
    productos_txt = "=== CATÁLOGO VAMA ===\n"
    for i, p in enumerate(productos, 1):
        if p['precio_m2'] > 0:
            precio_caja = p['precio_m2'] * p['metraje_caja']
            productos_txt += f"{i}. {p['descripcion']} | {p['proveedor']} | ${p['precio_m2']:.2f}/m² | Caja: ${precio_caja:.2f}\n"
        else:
            productos_txt += f"{i}. {p['descripcion']} | {p['proveedor']} | ${p['precio_unitario']:.2f}/unidad\n"
    productos_txt += "=== FIN CATÁLOGO ==="
    
    prompt = f"""{productos_txt}

El cliente dice: "{mensaje}"

Basado ÚNICAMENTE en los productos listados arriba, responde al cliente.
Debes recomendarle algunos productos y terminar preguntando qué número le interesa.

REGLAS:
- SOLO menciona productos que están en la lista
- Usa los precios exactos de la lista
- Recomienda máximo 3 productos
- Termina con: "¿Cuál te interesa? (1-{len(productos)})"

Respuesta:"""
    
    try:
        respuesta = ollama.generate(
            model=MODELO,
            prompt=prompt,
            options={'temperature': 0.3, 'num_predict': 300}
        )['response'].strip()
        
        # Verificar que no esté vacía
        if not respuesta or len(respuesta) < 10:
            # Fallback manual
            return None
            
        return respuesta
    except Exception as e:
        print(f"⚠️ Error en Qwen: {e}")
        return None

# ============================================================================
# VAMA 2.0 - GRIFERÍA CON DICCIONARIO QUEMADO
# ============================================================================

GRIFERIA_CATALOGO = [
    {
        "codigo": "CAS459",
        "descripcion": "MONOMANDO ALTO DAMASCO 3210C",
        "proveedor": "Castel",
        "precio_unitario": 689.00,
        "color": "Gris",
        "tipo": "monomando",
        "uso": "regadera"
    },
    {
        "codigo": "CAS461",
        "descripcion": "MONOMANDO LAVABO BAJO DAMASCO 3212C",
        "proveedor": "Castel",
        "precio_unitario": 589.00,
        "color": "Gris",
        "tipo": "monomando",
        "uso": "lavabo"
    }
]

def formatear_griferia(productos):
    if not productos:
        return ""
    
    msg = "🚿 *GRIFERÍA DISPONIBLE:*\n\n"
    for i, p in enumerate(productos, 1):
        if p.get('uso') == "regadera":
            tipo_emoji = "🚿 Regadera"
        elif p.get('uso') == "lavabo":
            tipo_emoji = "🚰 Lavabo"
        else:
            tipo_emoji = "🚿 Monomando"
        
        color_txt = f" | 🎨 {p['color']}" if p.get('color') else ""
        
        msg += f"{i}. *{p['descripcion'][:50]}*\n"
        msg += f"   {tipo_emoji} | 🏢 {p['proveedor']}{color_txt}\n"
        msg += f"   💰 ${p['precio_unitario']:.2f} c/u\n\n"
    
    msg += "¿Cuál te interesa? Escribe el número (1-{})".format(len(productos))
    return msg

def detectar_tipo_griferia(mensaje):
    m = mensaje.lower()
    if any(x in m for x in ["regadera", "ducha", "teléfono", "telefono"]):
        return "regadera"
    elif any(x in m for x in ["lavabo", "baño", "vanitorio"]):
        return "lavabo"
    return None

def resp_buscar_griferia(sesion, mensaje, tipo_especifico=None):
    if not tipo_especifico:
        tipo_especifico = detectar_tipo_griferia(mensaje)
    
    if tipo_especifico:
        resultados = [p for p in GRIFERIA_CATALOGO if p.get('uso') == tipo_especifico]
    else:
        resultados = GRIFERIA_CATALOGO.copy()
    
    if not resultados and tipo_especifico:
        resultados = GRIFERIA_CATALOGO.copy()
    
    if not resultados:
        return "Lo siento, no tengo grifería en catálogo en este momento."
    
    sesion.guardar_productos(resultados)
    sesion.categoria_activa = "griferia"
    sesion.esperando_respuesta = "seleccionar_griferia"
    
    return formatear_griferia(resultados)

def resp_seleccionar_griferia(sesion, clas, mensaje_original=""):
    idx = clas.get("indice", -1)
    
    if idx < 0 or idx >= len(sesion.ultimos_productos):
        return f"Por favor escribe un número del 1 al {len(sesion.ultimos_productos)}"
    
    producto = sesion.seleccionar(idx)
    if not producto:
        return "Primero busca grifería."
    
    sesion.producto_seleccionado = producto
    sesion.esperando_respuesta = "cantidad_griferia"
    
    color = f" 🎨 {producto['color']}" if producto.get('color') else ""
    uso = f" para {producto['uso']}" if producto.get('uso') else ""
    
    msg = f"✅ *{producto['descripcion']}*\n"
    msg += f"🏢 {producto['proveedor']}{color}{uso}\n"
    msg += f"💰 ${producto['precio_unitario']:.2f} por pieza\n\n"
    msg += "¿Cuántas piezas necesitas? (ejemplo: 2)"
    
    return msg

def resp_cotizar_griferia(sesion, mensaje):
    cantidad = extraer_numero(mensaje)
    if not cantidad:
        return "¿Cuántas piezas necesitas? (ejemplo: 2)"
    
    if cantidad < 1:
        return "La cantidad debe ser al menos 1"
    
    producto = sesion.producto_seleccionado
    if not producto:
        return "Primero selecciona un producto"
    
    cantidad = int(cantidad)
    total = cantidad * producto['precio_unitario']
    calculo = {
        "cantidad": cantidad,
        "total": total,
        "detalle": f"{cantidad} pieza(s) = ${total:.2f}"
    }
    
    sesion.agregar_item(producto, calculo, cantidad)
    total_general = sesion.get_total()
    
    msg = f"💰 *GRIFERÍA AGREGADA*\n\n"
    msg += f"*{producto['descripcion']}*\n"
    msg += f"📦 {cantidad} pieza(s) x ${producto['precio_unitario']:.2f}\n"
    msg += f"💵 Subtotal: ${total:.2f}\n\n"
    msg += f"💵 *TOTAL COTIZACIÓN: ${total_general:.2f}*\n\n"
    msg += "¿Algo más? (otra grifería, pegamento, pisos) o 'listo'"
    
    sesion.producto_seleccionado = None
    return msg

# ============================================================================
# CLASIFICADOR DE INTENCIONES
# ============================================================================

def detectar_categoria_simple(mensaje: str):
    m = mensaje.lower()
    if any(x in m for x in ["piso", "porcelanato", "baño", "suelo"]):
        return "pisos"
    if any(x in m for x in ["muro", "azulejo", "pared"]):
        return "muros"
    if any(x in m for x in ["grifo", "llave", "regadera", "griferia", "monomando", "ducha", "mezcladora"]):
        return "griferia"
    if any(x in m for x in ["pega", "adhesivo", "boquilla", "cemento", "polvo"]):
        return "polvos"
    return None

def clasificar_intencion(mensaje: str, estado: EstadoConversacion):
    m = mensaje.lower().strip()
    
    if estado.esperando_respuesta == "post_cotizacion":
        m = mensaje.lower().strip()
        if m in ["1", "1️⃣", "nuevo", "nueva", "otra", "empezar", "si"]:
            return {"intencion": "nueva_cotizacion"}
        elif m in ["2", "2️⃣", "salir", "terminar", "adios", "chao"]:
            return {"intencion": "salir"}
        else:
            return {"intencion": "post_cotizacion_repetir"}
    
    if estado.esperando_respuesta == "m2_polvo":
        if extraer_numero(m) is not None:
            return {"intencion": "cotizar_polvo", "m2": extraer_numero(m)}
    
    if estado.esperando_respuesta == "cantidad_griferia":
        if extraer_numero(m) is not None:
            return {"intencion": "cotizar_griferia"}
    
    if estado.producto_seleccionado and extraer_numero(m) is not None:
        return {"intencion": "cotizar"}
    
    if m.isdigit() and 1 <= int(m) <= 5 and estado.ultimos_productos:
        return {"intencion": "seleccionar", "indice": int(m) - 1}
    
    if m in ["mas", "más", "otro", "otra", "diferente", "otros"] and estado.categoria_activa:
        return {"intencion": "buscar_con_qwen", "categoria": estado.categoria_activa, "mensaje_original": mensaje}
    
    if any(x in m for x in ["ayer", "anterior", "otro día", "recuerdas", "habíamos"]):
        return {"intencion": "recordar"}
    
    if any(x in m for x in ["gracias", "listo", "terminamos", "adiós", "chao"]):
        return {"intencion": "despedida"}
    
    if estado.items_cotizacion and any(x in m for x in ["pegamento", "adhesivo", "boquilla", "grifería", "grifo", "llave", "también", "además", "otro", "más", "agrega", "falta"]):
        cat = detectar_categoria_simple(m)
        if not cat:
            if any(x in m for x in ["pegamento", "adhesivo", "boquilla"]):
                cat = "polvos"
                estado.esperando_respuesta = "tipo_polvo"
            elif any(x in m for x in ["grifería", "grifo", "llave", "regadera", "monomando", "ducha"]):
                cat = "griferia"
                return {"intencion": "buscar_griferia"}
        
        if cat == "polvos":
            estado.esperando_respuesta = "tipo_polvo"
            return {"intencion": "buscar_polvo"}
        elif cat == "griferia":
            return {"intencion": "buscar_griferia"}
    
    categoria_qwen = qwen_interpretar_categoria(mensaje)
    return {"intencion": "buscar_con_qwen", "categoria": categoria_qwen, "mensaje_original": mensaje}

# ============================================================================
# RESPUESTAS
# ============================================================================

def resp_recordar(usuario_id, estado):
    hist = memoria_largo_plazo.obtener(usuario_id)
    if not hist["cotizaciones"]:
        return "No tengo registro anterior. ¿Qué buscas?"
    
    ultima = hist["cotizaciones"][-1]
    dias = (datetime.now() - datetime.fromisoformat(ultima["fecha"])).days
    
    msg = f"📚 *BIENVENIDO DE VUELTA {estado.nombre}*\n\n"
    if dias == 0:
        msg += "Hoy ya cotizaste.\n"
    elif dias == 1:
        msg += "Ayer estuviste aquí.\n"
    else:
        msg += f"Última visita: hace {dias} días.\n"
    
    msg += f"💰 Total histórico: ${hist['total_acumulado']:.2f}\n\n"
    msg += "*ÚLTIMA COTIZACIÓN:*\n"
    for i, item in enumerate(ultima["items"][-2:], 1):
        p = item["producto"]
        msg += f"{i}. {p['descripcion'][:30]}... = ${item['calculo']['total']:.2f}\n"
    
    msg += "\n¿Repetir o nueva búsqueda?"
    return msg

def resp_buscar_con_qwen(sesion, clas):
    categoria = clas.get("categoria", "pisos")
    mensaje_original = clas.get("mensaje_original", "")
    
    colecciones_map = {
        "pisos": ["nacionales", "importados"],
        "muros": ["nacionales", "importados"],
        "griferia": ["griferia", "otras"],
        "polvos": ["polvos"]
    }
    
    colecciones = colecciones_map.get(categoria, ["nacionales", "importados"])
    tipo = categoria[:-1] if categoria in ["pisos", "muros"] else None
    
    productos = db.buscar(mensaje_original, colecciones=colecciones, tipo=tipo, top_k=8)
    
    if not productos:
        return f"No encontré {categoria} con '{mensaje_original}'."
    
    sesion.guardar_productos(productos[:5])
    sesion.categoria_activa = categoria
    sesion.esperando_respuesta = "seleccionar_numero"
    
    respuesta_qwen = qwen_responder_con_catalogo(mensaje_original, productos[:5])
    
    if respuesta_qwen:
        return respuesta_qwen
    
    msg = f"🔍 *{categoria.upper()}* - {len(productos[:5])} opciones:\n\n"
    for i, p in enumerate(productos[:5], 1):
        if p['precio_m2'] > 0:
            msg += f"{i}. *{p['descripcion'][:40]}*\n"
            msg += f"   🏢 {p['proveedor']} | 💰 ${p['precio_m2']:.2f}/m²\n\n"
        else:
            msg += f"{i}. *{p['descripcion'][:40]}*\n"
            msg += f"   🏢 {p['proveedor']} | 💰 ${p['precio_unitario']:.2f}/unidad\n\n"
    
    msg += "¿Cuál te interesa? (1-5)"
    return msg

def resp_seleccionar(sesion, clas, mensaje_original=""):
    idx = clas.get("indice", 0)
    
    if idx < 0 or idx >= len(sesion.ultimos_productos):
        return f"Escribe un número del 1 al {len(sesion.ultimos_productos)}"
    
    producto = sesion.seleccionar(idx)
    if not producto:
        return "Primero busca productos."
    
    sesion.categoria_activa = None
    sesion.esperando_respuesta = None
    
    if producto.get('precio_m2', 0) > 0:
        return f"✅ *{producto['descripcion'][:40]}*\n💵 ${producto['precio_m2']:.2f}/m²\n📦 {producto.get('metraje_caja', 1.44)}m² por caja\n\n¿Cuántos m² necesitas?"
    else:
        return f"✅ *{producto['descripcion'][:40]}*\n💵 ${producto['precio_unitario']:.2f}/unidad\n\n¿Cuántas unidades necesitas?"

def resp_cotizar(sesion, mensaje):
    m2 = extraer_numero(mensaje)
    if not m2:
        return "¿Para cuántos m²? (ejemplo: 25)"
    
    if not sesion.producto_seleccionado:
        return "Primero selecciona un producto"
    
    producto = sesion.producto_seleccionado
    calc = calcular_piso(producto, m2)
    sesion.agregar_item(producto, calc, m2)
    
    total = sesion.get_total()
    
    msg = f"💰 *COTIZACIÓN*\n\n"
    msg += f"*{producto['descripcion'][:40]}*\n"
    msg += f"📐 {m2}m² = {calc['detalle']}\n\n"
    msg += f"💵 *TOTAL: ${total:.2f}*\n\n"
    msg += "¿Algo más? (pegamento, grifería) o 'listo'"
    
    return msg

def resp_buscar_polvo(sesion, clas):
    productos = db.buscar("pegamento", colecciones=["polvos"], top_k=5)
    if not productos:
        return "No encontré pegamentos."
    
    sesion.guardar_productos(productos)
    sesion.categoria_activa = "polvos"
    sesion.esperando_respuesta = "seleccionar_polvo"
    
    msg = "🔧 *PEGAMENTOS:*\n\n"
    for i, p in enumerate(productos[:5], 1):
        msg += f"{i}. {p['descripcion'][:40]} - ${p['precio_unitario']:.2f}\n"
    
    msg += "\n¿Cuál necesitas? (1-5)"
    return msg

def resp_seleccionar_polvo(sesion, clas):
    idx = clas.get("indice", 0)
    producto = sesion.seleccionar(idx)
    if not producto:
        return "No encontré ese pegamento."
    
    sesion.producto_seleccionado = producto
    sesion.esperando_respuesta = "m2_polvo"
    
    return f"✅ *{producto['descripcion'][:40]}*\n💵 ${producto['precio_unitario']:.2f}/unidad\n\n¿Para cuántos m²?"

def resp_cotizar_polvo(sesion, clas):
    m2 = clas.get("m2")
    if not m2:
        return "¿Para cuántos m²?"
    
    producto = sesion.producto_seleccionado
    if not producto:
        return "Primero selecciona un pegamento."
    
    rendimiento = 4.5
    sacos = math.ceil(m2 / rendimiento)
    total = sacos * producto['precio_unitario']
    
    calculo = {
        "sacos": sacos,
        "total": total,
        "detalle": f"{sacos} sacos = ${total:.2f}"
    }
    sesion.agregar_item(producto, calculo, m2)
    
    total_general = sesion.get_total()
    
    msg = f"💰 *PEGAMENTO AGREGADO*\n\n"
    msg += f"📦 {calculo['detalle']}\n"
    msg += f"💵 *TOTAL: ${total_general:.2f}*\n\n"
    msg += "¿Algo más?"
    
    sesion.producto_seleccionado = None
    sesion.esperando_respuesta = None
    
    return msg

def resp_despedida(usuario_id, sesion):
    if not sesion.items_cotizacion:
        return "¡Gracias! 👋"
    
    gestor_sesiones.guardar_cotizacion(usuario_id)
    total = sesion.get_total()
    
    fecha = datetime.now().strftime('%d/%m/%Y %H:%M')
    cotizacion_texto = f"COTIZACIÓN {fecha}\nTotal: ${total:.2f}"
    
    msg = f"🎉 *COTIZACIÓN GUARDADA*\n"
    msg += f"💵 *TOTAL: ${total:.2f}*\n\n"
    msg += "1️⃣ *nuevo* para otra\n"
    msg += "2️⃣ *salir* para terminar"
    
    sesion.reset()
    sesion.esperando_respuesta = "post_cotizacion"
    
    return msg

# ============================================================================
# PROCESADOR PRINCIPAL
# ============================================================================

def procesar_mensaje(usuario_id: str, nombre: str, mensaje: str) -> str:
    estado = gestor_sesiones.obtener(usuario_id, nombre)
    
    # --- SALUDO PERSONALIZADO (solo si NO se ha enviado antes) ---
    if not estado.saludo_enviado:
        estado.saludo_enviado = True
        hist = memoria_largo_plazo.obtener(usuario_id)
        
        if hist["cotizaciones"]:
            ultima = hist["cotizaciones"][-1]
            fecha_ultima = datetime.fromisoformat(ultima["fecha"])
            dias = (datetime.now() - fecha_ultima).days
            total_hist = hist['total_acumulado']
            
            saludo = f"👋 *¡Hola {nombre}!*\n\n"
            
            if dias == 0:
                saludo += "Veo que hoy ya cotizaste con nosotros. ¿Quieres continuar donde te quedaste o empezar algo nuevo?"
            elif dias == 1:
                saludo += "¡Qué rápido! Apenas ayer estuviste por aquí. ¿Necesitas retomar tu cotización?"
            elif dias <= 7:
                saludo += f"Bienvenido de vuelta. Hace {dias} días que no te veíamos. ¿Continuamos con tu última cotización?"
            else:
                saludo += f"¡Qué gusto verte de nuevo! Han pasado {dias} días desde tu última visita."
            
            saludo += f"\n\n📋 *Tu última cotización* (${ultima['total']:.2f}):\n"
            for i, item in enumerate(ultima["items"][-2:], 1):
                p = item["producto"]
                saludo += f"{i}. {p['descripcion'][:40]}... ${item['calculo']['total']:.2f}\n"
            
            if len(ultima["items"]) > 2:
                saludo += f"... y {len(ultima['items']) - 2} producto(s) más\n"
            
            saludo += f"\n💰 *Total histórico:* ${total_hist:.2f}\n\n"
            saludo += "¿Qué prefieres?\n"
            saludo += "1️⃣ *Continuar* (sigo con mi cotización anterior)\n"
            saludo += "2️⃣ *Nuevo* (empezar de cero)\n"
            saludo += "3️⃣ *Buscar* (directo a buscar productos)"
            
            estado.esperando_respuesta = "saludo_inicial"
            return saludo
    
    # --- MANEJO DE RESPUESTA AL SALUDO ---
    if estado.esperando_respuesta == "saludo_inicial":
        m = mensaje.lower().strip()
        estado.esperando_respuesta = None  # Salir del modo saludo SIEMPRE
        
        if m in ["1", "1️⃣", "continuar", "si", "sí", "seguir"]:
            return "Perfecto, continuemos con tu cotización anterior. ¿Qué necesitas agregar o modificar?"
        elif m in ["2", "2️⃣", "nuevo", "nueva", "cero", "empezar"]:
            estado.reset()
            return "🆕 Empezamos de cero. ¿Qué buscas? (pisos, azulejos, grifería, pegamento)"
        else:
            # Cualquier otra cosa (incluyendo "3") va a búsqueda
            pass
    
    # --- CLASIFICACIÓN NORMAL ---
    clas = clasificar_intencion(mensaje, estado)
    print(f"[{usuario_id[-10:]}] {clas['intencion']}: {mensaje[:40]}...")
    
    # ✅ AHORA SÍ podemos usar 'clas'
    if clas["intencion"] == "post_cotizacion_repetir":
        return "¿Nuevo o salir? (1 = nuevo, 2 = salir)"
    
    if clas["intencion"] == "salir":
        estado.esperando_respuesta = None
        return "¡Gracias! 👋"
    
    if clas["intencion"] == "nueva_cotizacion":
        estado.reset()
        return "🆕 Nueva cotización. ¿Qué buscas?"
    
    if clas["intencion"] == "post_cotizacion":
        return "¿Nuevo o salir?"
    
    if clas["intencion"] == "recordar":
        return resp_recordar(usuario_id, estado)
    
    if clas["intencion"] == "seleccionar":
        return resp_seleccionar(estado, clas, mensaje)
    
    if clas["intencion"] == "buscar_polvo":
        return resp_buscar_polvo(estado, clas)
    
    if clas["intencion"] == "seleccionar_polvo":
        return resp_seleccionar_polvo(estado, clas)
    
    if clas["intencion"] == "cotizar_polvo":
        return resp_cotizar_polvo(estado, clas)
    
    if clas["intencion"] == "buscar_griferia":
        return resp_buscar_griferia(estado, mensaje, None)
    
    if clas["intencion"] == "seleccionar_griferia":
        return resp_seleccionar_griferia(estado, clas, mensaje)
    
    if clas["intencion"] == "cotizar_griferia":
        return resp_cotizar_griferia(estado, mensaje)
    
    if clas["intencion"] == "cotizar":
        return resp_cotizar(estado, mensaje)
    
    if clas["intencion"] == "despedida":
        return resp_despedida(usuario_id, estado)
    
    return resp_buscar_con_qwen(estado, clas)

# ============================================================================
# MODO CONSOLA
# ============================================================================

def modo_consola():
    print("="*60)
    print("🏪 VAMA 2.0")
    print("="*60)
    print("Usuarios: 1=Papá, 2=Mamá, 3=Hermano, 4=Tú")
    print("Comandos: 'C'=Cambiar, 'S'=Salir")
    print("="*60)
    
    usuarios = {
        "1": ("5215512345678", "Papá"),
        "2": ("5215598765432", "Mamá"),
        "3": ("5215522223333", "Hermano"),
        "4": ("5215544445555", "Tú")
    }
    
    usuario_actual = None
    nombre_actual = None
    id_actual = None
    
    while True:
        if not usuario_actual:
            opt = input("\n¿Qué usuario? (1-4, S=salir): ").strip().upper()
            if opt == "S":
                break
            if opt not in usuarios:
                print("❌ Inválido")
                continue
            
            id_actual, nombre_actual = usuarios[opt]
            usuario_actual = gestor_sesiones.obtener(id_actual, nombre_actual)
            print(f"\n👤 {nombre_actual} activo")
        
        mensaje = input(f"👤 {nombre_actual}: ").strip()
        if not mensaje:
            continue
        
        if mensaje.upper() == "S":
            break
        elif mensaje.upper() == "C":
            usuario_actual = None
            continue
        
        respuesta = procesar_mensaje(id_actual, nombre_actual, mensaje)
        print(f"\n🤖 VAMA:\n{respuesta}\n")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        from flask import Flask, request, jsonify
        app = Flask(__name__)
        
        @app.route('/webhook', methods=['POST'])
        def webhook():
            data = request.json
            telefono = data.get('telefono', 'unknown')
            nombre = data.get('nombre', 'Cliente')
            mensaje = data.get('mensaje', '')
            respuesta = procesar_mensaje(telefono, nombre, mensaje)
            return jsonify({"respuesta": respuesta})
        
        print("🚀 API en puerto 5000")
        app.run(host='0.0.0.0', port=5000)
    else:
        modo_consola()