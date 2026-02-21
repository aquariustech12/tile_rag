#!/usr/bin/env python3
"""
VAMA PRO 3.0 - LLM + RAG + Memoria Persistente
Código fusionado y corregido
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
from typing import Dict, List, Optional
from datetime import datetime, timedelta

CHROMA_PATH = "chroma_db_v3"
MEMORIA_PATH = "memoria_vama.pkl"

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
# MEMORIA PERSISTENTE (LARGO PLAZO)
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
            except:
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
# ESTADO DE CONVERSACIÓN (CORTO PLAZO)
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
        self.esperando_respuesta = None  # Contexto de conversación
    
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

# ============================================================================
# GESTOR DE SESIONES
# ============================================================================

class GestorSesiones:
    def __init__(self):
        self.sesiones = {}
        self.expiracion = timedelta(hours=2)
    
    def obtener(self, usuario_id, nombre):
        ahora = datetime.now()
        
        # Limpiar expirados
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
# CÁLCULOS
# ============================================================================

def calcular_piso(producto, m2):
    cajas = math.ceil(m2 / producto["metraje_caja"])
    total = cajas * producto["precio_m2"] * producto["metraje_caja"]
    return {"cajas": cajas, "total": total, "detalle": f"{cajas} cajas = ${total:.2f}"}

# ============================================================================
# VAMA LLM - PROMPT RESTRICTIVO
# ============================================================================

class VAMALLM:
    def __init__(self, usar_llm=True):
        self.usar_llm = usar_llm
        self.modelo = "qwen3:30b-a3b-fp16"
        self.estado = None
    
    def set_estado(self, estado):
        self.estado = estado
    
    def buscar_productos(self, query, categoria):
        """Busca en ChromaDB primero"""
        if categoria == "pisos":
            return db.buscar(query, tipo="piso", top_k=5)
        elif categoria == "muros":
            return db.buscar(query, tipo="muro", top_k=5)
        elif categoria == "polvos":
            return db.buscar(query, colecciones=["polvos"], top_k=5)
        elif categoria == "griferia":
            return db.buscar(query, colecciones=["griferia"], top_k=5)
        else:
            return db.buscar(query, top_k=5)
    
    def formatear_productos(self, productos):
        """Formatea productos para el prompt"""
        if not productos:
            return "=== NO HAY PRODUCTOS EN EL CATÁLOGO PARA ESTA BÚSQUEDA ==="
        
        lineas = ["=== CATÁLOGO VAMA - USAR SOLO ESTOS PRODUCTOS ==="]
        for i, p in enumerate(productos, 1):
            if p['precio_m2'] > 0:
                precio_caja = p['precio_m2'] * p['metraje_caja']
                lineas.append(f"{i}. {p['descripcion']} | {p['proveedor']} | ${p['precio_m2']:.2f}/m² | Caja: ${precio_caja:.2f}")
            else:
                lineas.append(f"{i}. {p['descripcion']} | {p['proveedor']} | ${p['precio_unitario']:.2f}/unidad")
        lineas.append("=== FIN CATÁLOGO ===")
        return "\n".join(lineas)
    
    def generar_prompt(self, mensaje, productos, contexto_cotizacion=""):
        """Prompt ultra restrictivo"""
        
        productos_txt = self.formatear_productos(productos)
        
        prompt = f"""{productos_txt}

REGLAS ABSOLUTAS:
1. SOLO puedes hablar de los productos listados arriba
2. NO inventes productos que no estén en el catálogo
3. NO hables de cerámica genérica, porcelanato genérico, o piedra natural
4. Si el cliente pide algo que no está en la lista, di: "No tengo ese producto en catálogo"
5. Usa los precios EXACTOS de la lista
6. Recomienda máximo 3 productos del catálogo
7. Termina preguntando: "¿Cuál te interesa? (1-{len(productos)})"

{contexto_cotizacion}

CLIENTE: {mensaje}

VAMA (usa SOLO el catálogo de arriba):"""
        
        return prompt
    
    def procesar(self, mensaje, categoria=None):
        """Procesa mensaje: Chroma primero, LLM segundo"""
        if not self.usar_llm:
            return None
        
        # 1. SIEMPRE buscar en Chroma primero
        productos = self.buscar_productos(mensaje, categoria)
        
        # Guardar en estado para selección posterior
        if self.estado:
            self.estado.guardar_productos(productos)
            self.estado.ultimos_productos = productos
        
        # 2. Construir contexto de cotización si hay items previos
        contexto = ""
        if self.estado and self.estado.items_cotizacion:
            total = self.estado.get_total()
            contexto = f"\nCOTIZACIÓN ACTUAL: ${total:.2f} | {len(self.estado.items_cotizacion)} productos"
        
        # 3. Generar prompt restrictivo
        prompt = self.generar_prompt(mensaje, productos, contexto)
        
        # 4. Llamar a LLM
        try:
            respuesta = ollama.generate(
                model=self.modelo,
                prompt=prompt,
                options={'temperature': 0.1, 'num_predict': 400}
            )['response'].strip()
            
            # Verificar que no haya alucinado
            if any(x in respuesta.lower() for x in ["cerámica genérica", "porcelanato común", "piedra natural", "vinilo", "opciones comunes"]):
                print("   ⚠️ LLM alucinó, usando fallback")
                return None
            
            return respuesta
            
        except Exception as e:
            print(f"⚠️ Error LLM: {e}")
            return None

vama_llm = VAMALLM(usar_llm=False)

# ============================================================================
# CLASIFICADOR CON CONTEXTO - CORREGIDO
# ============================================================================

def clasificar_intencion(mensaje: str, estado: EstadoConversacion):
    m = mensaje.lower().strip()
    
    # Si está esperando selección de número
    if estado.esperando_respuesta == "seleccionar_numero":
        if m in ["mas", "más", "otro", "otra", "diferente", "otros"]:
            return {"intencion": "buscar", "categoria": estado.categoria_activa}
        
        if m.isdigit():
            return {"intencion": "seleccionar", "indice": int(m) - 1}
        
        # Cualquier otra cosa, intentar seleccionar por texto
        return {"intencion": "seleccionar", "indice": -1, "texto": mensaje}
    
    # Si está esperando una respuesta específica (contexto)
    if estado.esperando_respuesta:
        if estado.esperando_respuesta == "tipo_polvo":
            # Cualquier respuesta ahora se busca en polvos
            return {"intencion": "buscar_polvo", "categoria": "polvos", "contexto": "esperando_tipo"}
        elif estado.esperando_respuesta == "m2_polvo":
            # Esperando metros cuadrados para polvo
            if re.search(r'\d+', m):
                return {"intencion": "cotizar_polvo", "categoria": "polvos", "m2": extraer_numero(m)}
        estado.esperando_respuesta = None
    
    # Detectar número de selección
    if m.isdigit() and 1 <= int(m) <= 5 and estado.ultimos_productos:
        # Si estábamos esperando seleccionar polvo
        if estado.categoria_activa == "polvos" and estado.items_cotizacion:
            return {"intencion": "seleccionar_polvo", "indice": int(m) - 1}
        return {"intencion": "seleccionar", "indice": int(m) - 1}
    
    # Recordar
    if any(x in m for x in ["ayer", "anterior", "otro día", "recuerdas", "habíamos"]):
        return {"intencion": "recordar"}
    
    # Despedida
    if any(x in m for x in ["gracias", "listo", "terminamos", "adiós", "chao"]):
        return {"intencion": "despedida"}
    
    # Agregar complemento - CORREGIDO: detectar productos directamente
    if estado.items_cotizacion and any(x in m for x in ["pegamento", "adhesivo", "boquilla", "grifería", "grifo", "llave", "también", "además", "otro", "más", "agrega", "falta"]):
        cat = detectar_categoria(m)
        # Forzar categoría si no se detectó pero hay palabras clave
        if not cat:
            if any(x in m for x in ["pegamento", "adhesivo", "boquilla"]):
                cat = "polvos"
            elif any(x in m for x in ["grifería", "grifo", "llave", "regadera"]):
                cat = "griferia"
        
        if cat == "polvos":
            estado.esperando_respuesta = "tipo_polvo"
        return {"intencion": "agregar", "categoria": cat}
    
    # Buscar
    if any(x in m for x in ["busco", "tienes", "opciones", "muéstrame", "necesito", "dame"]):
        cat = detectar_categoria(m)
        return {"intencion": "buscar", "categoria": cat}
    
    # Cotizar
    if re.search(r'\d+', m) and any(x in m for x in ["m2", "metros", "precio", "cuesta", "cotizar"]):
        return {"intencion": "cotizar"}
    
    # Por defecto, si hay productos previos, asumir selección o cotización
    if estado.ultimos_productos:
        if estado.categoria_activa == "polvos":
            return {"intencion": "seleccionar_polvo", "indice": 0}
        return {"intencion": "cotizar"}
    
    return {"intencion": "buscar", "categoria": detectar_categoria(m)}

def extraer_numero(mensaje):
    match = re.search(r'(\d+(?:\.\d+)?)', mensaje.replace(",", "."))
    if match:
        return float(match.group(1))
    return None

def detectar_categoria(mensaje: str):
    m = mensaje.lower()
    if any(x in m for x in ["piso", "porcelanato", "baño", "suelo"]):
        return "pisos"
    if any(x in m for x in ["muro", "azulejo", "pared"]):
        return "muros"
    if any(x in m for x in ["grifo", "llave", "regadera", "griferia"]):
        return "griferia"
    if any(x in m for x in ["pega", "adhesivo", "boquilla", "cemento", "polvo"]):
        return "polvos"
    return None

# ============================================================================
# RESPUESTAS DETERMINISTAS (FALLBACK)
# ============================================================================

def resp_recordar(usuario_id, estado):
    hist = memoria_largo_plazo.obtener(usuario_id)
    
    if not hist["cotizaciones"]:
        return "No tengo registro anterior. Empecemos de nuevo. ¿Qué buscas?"
    
    ultima = hist["cotizaciones"][-1]
    dias = (datetime.now() - datetime.fromisoformat(ultima["fecha"])).days
    
    msg = f"📚 *BIENVENIDO DE VUELTA {estado.nombre}*\n\n"
    if dias == 0:
        msg += "Hoy ya cotizaste con nosotros.\n"
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

def resp_buscar(sesion, clas):
    categoria = clas.get("categoria")
    
    if not categoria:
        return "¿Qué buscas? (pisos, azulejos, grifería, pegamento)"
    
    # Buscar en Chroma
    productos = db.buscar(categoria, tipo=categoria[:-1] if categoria in ["pisos", "muros"] else None, top_k=5)
    
    if not productos:
        return f"No encontré {categoria}. Intenta con otras palabras como color, tamaño o estilo."
    
    sesion.guardar_productos(productos)
    sesion.categoria_activa = categoria
    sesion.esperando_respuesta = "seleccionar_numero"  # Contexto: esperando número
    
    msg = f"🔍 *{categoria.upper()}* - {len(productos)} opciones:\n\n"
    for i, p in enumerate(productos[:5], 1):
        if p['precio_m2'] > 0:
            precio_caja = p['precio_m2'] * p['metraje_caja']
            msg += f"{i}. *{p['descripcion'][:40]}*\n"
            msg += f"   🏢 {p['proveedor']} | 💰 ${p['precio_m2']:.2f}/m²\n\n"
        else:
            msg += f"{i}. *{p['descripcion'][:40]}*\n"
            msg += f"   🏢 {p['proveedor']} | 💰 ${p['precio_unitario']:.2f}/unidad\n\n"
    
    msg += "¿Cuál te interesa? Escribe el número (1-5)\n"
    msg += "O escribe 'más' para ver otras opciones, 'otro' para buscar diferente"
    return msg

def resp_seleccionar(sesion, clas, mensaje_original=""):
    # Si llegó un número directo
    idx = clas.get("indice", 0)
    
    # Si no hay índice válido pero hay texto, buscar coincidencia
    if idx == -1 and mensaje_original:
        m = mensaje_original.lower()
        coincidencias = []
        for i, p in enumerate(sesion.ultimos_productos, 1):
            if m in p['descripcion'].lower() or m in p['proveedor'].lower():
                coincidencias.append(i)
        
        if len(coincidencias) == 1:
            idx = coincidencias[0] - 1
        elif len(coincidencias) > 1:
            return f"Encontré {len(coincidencias)} productos con '{mensaje_original}'. Por favor escribe solo el número (1-5)"
        else:
            return f"No reconocí '{mensaje_original}'. Por favor escribe el número del 1 al {len(sesion.ultimos_productos)}, o 'más' para ver otras opciones"
    
    # Validar rango
    if idx < 0 or idx >= len(sesion.ultimos_productos):
        return f"Por favor escribe un número del 1 al {len(sesion.ultimos_productos)}"

    # Si no hay número pero hay texto, buscar coincidencia
    if not mensaje_original.isdigit() and mensaje_original:
        m = mensaje_original.lower()
        coincidencias = []
        for i, p in enumerate(sesion.ultimos_productos, 1):
            if m in p['descripcion'].lower() or m in p['proveedor'].lower():
                coincidencias.append(i)
        
        if len(coincidencias) == 1:
            idx = coincidencias[0] - 1
        elif len(coincidencias) > 1:
            return f"Encontré {len(coincidencias)} productos con '{mensaje_original}'. Por favor escribe solo el número (1-5)"
        else:
            return f"No reconocí '{mensaje_original}'. Por favor escribe el número del 1 al {len(sesion.ultimos_productos)}, o 'más' para ver otras opciones"
    
    # Validar rango
    if idx < 0 or idx >= len(sesion.ultimos_productos):
        return f"Opción no válida. Por favor escribe un número del 1 al {len(sesion.ultimos_productos)}"
    
    producto = sesion.seleccionar(idx)
    if not producto:
        return "Primero busca productos. ¿Qué necesitas?"
    
    sesion.categoria_activa = None
    sesion.esperando_respuesta = None
    
    if producto['precio_m2'] > 0:
        return f"✅ *{producto['descripcion'][:40]}*\n💵 ${producto['precio_m2']:.2f}/m²\n📦 {producto['metraje_caja']}m² por caja\n\n¿Cuántos m² necesitas?"
    else:
        return f"✅ *{producto['descripcion'][:40]}*\n💵 ${producto['precio_unitario']:.2f}/unidad\n\n¿Cuántas unidades?"

def resp_cotizar(sesion, mensaje):
    m2 = extraer_numero(mensaje)
    if not m2:
        return "¿Para cuántos m²? (ejemplo: 25)"
    
    if not sesion.producto_seleccionado:
        if sesion.ultimos_productos:
            sesion.producto_seleccionado = sesion.ultimos_productos[0]
        else:
            return "Primero selecciona un producto (1-5)"
    
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

# NUEVO: Buscar polvos específicamente
def resp_buscar_polvo(sesion, clas):
    productos = db.buscar("pegamento", colecciones=["polvos"], top_k=5)
    
    if not productos:
        return "No encontré pegamentos. Intenta con otra búsqueda."
    
    sesion.guardar_productos(productos)
    sesion.categoria_activa = "polvos"
    sesion.esperando_respuesta = "seleccionar_polvo"  # Esperar que elija uno
    
    msg = "🔧 *PEGAMENTOS DISPONIBLES:*\n\n"
    for i, p in enumerate(productos[:5], 1):
        msg += f"{i}. {p['descripcion'][:40]}\n"
        msg += f"   💰 ${p['precio_unitario']:.2f}/unidad | {p['proveedor']}\n\n"
    
    msg += "¿Cuál necesitas? (1-5)"
    return msg

# NUEVO: Seleccionar polvo y preguntar m²
def resp_seleccionar_polvo(sesion, clas):
    idx = clas.get("indice", 0)
    producto = sesion.seleccionar(idx)
    
    if not producto:
        return "No encontré ese pegamento. Intenta de nuevo."
    
    # Guardar como producto seleccionado temporalmente
    sesion.producto_seleccionado = producto
    sesion.esperando_respuesta = "m2_polvo"
    
    return f"✅ *{producto['descripcion'][:40]}*\n💵 ${producto['precio_unitario']:.2f}/unidad\n\n¿Para cuántos m² de piso necesitas pegamento?"

# NUEVO: Cotizar polvo (sacos necesarios)
def resp_cotizar_polvo(sesion, clas):
    m2 = clas.get("m2") or extraer_numero(input("m2: "))  # Fallback
    
    if not m2:
        return "¿Para cuántos m² necesitas el pegamento?"
    
    producto = sesion.producto_seleccionado
    if not producto:
        return "Primero selecciona un pegamento."
    
    # Calcular sacos (rendimiento aproximado 4-5 m² por saco de 20kg)
    rendimiento = 4.5  # m² por saco
    sacos = math.ceil(m2 / rendimiento)
    total = sacos * producto['precio_unitario']
    
    # Agregar a cotización
    calculo = {
        "sacos": sacos,
        "total": total,
        "detalle": f"{sacos} sacos (rinde ~{sacos * rendimiento}m²) = ${total:.2f}"
    }
    sesion.agregar_item(producto, calculo, m2)
    
    total_general = sesion.get_total()
    
    msg = f"💰 *PEGAMENTO AGREGADO*\n\n"
    msg += f"*{producto['descripcion'][:40]}*\n"
    msg += f"📐 Para {m2}m² de piso\n"
    msg += f"📦 {calculo['detalle']}\n\n"
    msg += f"💵 *TOTAL COTIZACIÓN: ${total_general:.2f}*\n\n"
    msg += "¿Algo más? (otro pegamento, grifería) o 'listo'"
    
    # Limpiar selección temporal
    sesion.producto_seleccionado = None
    sesion.categoria_activa = None
    sesion.esperando_respuesta = None
    
    return msg

# ============================================================================
# NUEVO: FLUJO COMPLETO DE GRIFERÍA
# ============================================================================

def detectar_tipo_griferia(mensaje):
    """Detecta qué tipo de grifería busca el usuario"""
    m = mensaje.lower()
    
    if any(x in m for x in ["monomando", "mezcladora", "una perilla", "una manija"]):
        return "monomando"
    elif any(x in m for x in ["regadera", "ducha", "teléfono", "telefono"]):
        return "regadera"
    elif any(x in m for x in ["llave", "tarja", "fregadero", "cocina"]):
        return "llave"
    elif any(x in m for x in ["válvula", "valvula", "escape", "fluxor"]):
        return "valvula"
    else:
        return None

def formatear_griferia(productos):
    """Formatea productos de grifería para mostrar"""
    if not productos:
        return ""
    
    msg = "🚿 *GRIFERÍA DISPONIBLE:*\n\n"
    for i, p in enumerate(productos[:5], 1):
        # Extraer tipo de la descripción si es posible
        desc = p['descripcion']
        if "MONOMANDO" in desc.upper():
            tipo = "🚰 Monomando"
        elif "REGADERA" in desc.upper() or "DUCHA" in desc.upper():
            tipo = "🚿 Regadera"
        elif "LLAVE" in desc.upper():
            tipo = "🔧 Llave"
        else:
            tipo = "🚿 Grifo"
        
        color = p.get('color', '').capitalize() if p.get('color') else ''
        color_txt = f" | 🎨 {color}" if color else ""
        
        msg += f"{i}. *{desc[:50]}*\n"
        msg += f"   {tipo} | 🏢 {p['proveedor']}{color_txt}\n"
        msg += f"   💰 ${p['precio_unitario']:.2f} c/u\n\n"
    
    msg += "¿Cuál te interesa? Escribe el número (1-5)\n"
    msg += "O dime más específico: ¿monomando, regadera o llave?"
    return msg

def resp_buscar_griferia(sesion, mensaje, tipo_especifico=None):
    """Busca grifería en la base de datos"""
    # Determinar el tipo de búsqueda
    if not tipo_especifico:
        tipo_especifico = detectar_tipo_griferia(mensaje)
    
    # Construir query mejorada
    query = mensaje
    if tipo_especifico == "monomando":
        query = "monomando " + mensaje
    elif tipo_especifico == "regadera":
        query = "regadera " + mensaje
    elif tipo_especifico == "llave":
        query = "llave " + mensaje
    
    # Buscar en colecciones relevantes
    productos = []
    for col_name in ["griferia", "otras"]:  # Ambas pueden tener grifería
        if col_name in db.cols:
            try:
                col = db.cols[col_name]
                r = col.query(query_texts=[query], n_results=10)
                
                for meta, dist, pid in zip(r["metadatas"][0], r["distances"][0], r["ids"][0]):
                    # Solo incluir si parece grifería
                    desc = meta.get("descripcion", "").upper()
                    if any(x in desc for x in ["MONOMANDO", "REGADERA", "LLAVE", "GRIFO", "DUCHA", "MEZCLADORA"]):
                        productos.append({
                            "codigo": meta.get("codigo", ""),
                            "descripcion": meta.get("descripcion", ""),
                            "proveedor": meta.get("proveedor", ""),
                            "precio_unitario": float(meta.get("precio_unitario", 0)) if meta.get("precio_unitario") else 0,
                            "color": meta.get("color", ""),
                            "coleccion": col_name
                        })
            except Exception as e:
                print(f"⚠️ Error buscando en {col_name}: {e}")
                continue
    
    # Si no hay resultados, búsqueda más amplia
    if not productos:
        # Buscar sin filtrar por tipo
        for col_name in ["griferia", "otras"]:
            if col_name in db.cols:
                try:
                    col = db.cols[col_name]
                    r = col.query(query_texts=[mensaje], n_results=10)
                    for meta, dist, pid in zip(r["metadatas"][0], r["distances"][0], r["ids"][0]):
                        productos.append({
                            "codigo": meta.get("codigo", ""),
                            "descripcion": meta.get("descripcion", ""),
                            "proveedor": meta.get("proveedor", ""),
                            "precio_unitario": float(meta.get("precio_unitario", 0)) if meta.get("precio_unitario") else 0,
                            "color": meta.get("color", ""),
                            "coleccion": col_name
                        })
                except:
                    continue
    
    # Quitar duplicados
    vistos = set()
    unicos = []
    for p in productos:
        if p["codigo"] and p["codigo"] not in vistos:
            vistos.add(p["codigo"])
            unicos.append(p)
        elif not p["codigo"] and p["descripcion"] not in vistos:
            vistos.add(p["descripcion"])
            unicos.append(p)
    
    if not unicos:
        return None
    
    # Guardar en sesión
    sesion.guardar_productos(unicos[:5])
    sesion.categoria_activa = "griferia"
    sesion.esperando_respuesta = "seleccionar_griferia"
    
    return formatear_griferia(unicos[:5])

def resp_seleccionar_griferia(sesion, clas, mensaje_original=""):
    """Selecciona un producto de grifería"""
    # Si llegó un número directo
    idx = clas.get("indice", -1)
    
    # Si no hay índice pero hay texto, buscar coincidencia
    if idx < 0 and mensaje_original:
        m = mensaje_original.lower()
        # Buscar coincidencias en descripción o proveedor
        for i, p in enumerate(sesion.ultimos_productos, 1):
            if m in p['descripcion'].lower() or m in p['proveedor'].lower():
                idx = i - 1
                break
        
        # Si no hay coincidencia, preguntar de nuevo
        if idx < 0:
            return f"No encontré '{mensaje_original}'. Por favor escribe el número del 1 al {len(sesion.ultimos_productos)}"
    
    # Validar rango
    if idx < 0 or idx >= len(sesion.ultimos_productos):
        return f"Por favor escribe un número del 1 al {len(sesion.ultimos_productos)}"
    
    producto = sesion.seleccionar(idx)
    if not producto:
        return "Primero busca grifería. ¿Qué necesitas?"
    
    # Guardar selección y preguntar cantidad
    sesion.producto_seleccionado = producto
    sesion.esperando_respuesta = "cantidad_griferia"
    
    # Mostrar detalles
    color = f" 🎨 {producto['color'].capitalize()}" if producto.get('color') else ""
    msg = f"✅ *{producto['descripcion'][:50]}*\n"
    msg += f"🏢 {producto['proveedor']}{color}\n"
    msg += f"💰 ${producto['precio_unitario']:.2f} por pieza\n\n"
    msg += "¿Cuántas piezas necesitas? (ejemplo: 2)"
    
    return msg

def resp_cotizar_griferia(sesion, mensaje):
    """Cotiza la cantidad de grifería seleccionada"""
    # Extraer número del mensaje
    cantidad = extraer_numero(mensaje)
    
    if not cantidad:
        return "¿Cuántas piezas necesitas? (ejemplo: 2)"
    
    if cantidad < 1:
        return "La cantidad debe ser al menos 1"
    
    producto = sesion.producto_seleccionado
    if not producto:
        return "Primero selecciona un producto de grifería"
    
    # Redondear a entero (no se pueden comprar medias piezas)
    cantidad = int(cantidad)
    
    # Calcular total
    total = cantidad * producto['precio_unitario']
    calculo = {
        "cantidad": cantidad,
        "total": total,
        "detalle": f"{cantidad} pieza(s) = ${total:.2f}"
    }
    
    # Agregar a cotización
    sesion.agregar_item(producto, calculo, cantidad)
    
    # Calcular total general
    total_general = sesion.get_total()
    
    # Preparar mensaje
    msg = f"💰 *GRIFERÍA AGREGADA*\n\n"
    msg += f"*{producto['descripcion'][:50]}*\n"
    msg += f"📦 {cantidad} pieza(s) x ${producto['precio_unitario']:.2f}\n"
    msg += f"💵 Subtotal: ${total:.2f}\n\n"
    msg += f"💵 *TOTAL COTIZACIÓN: ${total_general:.2f}*\n\n"
    msg += "¿Algo más? (otra grifería, pegamento, pisos) o 'listo'"
    
    # Limpiar selección temporal
    sesion.producto_seleccionado = None
    
    return msg

# MODIFICAR la función clasificar_intencion (agregar casos para grifería)
# Busca la función y en la sección de "Agregar complemento", REEMPLAZA el bloque
# actual (líneas aprox 380-390) con este:

"""
    # Agregar complemento - CORREGIDO: detectar productos directamente
    if estado.items_cotizacion and any(x in m for x in ["pegamento", "adhesivo", "boquilla", "grifería", "grifo", "llave", "también", "además", "otro", "más", "agrega", "falta"]):
        cat = detectar_categoria(m)
        # Forzar categoría si no se detectó pero hay palabras clave
        if not cat:
            if any(x in m for x in ["pegamento", "adhesivo", "boquilla"]):
                cat = "polvos"
            elif any(x in m for x in ["grifería", "grifo", "llave", "regadera", "monomando", "ducha"]):
                cat = "griferia"
        
        if cat == "polvos":
            estado.esperando_respuesta = "tipo_polvo"
        elif cat == "griferia":
            # Para grifería, determinar tipo específico
            tipo = detectar_tipo_griferia(m)
            return {"intencion": "buscar_griferia", "categoria": "griferia", "tipo": tipo}
        
        return {"intencion": "agregar", "categoria": cat}
"""

# MODIFICAR la función detectar_categoria (agregar grifería)
# Busca la función y REEMPLAZA con esta versión:

def detectar_categoria(mensaje: str):
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

def resp_agregar(sesion, clas):
    if not sesion.items_cotizacion:
        return "Primero cotiza un producto. ¿Qué piso necesitas?"
    
    categoria = clas.get("categoria", "polvos")
    
    if categoria == "polvos":
        return resp_buscar_polvo(sesion, clas)
    elif categoria == "griferia":
        return "🚿 *GRIFERÍA*\n\n¿Qué necesitas?\n• Monomando\n• Regadera\n• Llave de tarja\n\nEspecifica cuál:"
    
    return "¿Qué quieres agregar? (pegamento, grifería, etc.)"

def resp_despedida(usuario_id, sesion):
    if not sesion.items_cotizacion:
        return "¡Gracias! VAMA https://vama.com.mx  👋"
    
    gestor_sesiones.guardar_cotizacion(usuario_id)
    total = sesion.get_total()
    
    # Generar texto copiable
    fecha = datetime.now().strftime('%d/%m/%Y %H:%M')
    cotizacion_texto = f"""COTIZACIÓN VAMA
Fecha: {fecha}
Cliente: {sesion.nombre}
{'='*40}"""
    
    for i, item in enumerate(sesion.items_cotizacion, 1):
        p = item["producto"]
        cotizacion_texto += f"\n{i}. {p['descripcion'][:40]}"
        if 'cajas' in item['calculo']:
            cotizacion_texto += f"\n   {item['calculo']['cajas']} cajas - ${item['calculo']['total']:.2f}"
        else:
            cotizacion_texto += f"\n   {item['calculo'].get('sacos', item['calculo'].get('unidades', 1))} un - ${item['calculo']['total']:.2f}"
    
    cotizacion_texto += f"\n{'='*40}\nTOTAL: ${total:.2f}\nVAMA https://vama.com.mx "
    
    # Guardar también como archivo temporal para descargar
    archivo_cotizacion = f"cotizacion_{usuario_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(archivo_cotizacion, 'w', encoding='utf-8') as f:
        f.write(cotizacion_texto)
    
    msg = f"🎉 *COTIZACIÓN GUARDADA*\n\n"
    msg += f"💵 *TOTAL: ${total:.2f}*\n"
    msg += f"📋 {len(sesion.items_cotizacion)} productos\n\n"
    
    msg += "*COPIA TU COTIZACIÓN:*\n"
    msg += "```\n"
    msg += cotizacion_texto
    msg += "\n```\n\n"
    
    msg += f"📄 También guardada en: `{archivo_cotizacion}`\n\n"
    msg += "*¿Qué sigue?*\n"
    msg += "1️⃣ Escribe *nuevo* para otra cotización\n"
    msg += "2️⃣ Escribe *salir* para terminar\n"
    msg += "3️⃣ O escribe *pdf* para generar PDF (próximamente)"
    
    sesion.reset()
    sesion.esperando_respuesta = "post_cotizacion"
    
    return msg

def resp_post_cotizacion(estado, clas, mensaje):
    """Maneja opciones después de guardar cotización"""
    m = mensaje.lower().strip()
    
    if m in ["salir", "adios", "adiós", "chao", "terminar"]:
        estado.esperando_respuesta = None
        return "¡Gracias por usar VAMA! 👋\nTu cotización quedó guardada.\n\nVAMA https://vama.com.mx "
    
    if m == "pdf":
        return "📄 Función PDF en desarrollo.\n\nEscribe *nuevo* para otra cotización o *salir* para terminar."
    
    if m in ["nuevo", "nueva", "otra", "empezar", "si", "sí"]:
        estado.esperando_respuesta = None
        estado.reset()
        return "🆕 Nueva cotización. ¿Qué buscas? (pisos, azulejos, grifería, pegamento)"
    
    # Cualquier otra cosa
    return "¿Qué sigue?\n1️⃣ *nuevo* para otra cotización\n2️⃣ *salir* para terminar"

# ============================================================================
# PROCESADOR PRINCIPAL - CORREGIDO
# ============================================================================

def procesar_mensaje(usuario_id: str, nombre: str, mensaje: str) -> str:
    # Obtener sesión
    estado = gestor_sesiones.obtener(usuario_id, nombre)
    
    # Clasificar intención (con contexto)
    clas = clasificar_intencion(mensaje, estado)
    print(f"[{usuario_id[-10:]}] {clas['intencion']}: {mensaje[:40]}...")
    
    # Post-cotización: manejar "nuevo", "salir", "pdf"
    if estado.esperando_respuesta == "post_cotizacion":
        return resp_post_cotizacion(estado, clas, mensaje)
    
    # Manejar recordar primero
    if clas["intencion"] == "recordar":
        return resp_recordar(usuario_id, estado)
    
    # NUNCA usar LLM para estas intenciones - siempre determinista
    if clas["intencion"] == "seleccionar":
        return resp_seleccionar(estado, clas, mensaje)
    
    if clas["intencion"] == "seleccionar_polvo":
        return resp_seleccionar_polvo(estado, clas)
    
    if clas["intencion"] == "buscar_polvo":
        return resp_buscar_polvo(estado, clas)
    
    if clas["intencion"] == "cotizar_polvo":
        return resp_cotizar_polvo(estado, clas)
    
    if clas["intencion"] == "cotizar" and estado.producto_seleccionado:
        return resp_cotizar(estado, mensaje)
    
    if clas["intencion"] == "agregar":
        return resp_agregar(estado, clas)
    
    if clas["intencion"] == "despedida":
        return resp_despedida(usuario_id, estado)
    
    # NUEVOS: Grifería
    if clas["intencion"] == "buscar_griferia":
        respuesta = resp_buscar_griferia(estado, mensaje, clas.get("tipo"))
        if respuesta:
            return respuesta
    
    if clas["intencion"] == "seleccionar_griferia":
        return resp_seleccionar_griferia(estado, clas, mensaje)
    
    if clas["intencion"] == "cotizar_griferia" and estado.esperando_respuesta == "cantidad_griferia":
        return resp_cotizar_griferia(estado, mensaje)

    # SOLO para búsqueda inicial usar LLM si está activo
    if clas["intencion"] == "buscar" and vama_llm.usar_llm:
        vama_llm.set_estado(estado)
        respuesta_llm = vama_llm.procesar(mensaje, clas.get("categoria"))
        
        if respuesta_llm:
            return respuesta_llm
    
    # Fallback determinista para todo lo demás
    if clas["intencion"] == "buscar":
        return resp_buscar(estado, clas)
    
    # Si llegó aquí, asumir búsqueda
    return resp_buscar(estado, clas)

# ============================================================================
# MODO CONSOLA (PARA PROBAR)
# ============================================================================

def modo_consola():
    print("="*60)
    print("🏪 VAMA 3.0 - LLM + RAG + Memoria Persistente")
    print("="*60)
    print("Usuarios: 1=Papá, 2=Mamá, 3=Hermano, 4=Tú")
    print("Comandos: 'C'=Cambiar, 'S'=Salir, 'reset'=Limpiar todo")
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
        elif mensaje.lower() == "reset":
            gestor_sesiones.sesiones.clear()
            print("🧹 Sesiones limpiadas")
            continue
        
        print("🤖 Procesando...")
        respuesta = procesar_mensaje(id_actual, nombre_actual, mensaje)
        print(f"\n🤖 VAMA:\n{respuesta}\n")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        # Modo API para N8N
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
        
        print("🚀 API lista en puerto 5000")
        app.run(host='0.0.0.0', port=5000)
    else:
        # Modo consola para probar
        modo = input("¿Usar LLM? (s/n) [n]: ").strip().lower()
        vama_llm.usar_llm = (modo == "s")
        modo_consola()