#!/usr/bin/env python3
"""
VAMA PRO - Asistente Inteligente con LLM + RAG
VERSIÓN 2.0 - Con memoria multi-usuario PERSISTENTE
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import chromadb
from chromadb.utils import embedding_functions
import ollama
import re
import json
import math
import pickle  # NUEVO: para guardar en disco
from typing import Dict, List, Optional
from datetime import datetime, timedelta

CHROMA_PATH = "chroma_db_v3"
MEMORIA_PATH = "memoria_vama.pkl"  # NUEVO: archivo de memoria persistente

# ============================================================================
# CONEXIÓN A DB
# ============================================================================

print("🔌 Conectando...")
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
    print("✅ Ollama disponible (opcional)")
except:
    print("⚠️ Ollama no responde, modo sin LLM activado")

print(f"✅ {sum(c.count() for c in cols.values())} productos\n")

# ============================================================================
# MEMORIA PERSISTENTE (NUEVO)
# ============================================================================

class MemoriaPersistente:
    """Guarda historial de cotizaciones en disco"""
    
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
    
    def obtener_historial(self, usuario_id):
        return self.datos.get(usuario_id, {
            "cotizaciones": [],
            "productos_vistos": [],
            "ultima_visita": None,
            "total_acumulado": 0
        })
    
    def agregar_cotizacion(self, usuario_id, items, total):
        if usuario_id not in self.datos:
            self.datos[usuario_id] = self.obtener_historial(usuario_id)
        
        self.datos[usuario_id]["cotizaciones"].append({
            "fecha": datetime.now().isoformat(),
            "items": items,
            "total": total
        })
        self.datos[usuario_id]["ultima_visita"] = datetime.now().isoformat()
        self.datos[usuario_id]["total_acumulado"] += total
        
        # Guardar productos vistos
        for item in items:
            desc = item["producto"]["descripcion"]
            if desc not in self.datos[usuario_id]["productos_vistos"]:
                self.datos[usuario_id]["productos_vistos"].append(desc)
        
        self.guardar()
    
    def dias_desde_ultima_visita(self, usuario_id):
        hist = self.obtener_historial(usuario_id)
        if not hist["ultima_visita"]:
            return None
        ultima = datetime.fromisoformat(hist["ultima_visita"])
        return (datetime.now() - ultima).days

memoria_largo_plazo = MemoriaPersistente()

# ============================================================================
# RENDIMIENTOS POLVOS
# ============================================================================

RENDIMIENTOS_POLVOS = {
    "adhesivo": 4.5,
    "pegazulejo": 5.0,
    "cemento": 6.0,
    "boquilla": 8.0,
    "default": 4.5
}

def detectar_tipo_polvo(descripcion: str) -> str:
    d = descripcion.lower()
    if "adhesivo" in d or "porcelanico" in d:
        return "adhesivo"
    elif "pegazulejo" in d or "pegapiso" in d:
        return "pegazulejo"
    elif "cemento" in d:
        return "cemento"
    elif "boquilla" in d:
        return "boquilla"
    else:
        return "default"

# ============================================================================
# CLASE DB
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
                        "es_promo": meta.get("es_promo", False),
                        "tipo": meta.get("tipo", ""),
                        "coleccion": nombre,
                        "presentacion": meta.get("presentacion", ""),
                        "rendimiento": meta.get("rendimiento", 4.5)
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

# ============================================================================
# ESTADO DE CONVERSACIÓN (CORTO PLAZO - RAM)
# ============================================================================

class EstadoConversacion:
    def __init__(self, usuario_id):
        self.usuario_id = usuario_id
        self.productos_vistos = []
        self.producto_seleccionado = None
        self.m2_proyecto = None
        self.items_cotizacion = []
        self.ultima_categoria = None
        self.ultimos_productos = []
        self.categoria_activa = None
        self.ultimo_mensaje = datetime.now()
        self.historial = []
    
    def actualizar_actividad(self):
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
            "m2": m2,
            "timestamp": datetime.now().isoformat()
        }
        self.items_cotizacion.append(item)
        if m2:
            self.m2_proyecto = m2
    
    def get_total(self):
        total = 0
        for item in self.items_cotizacion:
            total += item["calculo"].get("total", 0)
        return total
    
    def get_resumen(self):
        if not self.items_cotizacion:
            return "No hay items en la cotización."
        
        lineas = ["📋 RESUMEN DE COTIZACIÓN"]
        for i, item in enumerate(self.items_cotizacion[-5:], 1):
            prod = item["producto"]
            calc = item["calculo"]
            if "cajas" in calc:
                lineas.append(f"{i}. {prod['descripcion'][:30]} - {calc['cajas']} cajas = ${calc['total']:.2f}")
            elif "sacos" in calc:
                lineas.append(f"{i}. {prod['descripcion'][:30]} - {calc['sacos']} sacos = ${calc['total']:.2f}")
            else:
                lineas.append(f"{i}. {prod['descripcion'][:30]} - {calc.get('unidades', 1)} un = ${calc['total']:.2f}")
        
        lineas.append(f"\n💰 TOTAL: ${self.get_total():.2f}")
        return "\n".join(lineas)

# ============================================================================
# GESTOR DE SESIONES (INTEGRA CORTO Y LARGO PLAZO)
# ============================================================================

class GestorSesiones:
    def __init__(self):
        self.sesiones = {}  # Corto plazo (RAM)
        self.tiempo_expiracion = timedelta(hours=2)  # Sesión activa 2 horas
    
    def obtener_sesion(self, usuario_id):
        ahora = datetime.now()
        
        # Limpiar sesiones inactivas de corto plazo
        expirados = []
        for uid, sesion in self.sesiones.items():
            if ahora - sesion.ultimo_mensaje > self.tiempo_expiracion:
                expirados.append(uid)
        
        for uid in expirados:
            del self.sesiones[uid]
            print(f"🧹 Sesión corto plazo expirada: {uid}")
        
        # Crear nueva sesión de corto plazo si no existe
        if usuario_id not in self.sesiones:
            self.sesiones[usuario_id] = EstadoConversacion(usuario_id)
            print(f"🆕 Nueva sesión activa: {usuario_id}")
            
            # Verificar si hay historial de largo plazo
            dias = memoria_largo_plazo.dias_desde_ultima_visita(usuario_id)
            if dias is not None:
                if dias == 0:
                    print(f"   📚 Cliente regresó hoy")
                else:
                    print(f"   📚 Última visita: hace {dias} días")
        
        self.sesiones[usuario_id].actualizar_actividad()
        return self.sesiones[usuario_id]
    
    def guardar_cotizacion_largo_plazo(self, usuario_id):
        """Persiste la cotización actual en disco"""
        if usuario_id in self.sesiones:
            sesion = self.sesiones[usuario_id]
            if sesion.items_cotizacion:
                total = sesion.get_total()
                memoria_largo_plazo.agregar_cotizacion(
                    usuario_id, 
                    sesion.items_cotizacion, 
                    total
                )
                print(f"💾 Cotización guardada en memoria persistente: ${total:.2f}")

gestor_sesiones = GestorSesiones()

# ============================================================================
# CALCULADORA
# ============================================================================

class CalculadoraVAMA:
    @staticmethod
    def calcular_piso(producto, m2):
        cajas = math.ceil(m2 / producto["metraje_caja"])
        m2_real = cajas * producto["metraje_caja"]
        precio_caja = producto["precio_m2"] * producto["metraje_caja"]
        total = cajas * precio_caja
        
        return {
            "tipo": "piso",
            "cajas": cajas,
            "m2_real": m2_real,
            "precio_caja": precio_caja,
            "total": total,
            "detalle": f"{cajas} cajas = ${total:.2f}"
        }
    
    @staticmethod
    def calcular_polvo(producto, m2):
        tipo = detectar_tipo_polvo(producto["descripcion"])
        rendimiento = RENDIMIENTOS_POLVOS.get(tipo, RENDIMIENTOS_POLVOS["default"])
        
        sacos = math.ceil(m2 / rendimiento)
        m2_reales = sacos * rendimiento
        total = sacos * producto["precio_unitario"]
        
        return {
            "tipo": "polvo",
            "sacos": sacos,
            "m2_reales": m2_reales,
            "rendimiento": rendimiento,
            "total": total,
            "detalle": f"{sacos} sacos = ${total:.2f} (rinde {rendimiento}m²/saco)"
        }
    
    @staticmethod
    def calcular_griferia(producto, cantidad=1):
        total = cantidad * producto["precio_unitario"]
        
        return {
            "tipo": "griferia",
            "unidades": cantidad,
            "total": total,
            "detalle": f"{cantidad} unidad(es) = ${total:.2f}"
        }
    
    @staticmethod
    def calcular_otro(producto, cantidad=1):
        total = cantidad * producto.get("precio_unitario", 0)
        
        return {
            "tipo": "otro",
            "unidades": cantidad,
            "total": total,
            "detalle": f"{cantidad} unidad(es) = ${total:.2f}"
        }
    
    @classmethod
    def calcular(cls, producto, m2=None, cantidad=1):
        coleccion = producto.get("coleccion", "")
        
        if coleccion in ["nacionales", "importados"] and producto.get("precio_m2", 0) > 0:
            return cls.calcular_piso(producto, m2 if m2 else 10)
        elif coleccion == "polvos":
            return cls.calcular_polvo(producto, m2 if m2 else 10)
        elif coleccion == "griferia":
            return cls.calcular_griferia(producto, cantidad)
        else:
            return cls.calcular_otro(producto, cantidad)

calculadora = CalculadoraVAMA()

# ============================================================================
# CLASIFICADOR DE INTENCIÓN
# ============================================================================

def clasificar_intencion(mensaje: str, estado: EstadoConversacion) -> Dict:
    m = mensaje.lower()
    
    # Detectar recordar/memoria
    if any(x in m for x in ["ayer", "anterior", "otro día", "seguimos", "todavía tengo", "la vez pasada", "recuerdas", "habíamos quedado"]):
        return {"intencion": "recordar", "categoria": None, "es_complemento": False}
    
    if any(x in m for x in ["gracias", "adiós", "hasta luego", "eso es todo", "listo", "terminamos"]):
        return {"intencion": "despedida", "categoria": None, "es_complemento": False}
    
    if any(x in m for x in ["busco", "tienes", "opciones", "muéstrame", "muestrame", "ver", "catalogo", "hay"]):
        return {"intencion": "buscar", "categoria": detectar_categoria(m), "es_complemento": False}
    
    if any(x in m for x in ["cotizar", "precio", "cuanto", "cuesta", "m2", "m²"]) or re.search(r'\d+\s*(?:m2|m²)', m):
        return {"intencion": "cotizar", "categoria": detectar_categoria(m), "es_complemento": False}
    
    if any(x in m for x in ["agrega", "tambien", "ademas", "y tambien", "también", "además", "otro"]) or \
       (estado.items_cotizacion and any(x in m for x in ["pegamento", "griferia", "grifo", "llave", "adhesivo", "boquilla"])):
        return {"intencion": "agregar", "categoria": detectar_categoria(m), "es_complemento": True}
    
    if m.strip().isdigit() and 1 <= int(m) <= 5 and estado.ultimos_productos:
        return {"intencion": "seleccionar", "indice": int(m) - 1, "categoria": None, "es_complemento": False}
    
    return {"intencion": "info", "categoria": detectar_categoria(m), "es_complemento": False}

def detectar_categoria(mensaje: str) -> Optional[str]:
    m = mensaje.lower()
    
    if any(x in m for x in ["piso", "porcelanato", "porcelánico", "loseta"]):
        return "pisos"
    elif any(x in m for x in ["muro", "azulejo", "pared", "revestimiento"]):
        return "muros"
    elif any(x in m for x in ["grifo", "llave", "monomando", "mezcladora", "regadera", "griferia"]):
        return "griferia"
    elif any(x in m for x in ["pegamento", "adhesivo", "cemento", "boquilla", "pega", "polvo"]):
        return "polvos"
    elif any(x in m for x in ["mueble", "espejo", "tinaco", "tarja", "lavabo"]):
        return "otras"
    
    return None

# ============================================================================
# VAMA LLM
# ============================================================================

class VAMALLM:
    def __init__(self, usar_llm=True):
        self.usar_llm = usar_llm
        self.modelo = "qwen3:30b-a3b-fp16"
        self.estado = None
    
    def set_estado(self, estado):
        self.estado = estado
    
    def clasificar_intencion(self, mensaje):
        return clasificar_intencion(mensaje, self.estado)
    
    def obtener_productos_rag(self, clas, mensaje):
        categoria = clas.get('categoria')
        db = DB()
        if categoria == "pisos":
            return db.buscar(mensaje, tipo="piso", top_k=5)
        elif categoria == "muros":
            return db.buscar(mensaje, tipo="muro", top_k=5)
        elif categoria == "polvos":
            return db.buscar(mensaje, colecciones=["polvos"], top_k=5)
        elif categoria == "griferia":
            return db.buscar(mensaje, colecciones=["griferia"], top_k=5)
        else:
            return db.buscar(mensaje, top_k=5)
    
    def fallback_determinista(self, mensaje, clas):
        return None
    
    def generar_prompt(self, mensaje: str, clas: Dict, productos_info: str) -> str:
        contexto = ""
        if self.estado and self.estado.items_cotizacion:
            contexto += f"Cotización actual: {len(self.estado.items_cotizacion)} items, Total: ${self.estado.get_total():.2f}\n"
        
        if self.estado and self.estado.m2_proyecto:
            contexto += f"Proyecto: {self.estado.m2_proyecto}m²\n"
        
        if productos_info:
            seccion_productos = f"""PRODUCTOS REALES ENCONTRADOS:
{productos_info}

INSTRUCCIÓN: SOLO usa estos productos. NO inventes."""
        else:
            seccion_productos = "NO SE ENCONTRARON PRODUCTOS."
        
        prompt = f"""Eres VAMA, experto en materiales de construcción.

CONTEXTO:
{contexto.strip()}

INTENCIÓN: {clas['intencion']}
CATEGORÍA: {clas.get('categoria', 'no especificada')}

{seccion_productos}

CLIENTE: {mensaje}

Responde como VAMA:"""
        
        return prompt
    
    def procesar_mensaje(self, mensaje):
        if not self.estado:
            return None
        
        clas = self.clasificar_intencion(mensaje)
        
        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:m2|m²)', mensaje.lower())
        if match:
            self.estado.m2_proyecto = float(match.group(1))
        
        productos_info = ""
        productos = []
        if clas['intencion'] in ['buscar', 'cotizar', 'agregar']:
            productos = self.obtener_productos_rag(clas, mensaje)
            if productos:
                lines = []
                for i, p in enumerate(productos[:5], 1):
                    if p['precio_m2'] > 0:
                        lines.append(f"{i}. {p['descripcion']} - ${p['precio_m2']:.2f}/m²")
                    else:
                        lines.append(f"{i}. {p['descripcion']} - ${p['precio_unitario']:.2f}/unidad")
                productos_info = "\n".join(lines)
        
        prompt = self.generar_prompt(mensaje, clas, productos_info)
        
        try:
            respuesta = ollama.generate(
                model=self.modelo,
                prompt=prompt,
                options={'temperature': 0.3}
            )['response'].strip()
            
            if productos:
                self.estado.guardar_productos(productos)
            
            return respuesta
        except Exception as e:
            print(f"⚠️ Error LLM: {e}")
            return self.fallback_determinista(mensaje, clas)

vama_llm = VAMALLM(usar_llm=False)

# ============================================================================
# RESPUESTAS DETERMINISTAS
# ============================================================================

def respuesta_recordar(usuario_id, mensaje, clas, estado):
    """Maneja '¿te acuerdas del piso del otro día?'"""
    hist = memoria_largo_plazo.obtener_historial(usuario_id)
    
    if not hist["cotizaciones"]:
        return "No tengo registro de cotizaciones anteriores. ¿En qué puedo ayudarte hoy?"
    
    ultima = hist["cotizaciones"][-1]
    dias = (datetime.now() - datetime.fromisoformat(ultima["fecha"])).days
    
    respuesta = "📚 *TU HISTORIAL CON VAMA*\n\n"
    
    if dias == 0:
        respuesta += "Hoy ya hiciste una cotización con nosotros.\n"
    elif dias == 1:
        respuesta += "Ayer estuviste cotizando.\n"
    else:
        respuesta += f"Tu última visita fue hace {dias} días.\n"
    
    respuesta += f"💰 Total histórico: ${hist['total_acumulado']:.2f}\n\n"
    respuesta += "*ÚLTIMA COTIZACIÓN:*\n"
    
    for i, item in enumerate(ultima["items"][-3:], 1):
        prod = item["producto"]
        calc = item["calculo"]
        respuesta += f"{i}. {prod['descripcion'][:35]}... = ${calc['total']:.2f}\n"
    
    respuesta += f"\n¿Quieres repetir esta cotización o hacer una nueva?"
    return respuesta

def respuesta_buscar(usuario_id, mensaje, clas, estado):
    categoria = clas.get('categoria')
    
    if not categoria:
        return "¿Qué tipo de producto buscas? (pisos, azulejos, grifería, pegamentos)"
    
    db = DB()
    if categoria == "pisos":
        productos = db.buscar(mensaje, tipo="piso", top_k=5)
    elif categoria == "muros":
        productos = db.buscar(mensaje, tipo="muro", top_k=5)
    elif categoria == "polvos":
        productos = db.buscar(mensaje, colecciones=["polvos"], top_k=5)
    elif categoria == "griferia":
        productos = db.buscar(mensaje, colecciones=["griferia"], top_k=5)
    else:
        productos = db.buscar(mensaje, top_k=5)
    
    if not productos:
        return f"No encontré {categoria}. ¿Puedes ser más específico?"
    
    estado.guardar_productos(productos)
    estado.categoria_activa = categoria
    
    respuesta = f"🔍 *{categoria.upper()}* - Encontré {len(productos)} opciones:\n\n"
    
    for i, p in enumerate(productos[:5], 1):
        if p['precio_m2'] > 0:
            precio_caja = p['precio_m2'] * p['metraje_caja']
            respuesta += f"{i}. *{p['descripcion'][:40]}*\n"
            respuesta += f"   🏢 {p['proveedor']} | {p['formato']}\n"
            respuesta += f"   💰 ${p['precio_m2']:.2f}/m² (${precio_caja:.2f}/caja)\n\n"
        else:
            respuesta += f"{i}. *{p['descripcion'][:40]}*\n"
            respuesta += f"   🏢 {p['proveedor']} | 💰 ${p['precio_unitario']:.2f}/unidad\n\n"
    
    respuesta += "¿Cuál te interesa? (di el número 1-5)"
    return respuesta

def respuesta_seleccionar(usuario_id, mensaje, clas, estado):
    indice = clas.get('indice', 0)
    
    producto = estado.seleccionar(indice)
    if not producto:
        return "Primero necesito que busques productos. ¿Qué necesitas?"
    
    if producto['precio_m2'] > 0:
        return f"✅ *Seleccionado:* {producto['descripcion'][:40]}\n\nPrecio: ${producto['precio_m2']:.2f}/m²\nMetraje por caja: {producto['metraje_caja']}m²\n\n¿Para cuántos metros cuadrados?"
    else:
        return f"✅ *Seleccionado:* {producto['descripcion'][:40]}\n\nPrecio unitario: ${producto['precio_unitario']:.2f}\n\n¿Cuántas unidades necesitas?"

def respuesta_cotizar(usuario_id, mensaje, clas, estado):
    m2 = None
    match = re.search(r'(\d+(?:\.\d+)?)\s*(?:m2|m²|metros)', mensaje.lower())
    if match:
        m2 = float(match.group(1))
        estado.m2_proyecto = m2
    
    if estado.producto_seleccionado and m2:
        calculo = calculadora.calcular(estado.producto_seleccionado, m2)
        estado.agregar_item(estado.producto_seleccionado, calculo, m2)
        
        respuesta = f"💰 *COTIZACIÓN*\n\n"
        respuesta += f"*{estado.producto_seleccionado['descripcion'][:40]}*\n"
        respuesta += f"📐 {m2}m² solicitados\n"
        respuesta += f"📦 {calculo['detalle']}\n\n"
        respuesta += f"💰 *TOTAL: ${calculo['total']:.2f}*\n\n"
        respuesta += "¿Quieres agregar algo más? (pegamento, grifería, etc.)"
        
        return respuesta
    
    elif estado.ultimos_productos and m2:
        producto = estado.ultimos_productos[0]
        calculo = calculadora.calcular(producto, m2)
        estado.agregar_item(producto, calculo, m2)
        
        respuesta = f"💰 *COTIZACIÓN PARA {m2}m²*\n\n"
        respuesta += f"*{producto['descripcion'][:40]}*\n"
        respuesta += f"{calculo['detalle']}\n\n"
        respuesta += f"💰 *TOTAL: ${calculo['total']:.2f}*\n\n"
        respuesta += "¿Quieres agregar algo más?"
        
        return respuesta
    
    else:
        return "Para cotizar necesito:\n1. Un producto seleccionado\n2. Los metros cuadrados\n\n¿Puedes darme esos datos?"

def respuesta_agregar(usuario_id, mensaje, clas, estado):
    if not estado.items_cotizacion:
        return "Primero necesitas una cotización. ¿Qué producto necesitas?"
    
    categoria = clas.get('categoria', 'polvos')
    
    if categoria == "polvos":
        return "¿Qué tipo de pegamento necesitas? (adhesivo, pegazulejo, cemento, boquilla)"
    elif categoria == "griferia":
        return "¿Qué tipo de grifería necesitas? (monomando, llave, regadera)"
    else:
        return "¿Qué quieres agregar? (pegamento, grifería, etc.)"

def respuesta_despedida(usuario_id, mensaje, clas, estado):
    if estado.items_cotizacion:
        # GUARDAR EN MEMORIA PERSISTENTE
        gestor_sesiones.guardar_cotizacion_largo_plazo(usuario_id)
        
        total = estado.get_total()
        respuesta = f"🎉 *COTIZACIÓN FINALIZADA*\n\n"
        respuesta += f"💰 **TOTAL: ${total:.2f}**\n\n"
        respuesta += estado.get_resumen()
        respuesta += "\n\n¡Gracias por contactar a VAMA! 👋"
    else:
        respuesta = "¡Gracias por contactar a VAMA! Estamos aquí cuando lo necesites. 👋"
    
    return respuesta

def respuesta_info(usuario_id, mensaje, clas, estado):
    return "🏪 *VAMA - Materiales de Construcción*\n\nPuedo ayudarte con:\n\n" \
           "1) 🏠 Buscar pisos y porcelanatos\n" \
           "2) 🧱 Ver azulejos para muros\n" \
           "3) 🚿 Cotizar grifería\n" \
           "4) 🔧 Agregar pegamentos\n" \
           "5) 💰 Crear cotizaciones\n\n" \
           "¿En qué puedo ayudarte?"

MANEJADORES = {
    "recordar": respuesta_recordar,  # NUEVO
    "buscar": respuesta_buscar,
    "seleccionar": respuesta_seleccionar,
    "cotizar": respuesta_cotizar,
    "agregar": respuesta_agregar,
    "despedida": respuesta_despedida,
    "info": respuesta_buscar
}

# ============================================================================
# FUNCIÓN PRINCIPAL DE PROCESAMIENTO
# ============================================================================

def procesar_mensaje(usuario_id: str, mensaje: str) -> str:
    estado = gestor_sesiones.obtener_sesion(usuario_id)
    estado.historial.append(f"👤 Cliente: {mensaje}")
    
    clas = clasificar_intencion(mensaje, estado)
    print(f"   [Usuario: {usuario_id[:8]}... | Intención: {clas['intencion']} | Cat: {clas.get('categoria')}]")
    
    respuesta_llm = None
    if vama_llm.usar_llm:
        vama_llm.set_estado(estado)
        respuesta_llm = vama_llm.procesar_mensaje(mensaje)
    
    if respuesta_llm:
        estado.historial.append(f"🤖 VAMA: {respuesta_llm[:100]}...")
        return respuesta_llm
    
    manejador = MANEJADORES.get(clas['intencion'], MANEJADORES['info'])
    respuesta = manejador(usuario_id, mensaje, clas, estado)
    
    estado.historial.append(f"🤖 VAMA: {respuesta[:100]}...")
    return respuesta

# ============================================================================
# SIMULADOR MULTI-USUARIO
# ============================================================================

def simulador_multi_usuario():
    print("="*60)
    print("🧠 VAMA 2.0 - MODO MULTI-USUARIO CON MEMORIA PERSISTENTE")
    print("="*60)
    print("Usuarios disponibles:")
    print("  U1: Juan (pisos)")
    print("  U2: María (azulejos)")
    print("  U3: Pedro (grifería)")
    print("  U4: Ana (cotizaciones)")
    print("="*60)
    print("COMANDOS:")
    print("  Escribe tu mensaje directamente")
    print("  'C' = Cambiar de usuario")
    print("  'S' = Salir")
    print("="*60)
    
    usuarios = {
        "1": "juan_whatsapp",
        "2": "maria_whatsapp", 
        "3": "pedro_whatsapp",
        "4": "ana_whatsapp"
    }
    
    nombres = {
        "1": "Juan",
        "2": "María",
        "3": "Pedro",
        "4": "Ana"
    }
    
    # CORREGIDO: Mantener usuario activo
    usuario_actual_id = None
    nombre_actual = None
    
    while True:
        # Solo preguntar usuario si no hay uno activo
        if not usuario_actual_id:
            print("\n" + "-"*60)
            usuario_opt = input("¿Qué usuario habla? (1-4, S=salir): ").strip().upper()
            
            if usuario_opt == "S":
                break
            
            if usuario_opt not in usuarios:
                print("❌ Usuario no válido")
                continue
            
            usuario_actual_id = usuarios[usuario_opt]
            nombre_actual = nombres[usuario_opt]
            print(f"\n👤 {nombre_actual} está ahora activo")
            print("   (Escribe 'C' en cualquier momento para cambiar de usuario)")
        
        # Loop de conversación con el mismo usuario
        print("-"*60)
        mensaje = input(f"👤 {nombre_actual}: ").strip()
        
        if not mensaje:
            continue
        
        # Comandos especiales
        if mensaje.upper() == "S":
            break
        elif mensaje.upper() == "C":
            usuario_actual_id = None
            nombre_actual = None
            print(f"\n🔄 Cambiando de usuario...")
            continue
        
        # Procesar mensaje
        print("🤖 Procesando...")
        respuesta = procesar_mensaje(usuario_actual_id, mensaje)
        
        print(f"\n🤖 VAMA:\n{respuesta}\n")
        
# ============================================================================
# MAIN
# ============================================================================

def main():
    global vama_llm
    print("="*60)
    print("🏪 VAMA 2.0 - Asistente Multi-Usuario con Memoria Persistente")
    print("="*60)
    print("✅ Memoria CORTO PLAZO: Sesión activa (2h)")
    print("✅ Memoria LARGO PLAZO: Historial guardado en disco")
    print("✅ Cálculos especializados (pisos, polvos, grifería)")
    print("✅ Multi-usuario: Juan, María, Pedro, Ana")
    print("="*60 + "\n")
    
    # Preguntar modo
    modo = input("¿Usar LLM? (s/n) [n]: ").strip().lower()
    vama_llm.usar_llm = (modo == "s")
    print(f"📌 Modo: {'LLM activado' if vama_llm.usar_llm else 'Modo determinista'}\n")
    
    # Ir al simulador
    simulador_multi_usuario()

if __name__ == "__main__":
    main()