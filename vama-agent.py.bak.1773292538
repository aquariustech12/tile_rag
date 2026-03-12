#!/usr/bin/env python3
"""
VAMA 2.3 - VERSIÓN FINAL ESTABLE (corregido)
Correcciones:
- Permite agregar productos sin precio (precio: None -> "precio por confirmar")
- Manejo robusto de respuestas afirmativas cortas usando 'ultima_pregunta' y 'ultimo_producto_mostrado'
- Flujo de pegamento integrado sin reiniciar diálogo
"""
import sys
import subprocess
import os
import re
import math
import pickle
import json
import time
from datetime import datetime
from flask import Flask, request, jsonify

# SQLite
try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pysqlite3-binary"])
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.utils import embedding_functions
import ollama

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
CHROMA_PATH = "chroma_db_v3"
MEMORIA_PATH = "memoria_vama.pkl"
MODELO = "qwen2.5:3b"
CACHE_SIMPLES = {}
CACHE_TTL = 3600

print("🔧 Forzando embeddings en CPU...")
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-m3", device="cpu"
)
client = chromadb.PersistentClient(path=CHROMA_PATH)

# ============================================================================
# COLECCIONES
# ============================================================================
print("📚 Cargando colecciones...")
cols = {
    "nacionales": client.get_collection("nacionales", embedding_function=embedding_func),
    "importados": client.get_collection("importados", embedding_function=embedding_func),
    "griferia": client.get_or_create_collection("griferia", embedding_function=embedding_func),
    "lavabos": client.get_or_create_collection("lavabos", embedding_function=embedding_func),
    "sanitarios": client.get_or_create_collection("sanitarios", embedding_function=embedding_func),
    "muebles": client.get_or_create_collection("muebles", embedding_function=embedding_func),
    "tinacos": client.get_or_create_collection("tinacos", embedding_function=embedding_func),
    "espejos": client.get_or_create_collection("espejos", embedding_function=embedding_func),
    "tarjas": client.get_or_create_collection("tarjas", embedding_function=embedding_func),
    "herramientas": client.get_or_create_collection("herramientas", embedding_function=embedding_func),
    "polvos": client.get_or_create_collection("polvos", embedding_function=embedding_func),
    "otras": client.get_or_create_collection("otras", embedding_function=embedding_func)
}
print(f"✅ {len(cols)} colecciones cargadas")

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
                with open(self.archivo, 'rb') as f: return pickle.load(f)
            except: return {}
        return {}
    
    def guardar(self):
        with open(self.archivo, 'wb') as f:
            pickle.dump(self.datos, f)
    
    def obtener(self, usuario_id):
        if usuario_id not in self.datos:
            self.datos[usuario_id] = {
                "nombre": "", 
                "m2_proyecto": 0, 
                "cotizaciones": [], 
                "ultima_visita": None, 
                "interacciones": 0,
                "ultima_respuesta": "",
                "timestamp_ultima": 0,
                "ultimo_mensaje_procesado": "",
                "productos_cotizados": {},
                "ultimos_mensajes": [],
                # flags/estado de diálogo
                "ultima_pregunta": None,            # e.g., "confirmar_cantidad", "elegir_pegamento"
                "ultimo_producto_mostrado": None   # guardamos dict con campos clave del producto
            }
        else:
            # asegurar claves
            if "productos_cotizados" not in self.datos[usuario_id]:
                self.datos[usuario_id]["productos_cotizados"] = {}
            if "ultimo_mensaje_procesado" not in self.datos[usuario_id]:
                self.datos[usuario_id]["ultimo_mensaje_procesado"] = ""
            if "ultimos_mensajes" not in self.datos[usuario_id]:
                self.datos[usuario_id]["ultimos_mensajes"] = []
            if "ultima_pregunta" not in self.datos[usuario_id]:
                self.datos[usuario_id]["ultima_pregunta"] = None
            if "ultimo_producto_mostrado" not in self.datos[usuario_id]:
                self.datos[usuario_id]["ultimo_producto_mostrado"] = None
        
        return self.datos[usuario_id]

memoria_largo_plazo = MemoriaPersistente()

# ============================================================================
# SESIONES
# ============================================================================
class EstadoConversacion:
    def __init__(self, usuario_id, nombre, m2_previos=0):
        self.usuario_id = usuario_id
        self.nombre = nombre
        self.ultimos_productos = []
        self.m2_proyecto = m2_previos
        self.ultimo_mensaje = datetime.now()
        self.ultima_respuesta_enviada = ""

    def actualizar(self): self.ultimo_mensaje = datetime.now()
    def guardar_productos(self, productos): self.ultimos_productos = productos

class GestorSesiones:
    def __init__(self): self.sesiones = {}
    
    def obtener(self, usuario_id, nombre_webhook):
        hist = memoria_largo_plazo.obtener(usuario_id)
        if usuario_id not in self.sesiones:
            nombre_real = hist["nombre"] or nombre_webhook
            self.sesiones[usuario_id] = EstadoConversacion(usuario_id, nombre_real, hist.get("m2_proyecto", 0))
        self.sesiones[usuario_id].actualizar()
        return self.sesiones[usuario_id]

gestor_sesiones = GestorSesiones()

# ============================================================================
# BÚSQUEDA
# ============================================================================
class DB:
    def buscar(self, query, colecciones=None, top_k=5):
        if colecciones is None:
            colecciones = list(cols.keys())
        resultados = []
        for nombre in colecciones:
            if nombre not in cols: continue
            try:
                r = cols[nombre].query(query_texts=[query], n_results=top_k)
                for meta in r["metadatas"][0]:
                    meta['coleccion'] = nombre
                    resultados.append(meta)
            except Exception as e:
                print(f"Error buscando en {nombre}: {e}")
                continue
        return resultados

db = DB()

# ============================================================================
# FORMATEADOR PARA PROMPT (VERSIÓN RUSS)
# ============================================================================
def formatear_productos_russ(productos):
    if not productos: return "Ninguno"
    texto = ""
    for i, p in enumerate(productos[:3], 1):
        codigo = p.get('codigo', 'N/A')
        desc = p.get('descripcion', 'Producto')
        formato = p.get('formato', 'N/A')
        precio = "Por confirmar"
        for k, v in p.items():
            if any(x in k.upper() for x in ["PRECIO", "OFERTA", "FINAL", "SISTEMA", "UNITARIO"]):
                if v and str(v).strip() not in ["0", "0.0", "", "None"]:
                    precio = f"${v}"
                    break
        texto += f"{i}. Código: {codigo} | {desc} | Formato: {formato} | Precio: {precio}\n"
    return texto.strip()

# ============================================================================
# UTILIDADES
# ============================================================================
def extraer_precio_de_meta(prod):
    precio = None
    for k, v in prod.items():
        if any(x in k.upper() for x in ["PRECIO", "OFERTA", "FINAL", "UNITARIO"]):
            try:
                precio = float(v)
                return precio
            except:
                continue
    # fallback
    for key in ['precio_unitario', 'precio_m2', 'precio_caja']:
        if prod.get(key):
            try:
                return float(prod.get(key))
            except:
                continue
    return None

def descripcion_corta(prod):
    return prod.get('descripcion') or prod.get('codigo') or "Producto"

# ============================================================================
# CEREBRO - FINAL CON PROMPT RUSS (con manejo de afirmativos y flags)
# ============================================================================
def generar_respuesta_llm(mensaje, estado, usuario_id):
    msg_limpio = mensaje.strip() if mensaje else ""
    msg_low = msg_limpio.lower()
    hist = memoria_largo_plazo.obtener(usuario_id)
    
    num_interaccion = hist.get("interacciones", 0)
    cache_key = f"{usuario_id}:{msg_limpio}"
    tiempo_actual = time.time()

    # ===== Manejo de respuestas afirmativas cortas y confirmaciones =====
    afirmativos = {"si", "sí", "claro", "ok", "vale", "correcto", "sí.", "si.", "si,", "sí,"}
    # Si el usuario responde afirmativamente y hay una ultima pregunta relevante, procesarla
    if msg_low in afirmativos and hist.get("ultima_pregunta"):
        ultima = hist.get("ultima_pregunta")
        # Confirmar cantidad para ultimo producto mostrado
        if ultima == "confirmar_cantidad" and hist.get("ultimo_producto_mostrado"):
            prod = hist["ultimo_producto_mostrado"]
            desc = descripcion_corta(prod)
            precio = extraer_precio_de_meta(prod)
            # calcular cantidad según metraje por caja si existe y si hay m2 en estado
            cantidad = 1
            try:
                if estado.m2_proyecto and prod.get('metraje_caja'):
                    metraje_caja = float(prod.get('metraje_caja') or 0)
                    if metraje_caja > 0:
                        cantidad = math.ceil(estado.m2_proyecto / metraje_caja)
            except:
                cantidad = 1
            # guardar en cotización (precio puede ser None)
            if desc in hist["productos_cotizados"]:
                hist["productos_cotizados"][desc]["cantidad"] += cantidad
            else:
                hist["productos_cotizados"][desc] = {"precio": precio if precio not in [0, "0", "0.0", ""] else None, "cantidad": cantidad}
            # limpiar flag de pregunta
            hist["ultima_pregunta"] = None
            memoria_largo_plazo.guardar()
            if hist["productos_cotizados"][desc]["precio"] is None:
                return f"✅ Perfecto, agregué {cantidad} de {desc} a tu cotización (precio por confirmar). ¿Quieres que agregue pegamento o te paso el resumen parcial? Equipo VAMA."
            else:
                subtotal = hist["productos_cotizados"][desc]["precio"] * hist["productos_cotizados"][desc]["cantidad"]
                return f"✅ Perfecto, agregué {cantidad} de {desc} a tu cotización. Subtotal: ${subtotal:.2f}. ¿Quieres algo más o te paso el total? Equipo VAMA."
        # Confirmación para pegamento u otra pregunta simple
        if ultima == "confirmar_pegamento" and hist.get("ultimo_producto_mostrado"):
            # si la pregunta era "¿Quieres pegamento?" y responde si -> sugerir opciones
            hist["ultima_pregunta"] = "elegir_pegamento"
            memoria_largo_plazo.guardar()
            return "Perfecto. Te muestro opciones de pegamento: CMX01 - PEGAZULEJO BLANCO 20KG ($169), CMX134 - CEMIX PEGAZULEJO GRIS 20KG ($143). ¿Cuál prefieres? Equipo VAMA."
        # Si no hay manejo específico, continuar (no bloquear)
    
    # ========================================================================
    # CACHÉ
    # ========================================================================
    if cache_key in CACHE_SIMPLES:
        timestamp, respuesta_cache = CACHE_SIMPLES[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            print(f"DEBUG: Respuesta desde caché")
            return respuesta_cache
    
    # ========================================================================
    # ANTI-DUPLICADO (solo después de manejar afirmativos)
    # ========================================================================
    if len(msg_limpio) < 3:
        print(f"DEBUG: Mensaje muy corto")
        if estado.ultima_respuesta_enviada:
            return estado.ultima_respuesta_enviada
        return "Estoy aquí para ayudarte. ¿Qué producto necesitas? Equipo VAMA."
    
    tiempo_ultimo = hist.get("timestamp_ultima", 0)
    if tiempo_actual - tiempo_ultimo < 5 and msg_limpio == hist.get("ultimo_mensaje_procesado", ""):
        print(f"DEBUG: Duplicado detectado")
        return estado.ultima_respuesta_enviada or "Procesando..."
    
    hist["ultimo_mensaje_procesado"] = msg_limpio
    hist["timestamp_ultima"] = tiempo_actual

    # ========================================================================
    # DETECCIÓN DE REPETICIÓN
    # ========================================================================
    if "ultimos_mensajes" in hist and len(hist["ultimos_mensajes"]) >= 4:
        ultimos_3 = hist["ultimos_mensajes"][-3:]
        mensajes_usuario = [m["content"] for m in ultimos_3 if m["role"] == "user"]
        if len(mensajes_usuario) >= 2 and mensajes_usuario[-1] == mensajes_usuario[-2]:
            print("DEBUG: Cliente repitiendo mensaje, forzando respuesta directa")
            if estado.ultimos_productos:
                total = 0.0
                respuesta = "**Resumen de tu cotización:**\n"
                for prod in estado.ultimos_productos[:3]:
                    desc = descripcion_corta(prod)
                    precio = extraer_precio_de_meta(prod) or 0
                    if estado.m2_proyecto > 0 and prod.get('metraje_caja', 0) > 0:
                        cajas = math.ceil(estado.m2_proyecto / prod['metraje_caja'])
                        subtotal = cajas * precio
                        total += subtotal
                        respuesta += f"- {desc}: {cajas} cajas x ${precio:.2f} = ${subtotal:.2f}\n"
                    else:
                        respuesta += f"- {desc}: ${precio:.2f} (precio unitario)\n"
                respuesta += f"\n**TOTAL: ${total:.2f}**\n\n¿Te confirmo disponibilidad? Equipo VAMA."
                hist["ultimos_mensajes"].append({"role": "assistant", "content": respuesta})
                memoria_largo_plazo.guardar()
                CACHE_SIMPLES[cache_key] = (tiempo_actual, respuesta)
                return respuesta
    
    # ========================================================================
    # COMANDOS ESPECIALES
    # ========================================================================
    if any(w in msg_low for w in ['nueva cotizacion', 'nueva cotización', 'empezar de nuevo', 'reiniciar', 'borrar todo']):
        hist["productos_cotizados"] = {}
        hist["m2_proyecto"] = 0
        hist["interacciones"] = 0
        estado.m2_proyecto = 0
        hist["ultima_pregunta"] = None
        hist["ultimo_producto_mostrado"] = None
        memoria_largo_plazo.guardar()
        respuesta = "Listo, hemos reiniciado tu cotización. ¿Qué necesitas cotizar hoy? Equipo VAMA."
        CACHE_SIMPLES[cache_key] = (tiempo_actual, respuesta)
        return respuesta

    if "interacciones" not in hist: hist["interacciones"] = 0
    if "m2_proyecto" not in hist: hist["m2_proyecto"] = estado.m2_proyecto
    
    # ========================================================================
    # DETECCIÓN DE TOTAL
    # ========================================================================
    es_pedido_total = any(p in msg_low for p in ["total", "cotización", "cuánto es", "suma", "precio final"])
    if es_pedido_total and hist["productos_cotizados"]:
        total = 0.0
        detalle = "**Resumen de tu cotización:**\n"
        for desc, data in hist["productos_cotizados"].items():
            if data["precio"] is None:
                detalle += f"- {desc}: {data['cantidad']} (precio por confirmar)\n"
            else:
                subtotal = data["precio"] * data["cantidad"]
                total += subtotal
                detalle += f"- {desc}: {data['cantidad']} x ${data['precio']} = ${subtotal:.2f}\n"
        if total > 0:
            detalle += f"\n**TOTAL: ${total:.2f}**\n\n¿Te confirmo disponibilidad o agregamos algo más? Equipo VAMA."
        else:
            detalle += "\n**TOTAL: por confirmar (faltan precios)**\n\n¿Quieres que te avise en cuanto tenga los precios? Equipo VAMA."
        CACHE_SIMPLES[cache_key] = (tiempo_actual, detalle)
        return detalle

    # ========================================================================
    # CAPTURAR M2
    # ========================================================================
    match_m2 = re.findall(r'(\d+)\s*(?:m2|metros|mts|cuadrados)', msg_low)
    if match_m2:
        estado.m2_proyecto = int(match_m2[0])
        hist["m2_proyecto"] = estado.m2_proyecto
    
    nombre_cliente = hist["nombre"] or "Cliente"

    # ========================================================================
    # DETECCIÓN DE OFERTAS Y SALUDOS
    # ========================================================================
    es_oferta = any(w in msg_low for w in ['oferta', 'promo', 'descuento', 'barato', 'ofertas', 'promociones'])
    saludos = ['hola', 'buenos dias', 'buenas tardes', 'buenas noches', 'que tal', 'hey', 'saludos', 'buen dia']
    es_saludo = any(s in msg_low for s in saludos) and len(msg_low) < 50

    # ========================================================================
    # CATEGORÍAS Y BÚSQUEDA
    # ========================================================================
    if not es_saludo:
        es_griferia = any(w in msg_low for w in ['monomando', 'grifo', 'mezcladora', 'llave', 'regadera'])
        es_lavabo = any(w in msg_low for w in ['lavabo', 'lavamanos'])
        es_sanitario = any(w in msg_low for w in ['wc', 'inodoro', 'taza', 'sanitario', 'tanque', 'pedestal'])
        es_mueble = any(w in msg_low for w in ['mueble', 'gabinete', 'armario'])
        es_tinaco = any(w in msg_low for w in ['tinaco', 'cisterna', 'tanque de agua'])
        es_espejo = any(w in msg_low for w in ['espejo'])
        es_tarja = any(w in msg_low for w in ['tarja', 'fregadero', 'lavaplatos'])
        es_herramienta = any(w in msg_low for w in ['herramienta', 'redtools'])
        es_pegamento = any(w in msg_low for w in ['pegazulejo', 'pegamento', 'pega', 'adhesivo', 'polvo', 'boquilla', 'pastina', 'lechada', 'junta', 'cemento', 'mortero'])
        es_piso = any(w in msg_low for w in ['piso', 'azulejo', 'malla', 'porcelanato', 'ceramica', 'marmol', 'loseta', 'gres'])
        
        colecciones_buscar = []
        if es_oferta:
            colecciones_buscar = ["nacionales", "importados"]
        else:
            if es_pegamento:
                colecciones_buscar.extend(["polvos", "otras"])
            else:
                if es_griferia: colecciones_buscar.append("griferia")
                if es_lavabo: colecciones_buscar.append("lavabos")
                if es_sanitario: colecciones_buscar.append("sanitarios")
                if es_mueble: colecciones_buscar.append("muebles")
                if es_tinaco: colecciones_buscar.append("tinacos")
                if es_espejo: colecciones_buscar.append("espejos")
                if es_tarja: colecciones_buscar.append("tarjas")
                if es_herramienta: colecciones_buscar.append("herramientas")
                if es_piso: colecciones_buscar.extend(["nacionales", "importados"])
        
        if not colecciones_buscar:
            colecciones_buscar = list(cols.keys())
        
        colecciones_buscar = list(set(colecciones_buscar))
        
        resultados = []
        for coleccion in colecciones_buscar:
            if coleccion in cols:
                try:
                    k = 3 if coleccion in ["nacionales", "importados"] else 5
                    r = cols[coleccion].query(query_texts=[msg_limpio], n_results=k)
                    for meta in r["metadatas"][0]:
                        meta['coleccion'] = coleccion
                        resultados.append(meta)
                except Exception as e:
                    continue
        
        query_words = [w for w in re.sub(r'[^\w\s]', '', msg_low).split() if len(w) > 3]
        if resultados:
            resultados.sort(key=lambda x: sum(w in x.get('descripcion','').lower() for w in query_words), reverse=True)
        
        estado.guardar_productos(resultados[:6])
        print(f"DEBUG: {len(resultados)} resultados")
        
        # ========================================================================
        # GUARDAR COMPRA (CORREGIDO: permite agregar sin precio y usa flags)
        # ========================================================================
        es_compra = any(w in msg_low for w in [
            'me llevo', 'quiero', 'agrega', 'compro', 'añade', 
            'cotiza', 'incluye', 'pon', 'agregar', 'comprame',
            'deme', 'dame', 'selecciona', 'escojo', 'elijo',
            'prefiero', 'me quedo', 'llevo'
        ])
        
        # Si el usuario pide pegamento explícitamente y hay resultados de polvos, mostrar opciones
        if es_pegamento and resultados:
            # mostrar opciones de pegamento (hasta 5)
            opciones = []
            for p in resultados[:5]:
                codigo = p.get('codigo', 'N/A')
                desc = descripcion_corta(p)
                precio = extraer_precio_de_meta(p)
                precio_txt = f"${precio}" if precio else "Precio: por confirmar"
                opciones.append(f"{codigo} - {desc} - {precio_txt}")
            # marcar que la ultima pregunta es elegir pegamento
            hist["ultima_pregunta"] = "elegir_pegamento"
            memoria_largo_plazo.guardar()
            respuesta = "Precio de pegamento: " + ", ".join(opciones) + ". ¿Cuántos necesitas? Equipo VAMA."
            CACHE_SIMPLES[cache_key] = (tiempo_actual, respuesta)
            return respuesta
        
        if es_compra and resultados:
            print("DEBUG: Detectada compra")
            match_precio = re.search(r'\$?(\d+(?:\.\d+)?)', msg_low)
            precio_mencionado = float(match_precio.group(1)) if match_precio else None
            producto_agregado = None
            # Intentar identificar producto por coincidencia en descripción o código
            for prod in resultados[:8]:
                desc = descripcion_corta(prod)
                codigo = prod.get('codigo','').lower()
                # si el mensaje menciona el código o parte de la descripción, priorizar
                if codigo and codigo in msg_low:
                    seleccionado = prod
                elif any(word in desc.lower() for word in re.sub(r'[^\w\s]', '', msg_low).split()):
                    seleccionado = prod
                else:
                    seleccionado = None
                
                precio = extraer_precio_de_meta(prod)
                if precio_mencionado and precio and abs(precio - precio_mencionado) < 1:
                    seleccionado = prod
                
                if seleccionado:
                    desc_sel = descripcion_corta(seleccionado)
                    precio_sel = extraer_precio_de_meta(seleccionado)
                    # calcular cantidad: si usuario dio m2 y hay metraje por caja
                    cantidad = 1
                    try:
                        if estado.m2_proyecto and seleccionado.get('metraje_caja'):
                            metraje_caja = float(seleccionado.get('metraje_caja') or 0)
                            if metraje_caja > 0:
                                cantidad = math.ceil(estado.m2_proyecto / metraje_caja)
                    except:
                        cantidad = 1
                    if desc_sel in hist["productos_cotizados"]:
                        hist["productos_cotizados"][desc_sel]["cantidad"] += cantidad
                    else:
                        hist["productos_cotizados"][desc_sel] = {"precio": precio_sel if precio_sel not in [0, "0", "0.0", ""] else None, "cantidad": cantidad}
                    producto_agregado = desc_sel
                    # limpiar flags
                    hist["ultima_pregunta"] = None
                    hist["ultimo_producto_mostrado"] = None
                    memoria_largo_plazo.guardar()
                    break
            
            # Si no se identificó por texto, tomar el primer resultado mostrado
            if not producto_agregado and resultados:
                prod = resultados[0]
                desc = descripcion_corta(prod)
                precio = extraer_precio_de_meta(prod)
                cantidad = 1
                try:
                    if estado.m2_proyecto and prod.get('metraje_caja'):
                        metraje_caja = float(prod.get('metraje_caja') or 0)
                        if metraje_caja > 0:
                            cantidad = math.ceil(estado.m2_proyecto / metraje_caja)
                except:
                    cantidad = 1
                if desc in hist["productos_cotizados"]:
                    hist["productos_cotizados"][desc]["cantidad"] += cantidad
                else:
                    hist["productos_cotizados"][desc] = {"precio": precio if precio not in [0, "0", "0.0", ""] else None, "cantidad": cantidad}
                producto_agregado = desc
                hist["ultima_pregunta"] = None
                hist["ultimo_producto_mostrado"] = None
                memoria_largo_plazo.guardar()
            
            if producto_agregado:
                if hist["productos_cotizados"][producto_agregado]["precio"] is None:
                    respuesta = f"✅ Agregado: {producto_agregado} a tu cotización (precio por confirmar). ¿Quieres algo más o te paso el total parcial? Equipo VAMA."
                else:
                    respuesta = f"✅ Agregado: {producto_agregado} a tu cotización. ¿Quieres algo más o te paso el total? Equipo VAMA."
                CACHE_SIMPLES[cache_key] = (tiempo_actual, respuesta)
                return respuesta

    # ========================================================================
    # Si llegamos aquí, vamos a preparar prompt para LLM (si aplica)
    # Guardar ultimo producto mostrado para confirmar después si el LLM muestra productos
    productos_russ = formatear_productos_russ(estado.ultimos_productos[:3])
    # Si hay productos en estado.ultimos_productos, guardamos el primero como 'ultimo_producto_mostrado'
    if estado.ultimos_productos:
        # guardamos una copia ligera (solo campos necesarios)
        p0 = estado.ultimos_productos[0]
        hist["ultimo_producto_mostrado"] = {
            "codigo": p0.get("codigo"),
            "descripcion": p0.get("descripcion"),
            "formato": p0.get("formato"),
            "metraje_caja": p0.get("metraje_caja"),
            "precio_unitario": p0.get("precio_unitario"),
            "precio_m2": p0.get("precio_m2"),
            "precio_caja": p0.get("precio_caja"),
            # incluir cualquier campo de precio bruto que venga en metadatas
            **{k: v for k, v in p0.items() if "precio" in k.lower() or "oferta" in k.lower()}
        }
        # marcar que la ultima pregunta puede ser confirmar cantidad si el bot pregunta "¿Necesitas X m2?"
        # No forzamos aquí; se seteará cuando el LLM pregunte explícitamente.
    memoria_largo_plazo.guardar()

    # Obtener historial reciente
    historial_reciente = ""
    if "ultimos_mensajes" in hist and hist["ultimos_mensajes"]:
        ultimos = hist["ultimos_mensajes"][-6:]
        for m in ultimos:
            role = "Cliente" if m["role"] == "user" else "Russ"
            historial_reciente += f"{role}: {m['content']}\n"
    else:
        historial_reciente = "Sin historial"

    # ========================================================================
    # PROMPT - VERSIÓN FINAL (igual que antes)
    # ========================================================================
    prompt = f"""Eres VAMA, asistente de ventas por WhatsApp de VAMA, una tienda de materiales y acabados (pisos, cerámicos, recubrimientos, muros, baños/grifería).

Tu tarea es orientar, recomendar y cerrar una cotización rápida, sin inventar datos.

CONTEXTO ACTUAL:
- Cliente: {nombre_cliente}
- Historial de la conversación: {historial_reciente}
- Productos encontrados: {productos_russ if estado.ultimos_productos else 'Ninguno'}
- Metraje del cliente: {estado.m2_proyecto} m2

REGLAS DE ESTILO:
- Responde en español, texto plano listo para WhatsApp (sin markdown).
- Mensajes cortos: ideal 2 a 6 líneas. Si listas opciones, hasta 10 líneas.
- Tono humano, profesional y cercano.
- No uses tecnicismos innecesarios.

RESTRICCIONES (OBLIGATORIAS):
- No inventes precios, existencias, tiempos, descuentos, marcas, formatos ni metrajes.
- Solo usa información de los productos listados arriba.
- Si falta un dato (ej. precio), dilo exactamente así: "Precio: por confirmar".

USO DE CONTEXTO:
- Usa el historial para no repetir preguntas.
- Si el cliente ya dio formato/uso/área/cantidad, NO lo vuelvas a preguntar.
- Si el nombre existe, úsalo naturalmente máximo 1 vez por mensaje.

Tu respuesta debe ser SOLO el mensaje final para WhatsApp."""

    hist["interacciones"] = num_interaccion + 1
    hist["ultima_visita"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    memoria_largo_plazo.guardar()

    try:
        res = ollama.generate(
            model=MODELO, 
            prompt=prompt,
            options={
                "temperature": 0.6, 
                "num_predict": 300, 
                "num_ctx": 2048,
                "stop": ["CLIENTE:", "USUARIO:", "\n\n\n"]
            }
        )['response'].strip()
        
        # Limpieza básica
        res = re.sub(r'\*\*?|\*|__|\|', '', res)
        res = res.replace("  ", " ").strip()
        
        # Guardar en historial
        if "ultimos_mensajes" not in hist:
            hist["ultimos_mensajes"] = []
        hist["ultimos_mensajes"].append({"role": "user", "content": msg_limpio})
        hist["ultimos_mensajes"].append({"role": "assistant", "content": res})
        if len(hist["ultimos_mensajes"]) > 10:
            hist["ultimos_mensajes"] = hist["ultimos_mensajes"][-10:]
        
        # Si el LLM pregunta "¿Necesitas X m2?" o "¿Necesitas 7 m2?" detectarlo y setear ultima_pregunta
        if re.search(r'¿.*necesit.*\s*\d+\s*(m2|metros|mts|cuadrados)\?', res.lower()):
            hist["ultima_pregunta"] = "confirmar_cantidad"
        # Si el LLM pregunta por pegamento explícitamente
        if re.search(r'pegamento|pegazulejo|adhesivo', res.lower()) and "¿" in res:
            hist["ultima_pregunta"] = "confirmar_pegamento"
        memoria_largo_plazo.guardar()

        estado.ultima_respuesta_enviada = res
        hist["ultima_respuesta"] = res
        memoria_largo_plazo.guardar()
        CACHE_SIMPLES[cache_key] = (tiempo_actual, res)
        
        return res
        
    except Exception as e:
        print(f"DEBUG: Error: {e}")
        fallback = f"Entendido {nombre_cliente}, revisando disponibilidad. Equipo VAMA."
        estado.ultima_respuesta_enviada = fallback
        return fallback

# ============================================================================
# API
# ============================================================================
app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json() or {}
    tel = str(data.get('telefono') or '0')
    nom = data.get('nombre') or 'Cliente'
    msg = data.get('mensaje') or ""
    
    print(f"\n[WEBHOOK] {datetime.now().strftime('%H:%M:%S')} | {nom} ({tel}): {msg[:60]}")
    
    sesion = gestor_sesiones.obtener(tel, nom)
    respuesta = generar_respuesta_llm(msg, sesion, tel)
    
    print(f"[RESPONSE] {respuesta[:80]}...\n")
    return jsonify({"respuesta": respuesta})

if __name__ == "__main__":
    print(f"🚀 VAMA AGENT 2.3 RUSS | Modelo: {MODELO} | Puerto: 5000")
    print(f"📚 Colecciones: {list(cols.keys())}")
    print(f"🧠 Memoria de cliente: ACTIVADA")
    print(f"🤖 Prompt estilo Russ: ACTIVADO")
    app.run(host='0.0.0.0', port=5000, debug=False)
