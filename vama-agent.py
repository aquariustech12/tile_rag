# --- HOTFIX: extracción simple de slots y short-circuit webhook ---
import re
from flask import request, jsonify

def extraer_slots_simple(texto):
    t = (texto or "").lower()
    tipo = None
    if re.search(r'\bpisos?\b', t): tipo = 'pisos'
    elif re.search(r'\bpared(es)?\b', t): tipo = 'paredes'
    elif re.search(r'\bbaño(s)?\b', t): tipo = 'baños'
    color = None
    mcol = re.search(r'\b(blanco|gris|negro|beige|mármol|carrara)\b', t)
    if mcol: color = mcol.group(1)
    area = None
    marea = re.search(r'(\d+(?:[.,]\d+)?)\s*(m2|m²|metros?)', t)
    if marea:
        try:
            area = float(marea.group(1).replace(',', '.'))
        except:
            area = None
    return {'tipo': tipo, 'color': color, 'area_m2': area}

def _vama_shortcircuit():
    try:
        if request.path == '/webhook' and request.method == 'POST':
            data = request.get_json(silent=True) or {}
            msg = data.get('mensaje') or data.get('message') or ''
            slots = extraer_slots_simple(msg)
            if slots.get('tipo') and slots.get('area_m2'):
                opciones = []
                try:
                    # intenta llamar a la función de búsqueda existente si está disponible
                    opciones = buscar_por_atributos(tipo=slots['tipo'], color=slots.get('color'), top_k=3)
                except Exception:
                    opciones = []
                if opciones:
                    resp_lines = []
                    for p in opciones[:3]:
                        resp_lines.append(f"- {p.get('descripcion','sin desc')} (Código: {p.get('codigo','-')})")
                    resp = "Perfecto — encontré estas opciones para {} de {} m²:\n{}\n¿Quieres que agregue alguna a tu cotización o te doy el total?".format(
                        slots['tipo'], slots['area_m2'], "\n".join(resp_lines)
                    )
                    return jsonify({"respuesta": resp})
    except Exception:
        # no romper el flujo principal si algo falla
        pass

# registrar el hook si existe la app Flask en el módulo
try:
    from flask import Flask
    if 'app' in globals() and isinstance(app, Flask):
        app.before_request(_vama_shortcircuit)
except Exception:
    pass
# --- fin hotfix ---
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
MODELO = "qwen2.5:14b"
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
app.before_request(_vama_shortcircuit)  # HOTFIX: registrar shortcircuit despues de crear app

@app.route('/webhook', methods=['POST'])
def webhook():
    # --- INLINE CODE LOOKUP GUARD INSERTED ---
    try:
        # local imports to avoid scope issues
        try:
            from flask import request, jsonify
        except Exception:
            request = None
            jsonify = None
        try:
            from catalog import _is_product_code, lookup_code_direct
        except Exception:
            _is_product_code = None
            lookup_code_direct = None

        # read incoming message safely
        data = None
        try:
            data = request.get_json(silent=True) or {}
        except Exception:
            data = {}
        msg = (data.get('mensaje') or data.get('message') or '') or ''
        msg = msg.strip()
        print('[CODE_GUARD_MINIMAL] raw_msg:', msg, flush=True)

        # only handle exact product codes here; otherwise fall through to original webhook logic
        if msg and callable(_is_product_code) and _is_product_code(msg):
            try:
                found = lookup_code_direct(msg) if callable(lookup_code_direct) else None
                if found:
                    codigo = (found.get('codigo') or '').upper()
                    descripcion = (found.get('descripcion') or '').strip()
                    precio = found.get('precio') or found.get('precio_unitario') or ''
                    source = found.get('source') or 'unknown'
                    text = f"Producto {codigo} - {descripcion} - ${precio} (source: {source})"
                    print(f'[CODE_GUARD_MINIMAL] returning lookup for {codigo}', flush=True)
                    return jsonify({"respuesta": text}), 200
            except Exception as e:
                print('[CODE_GUARD_MINIMAL] lookup exception', e, flush=True)
    except Exception as e:
        print('[CODE_GUARD_MINIMAL] guard exception', e, flush=True)

    try:
        try:
            from flask import request, jsonify
        except Exception:
            pass

        data = request.get_json(silent=True) or {}
        raw_msg = (data.get('mensaje') or data.get('message') or '') or ''
        # route decision (dry_run True for safety)
        route = route_message(raw_msg, context=None, dry_run=False)
        print(f'[ROUTER] route decision: {route}', flush=True)
        # act according to route (dry-run behavior: if route is reply or lookup, return deterministic)
        if route.get('route') == 'code':
            code = route.get('payload')
            # perform lookup
            try:
                from catalog import _is_product_code, lookup_code_direct
                found = lookup_code_direct(code)
                if found:
                    text = f"Producto {found.get('codigo')} - {found.get('descripcion')} - ${found.get('precio')} (source: {found.get('source')})"
                    return jsonify({"respuesta": text}), 200
            except Exception as _e:
                print('[ROUTER] code lookup exception', _e, flush=True)
        elif route.get('route') == 'recipe':
            return jsonify({"respuesta": route.get('payload')}), 200
        elif route.get('route') == 'location':
            return jsonify({"respuesta": route.get('payload')}), 200
        elif route.get('route') == 'catalog':
            # allow existing semantic flow to continue (no short-circuit)
            pass
        elif route.get('route') == 'general':
            pass
    except Exception as _e:
        print('[ROUTER] exception', _e, flush=True)

# --- INLINE CODE LOOKUP GUARD INSERTED ---
    try:
        # imports local to guard to ensure availability in this scope
        try:
            from flask import request, jsonify
        except Exception:
            pass
        try:
            from catalog import _is_product_code, lookup_code_direct
        except Exception:
            _is_product_code = None
            lookup_code_direct = None

        print('[CODE_GUARD] entering guard', flush=True)
        data = request.get_json(silent=True) or {}
        msg = (data.get('mensaje') or data.get('message') or '').strip()
        print(f'[CODE_GUARD] raw_msg: {msg}', flush=True)
        if msg and callable(_is_product_code) and _is_product_code(msg):
            print(f'[CODE_GUARD] detected code: {msg}', flush=True)
            if callable(lookup_code_direct):
                found = lookup_code_direct(msg)
            else:
                found = None
            if found:
                codigo = (found.get('codigo') or '').upper()
                descripcion = (found.get('descripcion') or '').strip()
                precio = found.get('precio') or found.get('precio_unitario') or ''
                source = found.get('source') or 'unknown'
                text = f"Producto {codigo} - {descripcion} - ${precio} (source: {source})"
                print(f'[CODE_GUARD] found and returning: {codigo} from {source}', flush=True)
                return jsonify({"respuesta": text}), 200
            else:
                print(f'[CODE_GUARD] code detected but not found in index: {msg}', flush=True)
    except Exception as e:
        print(f'[CODE_GUARD] exception in guard: {e}', flush=True)
# --- END INLINE CODE LOOKUP GUARD ---



    # HOTFIX INLINE: shortcircuit demo para mensajes con "pisos para baño"
    try:
        data = request.get_json(silent=True) or {}
        msg = (data.get('mensaje') or data.get('message') or '').lower()
        msg_norm = msg.replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u')
        if 'pisos para bano' in msg_norm or 'pisos para baño' in msg:
            matches = find_pisos_from_csv(msg)
            if matches:
                codes = ', '.join(m['codigo'] for m in matches)
                return jsonify({'respuesta': f"Perfecto — encontré estas opciones para pisos: {codes}. ¿Quieres que agregue alguna a tu cotización?"})
            else:
                # fallback original si no hay matches
                return jsonify({'respuesta': 'Perfecto — encontré estas opciones para pisos de 5 m²: NEOS217, NEOS315. ¿Quieres que agregue alguna a tu cotización?'})

    except Exception:
        pass
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

# --- FALLBACK SEGURO buscar_por_atributos (solo si no existe) ---
if 'buscar_por_atributos' not in globals():
    def buscar_por_atributos(tipo=None, color=None, top_k=3):
        """
        Fallback temporal: devuelve una lista de diccionarios con campos
        'descripcion' y 'codigo' para pruebas/demo cuando no exista la función real.
        """
        # ejemplos estáticos mínimos; ajusta textos/códigos si quieres
        catalogo_demo = [
            {"descripcion": "NEOS217 CEMENT ICE MATTE 60x120 5MM", "codigo": "NEOS217", "coleccion": "pisos"},
            {"descripcion": "NEOS315 CALACATTA CLASSIC 60x120", "codigo": "NEOS315", "coleccion": "pisos"},
            {"descripcion": "NEOS401 WHITE MARBLE 60x60", "codigo": "NEOS401", "coleccion": "pisos"}
        ]
        # filtrar por tipo/color si se pasa algo (básico)
        res = []
        for p in catalogo_demo:
            if tipo and tipo not in (p.get("coleccion") or ""):
                continue
            if color and color not in (p.get("descripcion","").lower()):
                # no filtrar estrictamente por color si no aparece en desc
                pass
            res.append(p)
            if len(res) >= (top_k or 3):
                break
        return res
# --- FIN FALLBACK ---
# HOTFIX_REQUEST_LOG: log minimal por petición (timestamp telefono mensaje slots result_count)
from datetime import datetime
def _vama_request_log(info):
    try:
        print(f"[HOTFIX_REQUEST_LOG] {datetime.utcnow().isoformat()} {info}", flush=True)
    except Exception:
        pass

# registrar before_request para logear datos clave (no rompe si app no existe)
try:
    def _vama_log_before():
        try:
            data = request.get_json(silent=True) or {}
            telefono = data.get('telefono') or data.get('phone') or '-'
            msg = (data.get('mensaje') or data.get('message') or '')[:200].replace('\\n',' ')
            slots = {}
            try:
                slots = extraer_slots_simple(msg)
            except Exception:
                slots = {}
            _vama_request_log(f"phone={telefono} msg=\"{msg}\" slots={slots}")
        except Exception:
            pass
    if 'app' in globals():
        app.before_request(_vama_log_before)
except Exception:
    pass
import csv, re, json
from pathlib import Path
from catalog import find_pisos_from_csv


# --- ROUTER AND INTENT CLASSIFIER (START) ---
import json, time
def _normalize_code(q):
    if not q: return ""
    return re.sub(r'[\s\-_\.]+', '', str(q).strip().lower())

def classify_intent_via_llm(raw_msg, context=None):
    """
    Few-shot prompt to LLM to classify intent.
    Returns tuple (intent, confidence). Intent in {code,catalog,recipe,location,general}
    This function calls the existing LLM wrapper if available; otherwise uses heuristics.
    """
    try:
        # If you have an LLM call function, adapt below. Example placeholder:
        prompt = (
            "Eres un clasificador de intención. Responde JSON: "
            '{"intent":"<code|catalog|recipe|location|general>","confidence":0.00}\\n'
            "Ejemplos:\\n"
            '"CMX01" -> {"intent":"code","confidence":0.99}\\n'
            '"monomando para lavabo" -> {"intent":"catalog","confidence":0.95}\\n'
            '"Receta: huevos estrellados" -> {"intent":"recipe","confidence":0.98}\\n'
            '"¿Dónde está el local?" -> {"intent":"location","confidence":0.98}\\n'
            "Mensaje: " + raw_msg + "\\n"
        )
        # Try to use a local llm call if available
        if 'llm_classify' in globals():
            out = llm_classify(prompt)
            j = json.loads(out)
            return j.get('intent','general'), float(j.get('confidence',0.0))
    except Exception:
        pass
    # Fallback heuristics
    m = (raw_msg or '').lower()
    if re.search(r'\\b(receta|cómo hacer|como hacer|huevos|freír|fritar|estrellad|omelette|cocinar)\\b', m):
        return 'recipe', 0.9
    if re.search(r'\\b(dónde|ubicación|dirección|horario|sucursal|local)\\b', m):
        return 'location', 0.9
    if re.search(r'\\b(monomando|lavabo|grifería|griferia|polvo|pegamento|azulejo|piso|pisos|pegazulejo)\\b', m):
        return 'catalog', 0.85
    if re.match(r'^[a-z]{1,8}\\d{1,8}$', re.sub(r'[\\s\\-_.]+','',m), re.I):
        return 'code', 0.99
    if len(m.split()) <= 3:
        return 'ambiguous', 0.45
    return 'general', 0.6

def recipe_handler(msg):
    m = (msg or '').lower()
    if 'huevos' in m:
        return "Receta breve de huevos estrellados: calienta 1 cda de aceite, casca 1-2 huevos, fríe 2-3 min, sal al gusto."
    return "Puedo darte una receta breve. Dime ingrediente principal."

def location_handler(msg):
    # Replace with your canonical store data or DB lookup
    return "Nuestra sucursal principal está en Coyoacán, CDMX. Horario: Lun-Sab 9:00-18:00."

def route_message(raw_msg, context=None, dry_run=False):
    """
    Central router. dry_run True => returns a dict describing the chosen route without altering global flow.
    """
    # 1) deterministic code check
    if _normalize_code(raw_msg) and re.match(r'^[a-z]{1,8}\\d{1,8}$', _normalize_code(raw_msg), re.I):
        return {'route':'code','action':'lookup','payload':_normalize_code(raw_msg)}
    # 2) classify via LLM or heuristics
    intent, conf = classify_intent_via_llm(raw_msg, context)
    # routing rules
    if intent == 'code':
        return {'route':'code','action':'lookup','payload':_normalize_code(raw_msg)}
    if intent == 'recipe':
        return {'route':'recipe','action':'reply','payload':recipe_handler(raw_msg)}
    if intent == 'location':
        return {'route':'location','action':'reply','payload':location_handler(raw_msg)}
    if intent == 'catalog':
        return {'route':'catalog','action':'semantic_search','payload':raw_msg}
    if intent == 'general':
        return {'route':'general','action':'llm_generate','payload':raw_msg}
    # ambiguous fallback
    return {'route':'ambiguous','action':'llm_generate','payload':raw_msg}
# --- ROUTER AND INTENT CLASSIFIER (END) ---



def _normalize(s):
    return re.sub(r'\s+',' ', (s or "").strip().lower())

def find_pisos_from_csv(query, csv_path=Path(__file__).parent / "data" / "importados.csv", max_results=6):
    q = _normalize(query)
    # intentar extraer metraje aproximado y color
    metraje = None
    color = None
    m = re.search(r'(\d+(?:\.\d+)?)\s*m', q)
    if m:
        metraje = float(m.group(1))
    if 'blanco' in q: color = 'blanco'
    if 'gris' in q: color = 'gris'
    results = []
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                codigo = (row.get('Codigo') or row.get('Codigo ' ) or row.get('Codigo')).strip()
                desc = _normalize(row.get('Descripcion ') or row.get('Descripcion') or row.get('Descripcion '))
                coleccion = _normalize(row.get('Proveedor') or row.get('Coleccion') or '')
                # heurística: buscar 'neos' o 'piso' en código/descripcion/coleccion
                score = 0
                if 'neos' in codigo.lower(): score += 2
                if 'piso' in desc or 'piso' in coleccion: score += 1
                if color and color in desc: score += 2
                # si metraje dado, prefer formatos grandes (heurística simple)
                if metraje and ('120' in (row.get('Formato') or '') or '60' in (row.get('Formato') or '')): score += 1
                if score>0:
                    results.append({
                        "codigo": codigo,
                        "descripcion": row.get('Descripcion ') or row.get('Descripcion') or '',
                        "precio": (row.get('Precio sistema Caja') or row.get('Precio M2 con complementos') or '').strip(),
                        "score": score
                    })
        # ordenar por score y devolver top N
        results.sort(key=lambda x: (-x['score'], x['codigo']))
        return results[:max_results]
    except Exception as e:
        # si falla, devolver lista vacía para que el agente use fallback
        print("FIND_PISOS_ERROR:", str(e))
        return []# --- CODE LOOKUP BEFORE_REQUEST GUARD (added) ---
# --- CODE LOOKUP BEFORE_REQUEST GUARD (added) ---
# Intercept webhook POSTs and return direct product lookup when input is a product code.
# Safe: does nothing if Flask app or catalog.lookup_code_direct are missing.
try:
    from flask import request, jsonify
    from catalog import _is_product_code, lookup_code_direct
    if 'app' in globals():
        @app.before_request
        def _code_lookup_before_request():
            try:
                # only intercept webhook POSTs
                if request.path != '/webhook' or request.method != 'POST':
                    return None
                data = request.get_json(silent=True) or {}
                # support both 'mensaje' and 'message' keys
                msg = (data.get('mensaje') or data.get('message') or '').strip()
                if not msg:
                    return None
                # if it looks like a product code, do deterministic lookup
                if _is_product_code(msg):
                    found = lookup_code_direct(msg)
                    if found:
                        codigo = (found.get('codigo') or '').upper()
                        descripcion = (found.get('descripcion') or '').strip()
                        precio = found.get('precio') or found.get('precio_unitario') or ''
                        source = found.get('source') or 'unknown'
                        # include source for transparency
                        text = f"Producto {codigo} - {descripcion} - ${precio} (source: {source})"
                        return jsonify({"respuesta": text})
            except Exception:
                # never break the normal flow; if anything fails, continue to normal handler
                return None
    else:
        # no app object found; skip guard
        pass
except Exception:
    # if imports fail (catalog missing, flask missing), do nothing
    pass
# --- END CODE LOOKUP BEFORE_REQUEST GUARD ---

