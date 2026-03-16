#!/usr/bin/env python3
"""
VAMA 2.4 - VERSIÓN CORREGIDA
Fixes:
- Búsqueda por tipo/color/área funcional
- Memoria persistente de productos mostrados
- No repite saludos
- Muestra productos guardados cuando se pide "de nuevo"
- Prompt forzado a usar datos reales
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

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
CHROMA_PATH = "chroma_db_v3"
MEMORIA_PATH = "memoria_vama.pkl"
MODELO = "qwen2.5:14b"
CACHE_SIMPLES = {}
CACHE_TTL = 3600

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("🔌 Conectando a Ollama...")
try:
    client_ollama = ollama.Client(host='http://127.0.0.1:11434')
    models = client_ollama.list()
    print(f"✅ Ollama conectado. Modelos: {[m['model'] for m in models['models']]}")
    if MODELO not in [m['model'] for m in models['models']]:
        print(f"⚠️  ADVERTENCIA: {MODELO} no está instalado. Ejecuta: ollama pull {MODELO}")
except Exception as e:
    print(f"❌ ERROR: No se pudo conectar a Ollama: {e}")
    client_ollama = None

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
                "ultimos_productos": [],  # ← NUEVO: persistir productos mostrados
                "ultima_pregunta": None,
                "ultimo_producto_mostrado": None
            }
        else:
            # Asegurar claves nuevas existan
            if "ultimos_productos" not in self.datos[usuario_id]:
                self.datos[usuario_id]["ultimos_productos"] = []
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
    def __init__(self, usuario_id, nombre, m2_previos=0, productos_previos=None):
        self.usuario_id = usuario_id
        self.nombre = nombre
        self.ultimos_productos = productos_previos or []  # ← Recuperar de memoria
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
            # ← Recuperar productos previos de memoria persistente
            productos_previos = hist.get("ultimos_productos", [])
            self.sesiones[usuario_id] = EstadoConversacion(
                usuario_id, nombre_real, 
                hist.get("m2_proyecto", 0),
                productos_previos
            )
        self.sesiones[usuario_id].actualizar()
        return self.sesiones[usuario_id]

gestor_sesiones = GestorSesiones()

# ============================================================================
# BÚSQUEDA
# ============================================================================
class DB:
    def buscar(self, query, colecciones=None, top_k=5, filtros=None):
        if colecciones is None:
            colecciones = list(cols.keys())
        
        resultados = []
        for nombre in colecciones:
            if nombre not in cols: continue
            try:
                # Aumentar resultados para filtrar después
                k = top_k * 2 if filtros else top_k
                r = cols[nombre].query(query_texts=[query], n_results=k)
                
                for meta in r["metadatas"][0]:
                    meta['coleccion'] = nombre
                    # Aplicar filtros si existen
                    if filtros:
                        match = True
                        if filtros.get('color') and meta.get('color'):
                            if filtros['color'].lower() not in meta.get('color', '').lower():
                                match = False
                        if filtros.get('tipo') == 'baños' and meta.get('uso'):
                            usos = meta.get('uso', '').lower()
                            if 'baño' not in usos and 'bano' not in usos and 'muro' not in usos:
                                match = False
                        if not match:
                            continue
                    resultados.append(meta)
                    
            except Exception as e:
                print(f"Error buscando en {nombre}: {e}")
                continue
        
        # Ordenar por relevancia
        query_words = [w for w in re.sub(r'[^\w\s]', '', query.lower()).split() if len(w) > 3]
        if resultados and query_words:
            resultados.sort(
                key=lambda x: sum(w in x.get('descripcion','').lower() for w in query_words), 
                reverse=True
            )
        
        return resultados[:top_k]

db = DB()

# ============================================================================
# FORMATEADOR DE PRODUCTOS
# ============================================================================
def formatear_productos_russ(productos, con_precio=True):
    if not productos: return "Ninguno"
    texto = ""
    for i, p in enumerate(productos[:3], 1):
        codigo = p.get('codigo', 'N/A')
        desc = p.get('descripcion', 'Producto')
        formato = p.get('formato', 'N/A')
        color = p.get('color', '')
        
        precio_str = "Precio: por confirmar"
        if con_precio:
            precio = None
            for k, v in p.items():
                if any(x in k.upper() for x in ["PRECIO", "OFERTA", "FINAL", "UNITARIO", "CAJA"]):
                    if v and str(v).strip() not in ["0", "0.0", "", "None"]:
                        try:
                            precio = float(v)
                            precio_str = f"${precio:.2f}"
                            break
                        except:
                            continue
        
        color_str = f" | Color: {color}" if color else ""
        texto += f"{i}. Código: {codigo} | {desc} | Formato: {formato}{color_str} | {precio_str}\n"
    return texto.strip()

# ============================================================================
# UTILIDADES
# ============================================================================
def extraer_precio_de_meta(prod):
    for k, v in prod.items():
        if any(x in k.upper() for x in ["PRECIO", "OFERTA", "FINAL", "UNITARIO", "CAJA"]):
            try:
                return float(v)
            except:
                continue
    return None

def descripcion_corta(prod):
    return prod.get('descripcion') or prod.get('codigo') or "Producto"

def extraer_slots_mejorado(texto):
    """Extrae tipo, color y área del mensaje"""
    t = (texto or "").lower()
    slots = {'tipo': None, 'color': None, 'area_m2': None}
    
    # Tipo
    if re.search(r'\b(pisos?|azulejos?|porcelanatos?|ceramicos?)\b', t):
        slots['tipo'] = 'pisos'
    elif re.search(r'\b(pared(es)?|muros?|fachadas?)\b', t):
        slots['tipo'] = 'paredes'
    elif re.search(r'\b(baño|bano|regadera|wc)\b', t):
        slots['tipo'] = 'baños'
    
    # Color
    colores = ['blanco', 'gris', 'negro', 'beige', 'mármol', 'marmol', 'carrara', 'negro', 'azul', 'verde']
    for c in colores:
        if c in t:
            slots['color'] = c.replace('marml', 'mármol')
            break
    
    # Área
    marea = re.search(r'(\d+(?:[.,]\d+)?)\s*(m2|m²|metros?|mts)', t)
    if marea:
        try:
            slots['area_m2'] = float(marea.group(1).replace(',', '.'))
        except:
            pass
    
    return slots

# ============================================================================
# CEREBRO PRINCIPAL
# ============================================================================
def generar_respuesta_llm(mensaje, estado, usuario_id):
    msg_limpio = mensaje.strip() if mensaje else ""
    msg_low = msg_limpio.lower()
    hist = memoria_largo_plazo.obtener(usuario_id)
    
    num_interaccion = hist.get("interacciones", 0)
    cache_key = f"{usuario_id}:{msg_limpio}"
    tiempo_actual = time.time()
    
    # ===== NUEVO: Detectar si quiere ver productos de nuevo =====
    quiere_ver_de_nuevo = any(f in msg_low for f in [
        'muestra', 'ver', 'de nuevo', 'otra vez', 'cuales', 'cuáles', 
        'productos', 'opciones', 'lista', 'mostra', 'enseña', 'enseñar',
        'cual', 'cuál', 'que opciones', 'que productos'
    ]) and not any(c in msg_low for c in ['compro', 'llevo', 'agrega', 'cotiza'])
    
    if quiere_ver_de_nuevo:
        # Priorizar productos en sesión actual, luego memoria persistente
        productos_a_mostrar = estado.ultimos_productos or hist.get("ultimos_productos", [])
        if productos_a_mostrar:
            productos_formateados = formatear_productos_russ(productos_a_mostrar)
            return f"Te muestro de nuevo las opciones disponibles:\n\n{productos_formateados}\n\n¿Cuál te interesa? Equipo VAMA."
    
    # ===== Manejo de afirmativos =====
    afirmativos = {"si", "sí", "claro", "ok", "vale", "correcto", "sí.", "si.", "si,", "sí,"}
    if msg_low in afirmativos and hist.get("ultima_pregunta"):
        ultima = hist.get("ultima_pregunta")
        if ultima == "confirmar_cantidad" and hist.get("ultimo_producto_mostrado"):
            prod = hist["ultimo_producto_mostrado"]
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
                hist["productos_cotizados"][desc] = {
                    "precio": precio if precio not in [0, "0", "0.0", ""] else None, 
                    "cantidad": cantidad
                }
            hist["ultima_pregunta"] = None
            memoria_largo_plazo.guardar()
            
            if hist["productos_cotizados"][desc]["precio"] is None:
                return f"✅ Agregué {cantidad} de {desc} (precio por confirmar). ¿Algo más o te paso el total? Equipo VAMA."
            else:
                subtotal = hist["productos_cotizados"][desc]["precio"] * cantidad
                return f"✅ Agregué {cantidad} de {desc}. Subtotal: ${subtotal:.2f}. ¿Algo más? Equipo VAMA."
    
    # ===== Caché =====
    if cache_key in CACHE_SIMPLES:
        timestamp, respuesta_cache = CACHE_SIMPLES[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            return respuesta_cache
    
    # ===== Anti-duplicado =====
    if len(msg_limpio) < 3:
        if estado.ultima_respuesta_enviada:
            return estado.ultima_respuesta_enviada
        return "Estoy aquí para ayudarte. ¿Qué producto necesitas? Equipo VAMA."
    
    tiempo_ultimo = hist.get("timestamp_ultima", 0)
    if tiempo_actual - tiempo_ultimo < 5 and msg_limpio == hist.get("ultimo_mensaje_procesado", ""):
        return estado.ultima_respuesta_enviada or "Procesando..."
    
    hist["ultimo_mensaje_procesado"] = msg_limpio
    hist["timestamp_ultima"] = tiempo_actual
    
    # ===== Comandos especiales =====
    if any(w in msg_low for w in ['nueva cotizacion', 'nueva cotización', 'empezar de nuevo', 'reiniciar', 'borrar todo']):
        hist["productos_cotizados"] = {}
        hist["m2_proyecto"] = 0
        hist["interacciones"] = 0
        hist["ultimos_productos"] = []
        estado.m2_proyecto = 0
        estado.ultimos_productos = []
        hist["ultima_pregunta"] = None
        hist["ultimo_producto_mostrado"] = None
        memoria_largo_plazo.guardar()
        return "Listo, reinicié tu cotización. ¿Qué necesitas cotizar? Equipo VAMA."
    
    # ===== Total =====
    es_pedido_total = any(p in msg_low for p in ["total", "cotización", "cuánto es", "suma", "precio final"])
    if es_pedido_total and hist["productos_cotizados"]:
        total = 0.0
        detalle = "**Resumen:**\n"
        for desc, data in hist["productos_cotizados"].items():
            if data["precio"] is None:
                detalle += f"- {desc}: {data['cantidad']} (precio por confirmar)\n"
            else:
                subtotal = data["precio"] * data["cantidad"]
                total += subtotal
                detalle += f"- {desc}: {data['cantidad']} x ${data['precio']} = ${subtotal:.2f}\n"
        if total > 0:
            detalle += f"\n**TOTAL: ${total:.2f}**\n¿Confirmo disponibilidad? Equipo VAMA."
        else:
            detalle += "\n**TOTAL: por confirmar**\n¿Te aviso cuando tenga precios? Equipo VAMA."
        CACHE_SIMPLES[cache_key] = (tiempo_actual, detalle)
        return detalle
    
    # ===== Extraer slots y m2 =====
    slots = extraer_slots_mejorado(msg_limpio)
    match_m2 = re.findall(r'(\d+)\s*(?:m2|metros|mts|cuadrados)', msg_low)
    if match_m2:
        estado.m2_proyecto = int(match_m2[0])
        hist["m2_proyecto"] = estado.m2_proyecto
    
    nombre_cliente = hist["nombre"] or "Cliente"
    
    # ===== Detectar saludos (solo si es primera interacción) =====
    saludos = ['hola', 'buenos dias', 'buenas tardes', 'buenas noches', 'que tal', 'hey', 'saludos', 'buen dia']
    es_saludo = any(s in msg_low for s in saludos) and len(msg_low) < 50 and num_interaccion < 2
    
    # ===== BÚSQUEDA INTELIGENTE =====
    if not es_saludo:
        # Determinar colecciones
        es_griferia = any(w in msg_low for w in ['monomando', 'grifo', 'mezcladora', 'llave', 'regadera'])
        es_lavabo = any(w in msg_low for w in ['lavabo', 'lavamanos'])
        es_sanitario = any(w in msg_low for w in ['wc', 'inodoro', 'taza', 'sanitario'])
        es_mueble = any(w in msg_low for w in ['mueble', 'gabinete'])
        es_tinaco = any(w in msg_low for w in ['tinaco', 'cisterna'])
        es_espejo = any(w in msg_low for w in ['espejo'])
        es_tarja = any(w in msg_low for w in ['tarja', 'fregadero'])
        es_herramienta = any(w in msg_low for w in ['herramienta'])
        es_pegamento = any(w in msg_low for w in ['pegazulejo', 'pegamento', 'pega', 'adesivo', 'polvo'])
        es_piso = any(w in msg_low for w in ['piso', 'azulejo', 'porcelanato', 'ceramica', 'marmol', 'loseta'])
        
        colecciones_buscar = []
        filtros = {}
        
        if es_pegamento:
            colecciones_buscar = ["polvos", "otras"]
        elif es_griferia: 
            colecciones_buscar = ["griferia"]
        elif es_lavabo: 
            colecciones_buscar = ["lavabos"]
        elif es_sanitario: 
            colecciones_buscar = ["sanitarios"]
        elif es_mueble: 
            colecciones_buscar = ["muebles"]
        elif es_tinaco: 
            colecciones_buscar = ["tinacos"]
        elif es_espejo: 
            colecciones_buscar = ["espejos"]
        elif es_tarja: 
            colecciones_buscar = ["tarjas"]
        elif es_herramienta: 
            colecciones_buscar = ["herramientas"]
        else:
            # Default: pisos/paredes (nacionales + importados)
            colecciones_buscar = ["nacionales", "importados"]
            # Filtros por tipo/color si se detectaron
            if slots.get('tipo') == 'baños':
                filtros['tipo'] = 'baños'
            if slots.get('color'):
                filtros['color'] = slots['color']
        
        # Realizar búsqueda
        resultados = []
        for coleccion in colecciones_buscar:
            if coleccion in cols:
                try:
                    k = 5 if coleccion in ["nacionales", "importados"] else 3
                    r = db.buscar(msg_limpio, [coleccion], top_k=k, filtros=filtros if coleccion in ["nacionales", "importados"] else None)
                    resultados.extend(r)
                except Exception as e:
                    print(f"Error en {coleccion}: {e}")
        
        # Guardar en sesión Y en memoria persistente
        if resultados:
            estado.guardar_productos(resultados[:6])
            hist["ultimos_productos"] = resultados[:6]
            memoria_largo_plazo.guardar()
            print(f"DEBUG: {len(resultados)} resultados, guardados {len(resultados[:6])}")
        
        # ===== COMPRA DIRECTA =====
        es_compra = any(w in msg_low for w in [
            'me llevo', 'quiero', 'agrega', 'compro', 'añade', 
            'cotiza', 'incluye', 'pon', 'agregar', 'deme', 'dame'
        ])
        
        if es_compra and resultados:
            prod = resultados[0]
            desc = descripcion_corta(prod)
            precio = extraer_precio_de_meta(prod)
            
            # Calcular cantidad por m2
            cantidad = 1
            if estado.m2_proyecto and prod.get('metraje_caja'):
                try:
                    metraje = float(prod['metraje_caja'])
                    if metraje > 0:
                        cantidad = math.ceil(estado.m2_proyecto / metraje)
                except:
                    pass
            
            if desc in hist["productos_cotizados"]:
                hist["productos_cotizados"][desc]["cantidad"] += cantidad
            else:
                hist["productos_cotizados"][desc] = {
                    "precio": precio if precio else None, 
                    "cantidad": cantidad
                }
            
            hist["ultimos_productos"] = resultados[:6]  # Guardar por si pide ver de nuevo
            memoria_largo_plazo.guardar()
            
            if precio:
                subtotal = precio * cantidad
                return f"✅ Agregué {cantidad} de {desc} (${subtotal:.2f}). ¿Algo más? Equipo VAMA."
            else:
                return f"✅ Agregué {cantidad} de {desc} (precio por confirmar). ¿Algo más? Equipo VAMA."
    
    # ===== PREPARAR PARA LLM =====
    # Recuperar productos de donde sea disponible
    productos_para_mostrar = estado.ultimos_productos or hist.get("ultimos_productos", [])
    productos_russ = formatear_productos_russ(productos_para_mostrar)
    
    # Guardar primer producto para confirmación
    if productos_para_mostrar:
        p0 = productos_para_mostrar[0]
        hist["ultimo_producto_mostrado"] = {
            "codigo": p0.get("codigo"),
            "descripcion": p0.get("descripcion"),
            "formato": p0.get("formato"),
            "metraje_caja": p0.get("metraje_caja"),
            **{k: v for k, v in p0.items() if "precio" in k.lower()}
        }
        memoria_largo_plazo.guardar()
    
    # Historial reciente
    historial_reciente = ""
    if hist.get("ultimos_mensajes"):
        for m in hist["ultimos_mensajes"][-4:]:
            role = "Cliente" if m["role"] == "user" else "VAMA"
            historial_reciente += f"{role}: {m['content']}\n"
    
    # ===== PROMPT MEJORADO =====
    prompt = f"""Eres VAMA, asistente de ventas de VAMA (materiales y acabados).

CONTEXTO:
- Cliente: {nombre_cliente}
- Metraje: {estado.m2_proyecto} m2
- Historial: {historial_reciente or 'Primera conversación'}
- PRODUCTOS DISPONIBLES (USAR SOLO ESTOS):
{productos_russ if productos_para_mostrar else 'No se encontraron productos. Pide más detalles al cliente.'}

REGLAS ESTRICTAS:
1. USA ÚNICAMENTE los productos listados arriba. NO inventes códigos ni precios.
2. Si no hay productos, di: "No encontré opciones con esas características. ¿Puedes darme más detalles?"
3. Mensaje corto (2-6 líneas), tono profesional y cercano.
4. Si el cliente ya mencionó m2, NO vuelvas a preguntar.
5. Menciona máximo 3 productos de la lista.
6. Si falta precio, di exactamente: "Precio por confirmar".

Tu respuesta (solo el mensaje para WhatsApp):"""

    hist["interacciones"] = num_interaccion + 1
    hist["ultima_visita"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    memoria_largo_plazo.guardar()

    # ===== LLM =====
    try:
        if client_ollama is None:
            raise ConnectionError("Ollama no disponible")
        
        res = client_ollama.generate(
            model=MODELO,
            prompt=prompt,
            options={
                "temperature": 0.4,  # ← Más determinista
                "num_predict": 200,   # ← Más corto
                "num_ctx": 2048,
                "stop": ["CLIENTE:", "USUARIO:", "Producto 4:", "4."]
            }
        )['response'].strip()
        
        # Limpieza
        res = re.sub(r'\*\*?|\*|__|\|', '', res)
        res = res.replace("  ", " ").strip()
        
        # Guardar historial
        if "ultimos_mensajes" not in hist:
            hist["ultimos_mensajes"] = []
        hist["ultimos_mensajes"].append({"role": "user", "content": msg_limpio})
        hist["ultimos_mensajes"].append({"role": "assistant", "content": res})
        hist["ultimos_mensajes"] = hist["ultimos_mensajes"][-10:]
        
        # Detectar si preguntó cantidad
        if re.search(r'¿.*(cuántas|cajas|cantidad).*m2', res.lower()):
            hist["ultima_pregunta"] = "confirmar_cantidad"
            memoria_largo_plazo.guardar()
        
        estado.ultima_respuesta_enviada = res
        CACHE_SIMPLES[cache_key] = (tiempo_actual, res)
        return res
        
    except Exception as e:
        print(f"DEBUG: Error LLM: {e}")
        # Fallback: mostrar productos directamente si existen
        if productos_para_mostrar:
            return f"Encontré estas opciones:\n\n{productos_russ}\n\n¿Cuál te interesa? Equipo VAMA."
        return f"Entendido {nombre_cliente}, revisando disponibilidad. Equipo VAMA."

# ============================================================================
# API
# ============================================================================
app = Flask(__name__)

@app.route('/webhook', methods=['POST'], strict_slashes=False)
def webhook():
    data = request.get_json() or {}
    tel = str(data.get('telefono') or '0')
    nom = data.get('nombre') or 'Cliente'
    msg = data.get('mensaje') or ""
    
    print(f"\n[WEBHOOK] {datetime.now().strftime('%H:%M:%S')} | {nom} ({tel}): {msg[:60]}")
    
    sesion = gestor_sesiones.obtener(tel, nom)
    respuesta = generar_respuesta_llm(msg, sesion, tel)
    
    print(f"[RESPONSE] {respuesta[:80]}...")
    return jsonify({"respuesta": respuesta})

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        import sys
        print('ERROR STARTING FLASK:', e, file=sys.stderr)