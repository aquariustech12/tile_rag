#!/usr/bin/env python3
"""
VAMA 3.0 - REESCRITURA COMPLETA
- 12 colecciones: nacionales, importados, griferia, lavabos, sanitarios, 
  muebles, tinacos, espejos, tarjas, herramientas, polvos, otras
- Código decide el flujo, LLM solo habla
- Memoria real en cada paso
"""
import sys, os, re, math, pickle, json, time
from datetime import datetime
from flask import Flask, request, jsonify

# OLLAMA
import ollama
ollama_client = ollama.Client(host='http://127.0.0.1:11434')
MODELO = "qwen2.5:3b"

# CHROMA
import chromadb
from chromadb.utils import embedding_functions
os.environ["TOKENIZERS_PARALLELISM"] = "false"
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-m3", device="cpu"
)
chroma = chromadb.PersistentClient(path="chroma_db_v3")

# TUS 12 COLECCIONES EXACTAS
COLECCIONES = {
    "nacionales": chroma.get_collection("nacionales", embedding_function=embed_fn),
    "importados": chroma.get_collection("importados", embedding_function=embed_fn),
    "griferia": chroma.get_or_create_collection("griferia", embedding_function=embed_fn),
    "lavabos": chroma.get_or_create_collection("lavabos", embedding_function=embed_fn),
    "sanitarios": chroma.get_or_create_collection("sanitarios", embedding_function=embed_fn),
    "muebles": chroma.get_or_create_collection("muebles", embedding_function=embed_fn),
    "tinacos": chroma.get_or_create_collection("tinacos", embedding_function=embed_fn),
    "espejos": chroma.get_or_create_collection("espejos", embedding_function=embed_fn),
    "tarjas": chroma.get_or_create_collection("tarjas", embedding_function=embed_fn),
    "herramientas": chroma.get_or_create_collection("herramientas", embedding_function=embed_fn),
    "polvos": chroma.get_or_create_collection("polvos", embedding_function=embed_fn),
    "otras": chroma.get_or_create_collection("otras", embedding_function=embed_fn)
}

# MAPEO DE INTENCIONES A COLECCIONES
MAP_COLECCIONES = {
    'piso': ['nacionales', 'importados'],
    'azulejo': ['nacionales', 'importados'],
    'porcelanato': ['nacionales', 'importados'],
    'muro': ['nacionales', 'importados'],
    'pared': ['nacionales', 'importados'],
    'grifo': ['griferia'],
    'monomando': ['griferia'],
    'llave': ['griferia'],
    'regadera': ['griferia'],
    'lavabo': ['lavabos'],
    'wc': ['sanitarios'],
    'inodoro': ['sanitarios'],
    'taza': ['sanitarios'],
    'sanitario': ['sanitarios'],
    'mueble': ['muebles'],
    'gabinete': ['muebles'],
    'tinaco': ['tinacos'],
    'cisterna': ['tinacos'],
    'espejo': ['espejos'],
    'tarja': ['tarjas'],
    'fregadero': ['tarjas'],
    'herramienta': ['herramientas'],
    'pegamento': ['polvos', 'otras'],
    'pega': ['polvos', 'otras'],
    'adesivo': ['polvos', 'otras'],
    'cemix': ['polvos', 'otras']
}

# MEMORIA
class Memoria:
    def __init__(self):
        self.datos = self._cargar()
    
    def _cargar(self):
        try:
            with open("memoria_vama.pkl", 'rb') as f:
                return pickle.load(f)
        except:
            return {}
    
    def guardar(self):
        with open("memoria_vama.pkl", 'wb') as f:
            pickle.dump(self.datos, f)
    
    def get(self, uid):
        if uid not in self.datos:
            self.datos[uid] = {
                'nombre': '',
                'carrito': [],
                'ultimos_productos': [],
                'm2': 0,
                'ultimo_mensaje': '',
                'contador': 0
            }
        else:
            # Asegurar todos los campos en usuarios viejos
            defaults = {
                'nombre': '',
                'carrito': [],
                'ultimos_productos': [],
                'm2': 0,
                'ultimo_mensaje': '',
                'contador': 0
            }
            for key, val in defaults.items():
                if key not in self.datos[uid]:
                    self.datos[uid][key] = val

        return self.datos[uid]

memoria = Memoria()

# UTILIDADES
def extraer_precio(meta):
    for k in ['precio_caja', 'precio_unitario', 'precio_m2', 'precio_final', 'precio']:
        try:
            v = float(meta.get(k, 0))
            if v > 10:
                return v
        except:
            continue
    return None

def buscar(query, intenciones=None, color=None, top_k=5):
    """Busca en colecciones según intención"""
    if intenciones:
        cols = set()
        for intencion in intenciones:
            cols.update(MAP_COLECCIONES.get(intencion, ['nacionales', 'importados']))
    else:
        cols = list(COLECCIONES.keys())
    
    resultados = []
    for col_name in cols:
        if col_name not in COLECCIONES:
            continue
        try:
            col = COLECCIONES[col_name]
            res = col.query(query_texts=[query], n_results=top_k * 2)
            
            for meta in res['metadatas'][0]:
                if color and color.lower() not in meta.get('color', '').lower():
                    continue
                
                resultados.append({
                    'codigo': meta.get('codigo', 'N/A'),
                    'descripcion': meta.get('descripcion', ''),
                    'formato': meta.get('formato', ''),
                    'color': meta.get('color', ''),
                    'm2_caja': float(meta.get('metraje_caja', 0) or 0),
                    'precio': extraer_precio(meta),
                    'coleccion': col_name
                })
        except Exception as e:
            print(f"Error en {col_name}: {e}")
    
    # Ordenar: coincidencia exacta primero, luego con precio
    q = query.lower()
    resultados.sort(key=lambda x: (
        q in x['descripcion'].lower(),
        x['precio'] is not None,
        x['descripcion']
    ), reverse=True)
    
    return resultados[:top_k]

def detectar_intenciones(msg):
    msg = msg.lower()
    ints = []
    for palabra, cols in MAP_COLECCIONES.items():
        if palabra in msg:
            ints.append(palabra)
    return ints if ints else ['piso']

def detectar_color(msg):
    colores = ['blanco', 'gris', 'negro', 'beige', 'marmol', 'carrara', 'azul', 'verde', 'rojo']
    for c in colores:
        if c in msg.lower():
            return c
    return None

def detectar_m2(msg):
    m = re.search(r'(\d+)\s*(?:m2|m²|metros|mts)', msg.lower())
    return int(m.group(1)) if m else None

def formatear_lista(productos):
    if not productos:
        return "No encontré productos."
    lineas = []
    for i, p in enumerate(productos[:3], 1):
        precio = f"${p['precio']:.2f}" if p['precio'] else "Precio por confirmar"
        lineas.append(f"{i}. {p['codigo']} - {p['descripcion']} ({p['formato']}) - {precio}")
    return "\n".join(lineas)

def calcular_cantidad(m2_proyecto, m2_caja):
    if m2_proyecto and m2_caja:
        return max(1, math.ceil(m2_proyecto / m2_caja))
    return 1

def hablar(prompt_contexto, temperatura=0.2, max_tokens=100):
    """LLM solo para generar texto, nunca para decidir"""
    try:
        r = ollama_client.generate(
            model=MODELO,
            prompt=prompt_contexto + "\n\nREGLA: Responde en 2 líneas, sin saludar de nuevo, sin 'me veo' ni 'claro'. Directo al punto.",
            options={
                "temperature": temperatura,
                "num_predict": max_tokens,
                "stop": ["Cliente:", "Usuario:", "\n\n", "Me veo", "Claro,"]
            }
        )
        return r['response'].strip().replace("**", "").replace("*", "")
    except Exception as e:
        print(f"LLM error: {e}")
        return "¿Te gustaría ver más opciones? Equipo VAMA."

# FLUJOS
def flujo_presentar_productos(user, productos, es_follow_up=False):
    """Muestra productos, LLM solo decora"""
    # PRIMERO: siempre guardar productos
    lista = formatear_lista(productos)
    user['ultimos_productos'] = productos
    
    # SEGUNDO: verificar m2 solo si es primera vez y no lo tenemos
    if not user['m2'] and not es_follow_up:
        # Guardar productos pero pedir m2 primero
        return "¿Para cuántos m2 necesitas el material? Así calculo las cajas exactas. Equipo VAMA."
    
    # TERCERO: generar texto con LLM
    if es_follow_up:
        prompt = f"""El cliente ya está en conversación. Muestra estas opciones naturalmente, sin saludar.
        Productos:
        {lista}
        Pregunta cuál le interesa (diga 1, 2 o 3). 2 líneas."""
    else:
        prompt = f"""Presenta opciones al cliente. Productos: {lista}. Pregunta cuál prefiere. 2 líneas."""
    
    texto = hablar(prompt, temperatura=0.4)
    return f"{texto}\n\n{lista}\n\n¿Cuál te interesa? (dime 1, 2 o 3) Equipo VAMA."

def flujo_agregar_producto(user, msg):
    """Agrega al carrito, detecta número o código"""
    if not user['ultimos_productos']:
        return "Primero dime qué productos buscas. ¿Qué material necesitas? Equipo VAMA."
    
    seleccion = None
    
    # Por número
    num = re.search(r'\b([123])\b', msg)
    if num:
        idx = int(num.group(1)) - 1
        if 0 <= idx < len(user['ultimos_productos']):
            seleccion = user['ultimos_productos'][idx]
    
    # Por código
    if not seleccion:
        for p in user['ultimos_productos']:
            if p['codigo'].lower() in msg.lower():
                seleccion = p
                break
    
    if not seleccion:
        # No entendió, re-mostrar
        return flujo_presentar_productos(user, user['ultimos_productos'], es_follow_up=True)
    
    # Agregar
    cantidad = calcular_cantidad(user['m2'], seleccion['m2_caja'])
    item = {
        'codigo': seleccion['codigo'],
        'descripcion': seleccion['descripcion'],
        'precio': seleccion['precio'],
        'cantidad': cantidad,
        'subtotal': (seleccion['precio'] or 0) * cantidad
    }
    user['carrito'].append(item)
    
    precio_str = f"${seleccion['precio']:.2f}" if seleccion['precio'] else "precio por confirmar"
    subtotal_str = f"${item['subtotal']:.2f}" if seleccion['precio'] else "por confirmar"
    
    prompt = f"""Confirma al cliente: agregaste {cantidad} de {seleccion['descripcion']}.
    Subtotal: {subtotal_str}.
    Pregunta si quiere algo más o ver el total. 2 líneas, sin listas."""
    
    return hablar(prompt, max_tokens=80)

def flujo_total(user):
    """Cierra cotización"""
    if not user['carrito']:
        return "Tu cotización está vacía. ¿Qué necesitas? Equipo VAMA."
    
    lineas = []
    total = 0
    for item in user['carrito']:
        if item['precio']:
            lineas.append(f"- {item['descripcion']}: {item['cantidad']} x ${item['precio']:.2f} = ${item['subtotal']:.2f}")
            total += item['subtotal']
        else:
            lineas.append(f"- {item['descripcion']}: {item['cantidad']} (precio por confirmar)")
    
    resumen = "\n".join(lineas)
    total_str = f"${total:.2f}" if total > 0 else "Por confirmar"
    
    prompt = f"""Presenta resumen final al cliente. Total: {total_str}.
    Pregunta si confirma disponibilidad. Profesional, 2 líneas."""
    
    texto = hablar(prompt, max_tokens=80)
    return f"{texto}\n\n{resumen}\n\n**TOTAL: {total_str}**\n\n¿Confirmamos? Equipo VAMA."

# ROUTER PRINCIPAL
def procesar(user_id, nombre, mensaje):
    user = memoria.get(user_id)
    if nombre and not user['nombre']:
        user['nombre'] = nombre
    
    msg = mensaje.strip()
    msg_low = msg.lower()
    user['contador'] += 1
    
    # Reset explícito
    if any(w in msg_low for w in ['nueva cotizacion', 'reiniciar', 'empezar de nuevo', 'borrar todo']):
        user['carrito'] = []
        user['ultimos_productos'] = []
        user['m2'] = 0
        return "Listo, nueva cotización. ¿Qué material necesitas? Equipo VAMA."
    
    # Detectar intención de total
    if any(w in msg_low for w in ['total', 'eso es todo', 'terminamos', 'cuanto es', 'cerrar']):
        return flujo_total(user)
    
    # Detectar si está seleccionando de una lista previa
    if user['ultimos_productos'] and (
        re.search(r'\b[123]\b', msg) or 
        any(p['codigo'].lower() in msg_low for p in user['ultimos_productos'])
    ):
        return flujo_agregar_producto(user, msg)
    
    # Detectar "mostrar de nuevo" / "ver otra vez"
    if user['ultimos_productos'] and any(w in msg_low for w in ['muestra', 'ver', 'de nuevo', 'otra vez', 'cuales']):
        return flujo_presentar_productos(user, user['ultimos_productos'], es_follow_up=True)
    
    # Buscar nuevo producto
    ints = detectar_intenciones(msg)
    color = detectar_color(msg)
    m2 = detectar_m2(msg)
    
    if m2:
        user['m2'] = m2
    
    productos = buscar(msg, intenciones=ints, color=color, top_k=3)
    
    if not productos:
        return f"No encontré {', '.join(ints)} en color {color or 'cualquiera'}. ¿Otras características? Equipo VAMA."
    
    return flujo_presentar_productos(user, productos, es_follow_up=user['contador'] > 1)

# API
app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json() or {}
    tel = str(data.get('telefono', '0'))
    nom = data.get('nombre', 'Cliente')
    msg = data.get('mensaje', '')
    
    print(f"\n[{datetime.now():%H:%M}] {nom}: {msg[:50]}")
    
    resp = procesar(tel, nom, msg)
    memoria.guardar()
    
    print(f"[R] {resp[:80]}...")
    return jsonify({'respuesta': resp})

if __name__ == '__main__':
    print(f"🚀 VAMA 3.0 | 12 colecciones | Modelo: {MODELO}")
    print(f"Colecciones: {list(COLECCIONES.keys())}")
    app.run(host='0.0.0.0', port=5000, debug=False)