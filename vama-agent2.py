#!/usr/bin/env python3
"""
VAMA Agent - PARCHE MÍNIMO
Solo modifica la función procesar() para agregar productos automáticamente
cuando el usuario los menciona junto con "quiero pagar"
"""

import os
import sys
import json
import re
import math
import glob
from datetime import datetime
from flask import Flask, request, send_file, jsonify
import chromadb
from chromadb.utils import embedding_functions
from fpdf import FPDF
import ollama
import requests

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
CHROMA_PATH = "./chroma_db_v3"
MEMORIA_FILE = 'memoria_v2.json'
OLLAMA_HOST = 'http://127.0.0.1:11434'
MODELO = "qwen2.5:14b"

print("=" * 60)
print("🚀 VAMA Agent - PARCHE MÍNIMO")
print("=" * 60)

# ============================================================================
# INICIALIZAR OLLAMA
# ============================================================================
try:
    ollama_client = ollama.Client(host=OLLAMA_HOST)
    r = requests.get(f'{OLLAMA_HOST}/api/tags', timeout=5)
    if r.status_code == 200:
        print(f"✅ Ollama conectado, modelo: {MODELO}")
    else:
        print(f"⚠️ Ollama responde pero no se pudo verificar el modelo")
except Exception as e:
    print(f"⚠️ Error conectando a Ollama: {e}")
    print("   El sistema funcionará en modo texto (sin LLM)")
    ollama_client = None

# ============================================================================
# INICIALIZAR CHROMADB
# ============================================================================
print("\n📦 Cargando ChromaDB...")
try:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-m3", device="cpu"
    )
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)

    colecciones = {}
    colecciones_disponibles = [c.name for c in chroma.list_collections()]
    print(f"   Colecciones disponibles: {colecciones_disponibles}")

    for nombre in colecciones_disponibles:
        try:
            colecciones[nombre] = chroma.get_collection(nombre, embedding_function=embed_fn)
            print(f"   ✅ {nombre}")
        except Exception as e:
            print(f"   ⚠️ Error en {nombre}: {e}")

    print(f"\n✅ {len(colecciones)} colecciones cargadas")

except Exception as e:
    print(f"\n❌ ERROR CRÍTICO: {e}")
    sys.exit(1)

app = Flask(__name__)

# ============================================================================
# MEMORIA
# ============================================================================
class Memoria:
    def __init__(self):
        self.datos = {}
        if os.path.exists(MEMORIA_FILE):
            try:
                with open(MEMORIA_FILE, 'r') as f:
                    self.datos = json.load(f)
                print(f"💾 Memoria cargada: {len(self.datos)} usuarios")
            except:
                pass

    def get(self, uid):
        if uid not in self.datos:
            self.datos[uid] = {
                'nombre': '',
                'carrito': [],
                'ultimos_productos': [],
                'm2': 0,
                'historial': [],
                'ultima_busqueda': '',
                'estado': 'inicio'
            }
        return self.datos[uid]

    def save(self):
        try:
            with open(MEMORIA_FILE, 'w') as f:
                json.dump(self.datos, f, indent=2)
        except:
            pass

memoria = Memoria()

# ============================================================================
# MAPEO DE INTENCIONES (PRODUCTOS)
# ============================================================================
MAP_COLECCIONES = {
    'piso': ['nacionales', 'importados'],
    'azulejo': ['nacionales', 'importados'],
    'porcelanato': ['nacionales', 'importados'],
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
    'pegamento': ['polvos'],
    'pega': ['polvos'],
    'adhesivo': ['polvos'],
    'cemix': ['polvos']
}

# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================
def detectar_intenciones_producto(msg):
    msg = msg.lower()
    intenciones = []
    
    # Prioridad para frases compuestas que no deben caer en 'piso'
    if 'piso sobre piso' in msg or 'piso sobre piso' in msg:
        return ['pegamento']
    
    for palabra, cols in MAP_COLECCIONES.items():
        if palabra in msg:
            intenciones.append(palabra)
    
    # Si no hay intención clara y el mensaje tiene "piso" pero también "pegamento" o "adhesivo", priorizar pegamento
    if 'piso' in intenciones and any(p in msg for p in ['pegamento', 'pega', 'adhesivo', 'cemento']):
        intenciones = [p for p in intenciones if p != 'piso']
        intenciones.append('pegamento')
    
    return intenciones if intenciones else ['piso']

def detectar_color(msg):
    colores = ['blanco', 'gris', 'negro', 'beige', 'marmol', 'azul', 'verde', 'hueso', 'bone', 'arena']
    for c in colores:
        if c in msg.lower():
            return c
    return None

def detectar_m2(msg):
    m = re.search(r'(\d+)\s*(?:m2|m²|metros)', msg.lower())
    return int(m.group(1)) if m else None

def extraer_precio(meta):
    for k in ['precio_caja', 'precio_unitario', 'precio_m2', 'precio_final', 'precio']:
        try:
            v = float(meta.get(k, 0))
            if v > 10:
                return v
        except:
            continue
    return None

def buscar_productos(query, top_k=3):
    intenciones = detectar_intenciones_producto(query)
    color = detectar_color(query)

    colecciones_a_buscar = set()
    for intencion in intenciones:
        colecciones_a_buscar.update(MAP_COLECCIONES.get(intencion, ['nacionales', 'importados']))

    resultados = []
    for nombre_col in colecciones_a_buscar:
        if nombre_col not in colecciones:
            continue

        try:
            col = colecciones[nombre_col]
            res = col.query(query_texts=[query], n_results=top_k * 2)

            for meta in res['metadatas'][0]:
                if not meta:
                    continue

                if color and color not in meta.get('color', '').lower() and color not in meta.get('descripcion', '').lower():
                    continue

                precio = extraer_precio(meta)

                resultados.append({
                    'codigo': meta.get('codigo', 'N/A'),
                    'descripcion': meta.get('descripcion', ''),
                    'formato': meta.get('formato', ''),
                    'color': meta.get('color', ''),
                    'm2_caja': float(meta.get('metraje_caja', 1) or 1),
                    'precio': precio,
                    'coleccion': nombre_col
                })
        except Exception as e:
            print(f"Error en {nombre_col}: {e}")

    vistos = set()
    unicos = []
    for p in resultados:
        if p['codigo'] not in vistos:
            unicos.append(p)
            vistos.add(p['codigo'])

    return unicos[:top_k]

def buscar_sucursales(query, top_k=3):
    """Busca sucursales en ChromaDB"""
    if "sucursales" not in colecciones:
        return None

    try:
        col = colecciones["sucursales"]
        res = col.query(query_texts=[query], n_results=top_k)

        resultados = []
        for i, meta in enumerate(res['metadatas'][0]):
            if meta:
                resultados.append({
                    'nombre': meta.get('nombre', ''),
                    'direccion': meta.get('direccion', ''),
                    'telefono': meta.get('telefono', ''),
                    'horario': meta.get('horario', ''),
                    'ciudad': meta.get('ciudad', ''),
                    'zona': meta.get('zona', ''),
                    'relevancia': res['distances'][0][i] if res.get('distances') else 0
                })
        return resultados
    except Exception as e:
        print(f"Error buscando sucursales: {e}")
        return None

def formatear_productos_para_llm(productos, m2=None):
    if not productos:
        return "No encontré productos con esas características."

    texto = "Aquí están las opciones que encontré:\n"
    for i, p in enumerate(productos[:3], 1):
        texto += f"\nOpción {i}:\n"
        texto += f"- Producto: {p['descripcion']}\n"
        if p.get('formato'):
            texto += f"- Formato: {p['formato']}\n"
        if p.get('color'):
            texto += f"- Color: {p['color']}\n"
        if p.get('m2_caja'):
            texto += f"- Rendimiento: {p['m2_caja']} m² por caja\n"
        if p['precio']:
            texto += f"- Precio: ${p['precio']:.2f}\n"
        else:
            texto += "- Precio: por confirmar\n"

    if m2:
        texto += f"\nPara tus {m2} m², ¿cuál de estas opciones te interesa? Puedes decirme el número o el nombre."
    else:
        texto += "\n¿Cuál de estas opciones te interesa? Dime el número o el nombre."

    return texto

def calcular_cantidad(m2_proyecto, m2_caja):
    if m2_proyecto and m2_caja and m2_caja > 0:
        return math.ceil(m2_proyecto / m2_caja)
    return 1

def generar_pdf(user_id, user):
    """Genera PDF de cotización y devuelve la ruta del archivo"""
    try:
        pdf = FPDF()
        pdf.add_page()

        # Encabezado
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'COTIZACIÓN VAMA', 0, 1, 'C')
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 10, f'Fecha: {datetime.now().strftime("%d/%m/%Y %H:%M")}', 0, 1)
        pdf.cell(0, 10, f'Cliente: {user.get("nombre", "Cliente")}', 0, 1)
        pdf.cell(0, 10, f'Teléfono: {user_id}', 0, 1)
        pdf.ln(5)

        # Tabla de productos
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(80, 8, 'Producto', 1)
        pdf.cell(25, 8, 'Cantidad', 1)
        pdf.cell(35, 8, 'P.Unitario', 1)
        pdf.cell(35, 8, 'Subtotal', 1)
        pdf.ln()

        total = 0
        pdf.set_font('Arial', '', 9)

        for item in user.get('carrito', []):
            desc = item['descripcion'][:35] + '...' if len(item['descripcion']) > 35 else item['descripcion']
            pdf.cell(80, 8, desc, 1)
            pdf.cell(25, 8, str(item.get('cantidad', 1)), 1, 0, 'C')

            if item.get('precio_unitario'):
                pdf.cell(35, 8, f"${item['precio_unitario']:.2f}", 1, 0, 'R')
                subtotal = item['precio_unitario'] * item.get('cantidad', 1)
                pdf.cell(35, 8, f"${subtotal:.2f}", 1, 0, 'R')
                total += subtotal
            else:
                pdf.cell(35, 8, "P/C", 1, 0, 'C')
                pdf.cell(35, 8, "P/C", 1, 0, 'C')
            pdf.ln()

        # Total
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, f'TOTAL: ${total:.2f}', 0, 1, 'R')

        # Pie de página
        pdf.ln(10)
        pdf.set_font('Arial', '', 8)
        pdf.cell(0, 5, 'VAMA - Materiales y Acabados', 0, 1, 'C')
        pdf.cell(0, 5, 'Esta cotización tiene validez de 7 días', 0, 1, 'C')

        # Guardar archivo
        os.makedirs('cotizaciones', exist_ok=True)
        filename = f"cotizaciones/cot_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(filename)
        return filename

    except Exception as e:
        print(f"Error generando PDF: {e}")
        return None

def generar_respuesta_llm(user, mensaje_usuario, contexto_adicional=""):
    if not ollama_client:
        return None

    # ===== DETECCIÓN DE CONSULTA DE SUCURSALES =====
    palabras_ubicacion = ['donde', 'ubicacion', 'ubicados', 'direccion', 'sucursal', 'tienda', 'local', 'encuentro', 'están']
    es_consulta_ubicacion = any(p in mensaje_usuario.lower() for p in palabras_ubicacion)

    sucursales_info = ""
    if es_consulta_ubicacion and "sucursales" in colecciones:
        try:
            col = colecciones["sucursales"]
            res = col.query(query_texts=[mensaje_usuario], n_results=3)
            if res['metadatas'][0]:
                sucursales_info = "\nSUCURSALES ENCONTRADAS:\n"
                for meta in res['metadatas'][0]:
                    if meta:
                        sucursales_info += f"- {meta.get('nombre', 'Sucursal')}\n"
                        sucursales_info += f"  📍 {meta.get('direccion', '')}\n"
                        sucursales_info += f"  📞 {meta.get('telefono', '')}\n"
                        sucursales_info += f"  🕒 {meta.get('horario', '')}\n\n"
        except Exception as e:
            print(f"Error buscando sucursales: {e}")
    # ===============================================

    historial = ""
    for h in user.get('historial', [])[-6:]:
        quien = "Cliente" if h['role'] == 'user' else "Vendedor"
        historial += f"{quien}: {h['content']}\n"

    carrito_str = ""
    if user.get('carrito'):
        carrito_str = "Carrito actual:\n"
        for item in user['carrito']:
            precio = f"${item['subtotal']:.2f}" if item.get('subtotal') else "pendiente"
            carrito_str += f"- {item['descripcion']}: {item['cantidad']} unidades - {precio}\n"

    nombre = user.get('nombre', 'Cliente')

    prompt = f"""Eres VAMA, un vendedor amable y profesional de una tienda de materiales para construcción.

HISTORIAL RECIENTE:
{historial}

CONTEXTO ACTUAL:
- Cliente: {nombre}
- Metros cuadrados: {user.get('m2', 'no especificado')}
{carrito_str}
{contexto_adicional}
{sucursales_info}

MENSAJE DEL CLIENTE: "{mensaje_usuario}"

INSTRUCCIONES:
1. Responde de forma natural y cálida, como un vendedor real.
2. Usa el nombre del cliente cuando sea natural (no en cada frase).
3. Si el cliente saluda, saluda de vuelta y pregunta qué necesita.
4. Si ya mostraste productos y el cliente eligió uno, confirma la selección.
5. Si el cliente pide el total, da el total y pregunta si confirma.
6. Si el cliente se despide, despídete amablemente.
7. SI EL CLIENTE PREGUNTA POR UBICACIÓN O SUCURSALES, usa la información de SUCURSALES ENCONTRADAS para dar una respuesta precisa y amable.
REGLAS DE INTERPRETACIÓN:
- Si el cliente menciona SOLO un número (ej. "2", "tres"), puede ser cantidad de cajas o selección de opción.
- Si el cliente menciona una CIUDAD, pregunta si es para envío o si necesita sucursal.
- Si el cliente dice frases como "el más común", "el normal", "el que siempre usan", refiere al producto que estaban viendo.
- Si el cliente menciona "quiero pagar" o "quiero mi cotización" junto con productos, asume que quiere agregar esos productos al carrito y luego mostrar el total.
- Si el cleinte pregunta por el sitio web da la URL: https://vama.com.mx
- Si el cliente menciona "ese me interesa" , "ese me gusta", "ese quiero", asume que se refiere al último producto mostrado o a la opción que eligió.
- Si el cliente menciona "me interesa" , "me gusta", "quiero", no lo agregues al carrito directamente, primero confirma cuál producto se refiere.

RESPUESTA (solo el mensaje para el cliente):"""

    try:
        r = ollama_client.generate(
            model=MODELO,
            prompt=prompt,
            options={
                "temperature": 0.7,
                "num_predict": 300
            }
        )
        return r['response'].strip()
    except Exception as e:
        print(f"Error LLM: {e}")
        return None

# ============================================================================
# NUEVA FUNCIÓN: EXTRAER Y AGREGAR PRODUCTOS DEL MENSAJE
# ============================================================================
def extraer_y_agregar_productos(user, mensaje):
    """
    Extrae productos mencionados en el mensaje y los agrega al carrito.
    Retorna lista de productos agregados.
    """
    msg_lower = mensaje.lower()

    # Limpiar frases que no son productos
    for frase in ['quiero pagar', 'quiero comprar', 'dame mi cotizacion', 'es todo', 'si esos']:
        msg_lower = msg_lower.replace(frase, ' ')

    # Separar por conectores
    partes = re.split(r'\s+y\s+|,\s*|\s+\+\s+', msg_lower)
    partes = [p.strip() for p in partes if p.strip() and len(p.strip()) > 3]

    productos_agregados = []

    for parte in partes:
        # Detectar si esta parte menciona algún tipo de producto
        tiene_producto = any(palabra in parte for palabra in MAP_COLECCIONES.keys())
        if not tiene_producto:
            continue

        # Buscar el producto
        productos = buscar_productos(parte, top_k=1)
        if productos:
            producto = productos[0]

            # Verificar que no esté ya en el carrito
            codigos_carrito = {item['codigo'] for item in user.get('carrito', [])}
            if producto['codigo'] in codigos_carrito:
                continue

            cantidad = calcular_cantidad(user.get('m2', 0), producto['m2_caja'])
            item = {
                'codigo': producto['codigo'],
                'descripcion': producto['descripcion'],
                'cantidad': cantidad,
                'precio_unitario': producto['precio'],
                'subtotal': (producto['precio'] or 0) * cantidad
            }
            user['carrito'].append(item)
            productos_agregados.append(item)
            print(f"[AUTO-ADD] Agregado: {producto['descripcion']}")

    return productos_agregados


# ============================================================================
# FLUJOS PRINCIPALES
# ============================================================================
def flujo_buscar(user, msg):
    user['ultima_busqueda'] = msg

    m2 = detectar_m2(msg)
    if m2:
        user['m2'] = m2

    productos = buscar_productos(msg)

    if not productos:
        return "Lo siento, no encontré productos con esas características. ¿Podrías darme más detalles? (color, medida, tipo)"

    user['ultimos_productos'] = productos

    contexto = formatear_productos_para_llm(productos, user.get('m2'))
    respuesta = generar_respuesta_llm(user, msg, contexto)

    if not respuesta:
        respuesta = f"Encontré estas opciones para ti:\n\n"
        for i, p in enumerate(productos[:3], 1):
            precio = f"${p['precio']:.2f}" if p['precio'] else "precio por confirmar"
            respuesta += f"{i}. {p['descripcion']} - {precio}\n"
        if user.get('m2'):
            respuesta += f"\nPara tus {user['m2']}m², ¿cuál te interesa?"
        else:
            respuesta += "\n¿Cuál te interesa?"

    return respuesta

def flujo_seleccionar(user, msg):
    if not user.get('ultimos_productos'):
        return "Primero dime qué estás buscando para mostrarte opciones."

    num_match = re.search(r'[123]', msg)
    if not num_match:
        contexto = "El cliente no especificó un número claro. Debe elegir 1, 2 o 3."
        respuesta = generar_respuesta_llm(user, msg, contexto)
        if respuesta:
            return respuesta
        return "¿Podrías decirme qué número de producto te interesa? (1, 2 o 3)"

    idx = int(num_match.group()) - 1
    if idx >= len(user['ultimos_productos']):
        return "Esa opción no está disponible. Elige 1, 2 o 3."

    producto_seleccionado = user['ultimos_productos'][idx]
    cantidad = calcular_cantidad(user.get('m2', 0), producto_seleccionado['m2_caja'])

    item = {
        'codigo': producto_seleccionado['codigo'],
        'descripcion': producto_seleccionado['descripcion'],
        'cantidad': cantidad,
        'precio_unitario': producto_seleccionado['precio'],
        'subtotal': (producto_seleccionado['precio'] or 0) * cantidad
    }
    user['carrito'].append(item)
    user['ultimos_productos'] = []

    total = sum(i['subtotal'] for i in user['carrito'])
    contexto = f"El cliente eligió la opción {idx+1}: {producto_seleccionado['descripcion']}. "
    contexto += f"Para {user.get('m2', 'sus')} m² necesita {cantidad} cajas. "
    contexto += f"Subtotal: ${item['subtotal']:.2f}. Total acumulado: ${total:.2f}. "
    contexto += "Confirma si está correcto."

    respuesta = generar_respuesta_llm(user, msg, contexto)
    if respuesta:
        return respuesta

    return (f"¡Listo! Agregué {cantidad} unidad(es) de {producto_seleccionado['descripcion']}.\n"
            f"Subtotal: ${item['subtotal']:.2f}\n"
            f"Total acumulado: ${total:.2f}\n\n"
            f"¿Necesitas algo más?")

def flujo_total(user, user_id=None):
    if not user['carrito']:
        return "Aún no tienes productos en tu cotización. ¿Qué material necesitas?"

    total = sum(i.get('subtotal', 0) for i in user['carrito'])

    # Generar PDF
    pdf_file = None
    if total > 0 and user_id:
        pdf_file = generar_pdf(user_id, user)

    # Construir respuesta
    lineas = ["**Resumen de tu cotización:**\n"]
    for item in user['carrito']:
        if item.get('precio_unitario'):
            subtotal = item['precio_unitario'] * item.get('cantidad', 1)
            lineas.append(f"• {item['descripcion']}: {item['cantidad']} x ${item['precio_unitario']:.2f} = ${subtotal:.2f}")
        else:
            lineas.append(f"• {item['descripcion']}: {item['cantidad']} unidades (precio por confirmar)")

    lineas.append(f"\n**TOTAL: ${total:.2f}**\n")

    if pdf_file:
        base_url = "http://agente3.aegis-oceanit.work"
        pdf_url = f"{base_url}/pdf/{user_id}"
        lineas.append(f"📄 Descarga tu cotización aquí: {pdf_url}\n")

    lineas.append("¿Te parece bien? ¿Confirmamos disponibilidad?")

        # Vaciar carrito después de mostrar la cotización (como pagar en caja)
    user['carrito'] = []
    user['ultimos_productos'] = []
    user['m2'] = 0

    return "\n".join(lineas)

def flujo_eliminar_producto(user, msg):
    if not user.get('carrito'):
        return "Tu carrito está vacío, no hay nada que eliminar."

    msg_low = msg.lower()

    # Eliminar por número (si dice "quita el 1", "elimina el 2")
    num_match = re.search(r'(?:quita\s+el\s+)?([123])', msg_low)
    if num_match:
        idx = int(num_match.group(1)) - 1
        if idx < len(user['carrito']):
            eliminado = user['carrito'].pop(idx)
            total = sum(i.get('subtotal', 0) for i in user['carrito'])
            return f"✅ Eliminado {eliminado['descripcion']} de tu carrito.\nTotal actual: ${total:.2f}\n\n¿Algo más?"

    # Eliminar por nombre o código
    for idx, item in enumerate(user['carrito']):
        desc_lower = item['descripcion'].lower()
        cod_lower = item['codigo'].lower()
        if desc_lower in msg_low or cod_lower in msg_low:
            eliminado = user['carrito'].pop(idx)
            total = sum(i.get('subtotal', 0) for i in user['carrito'])
            return f"✅ Eliminado {eliminado['descripcion']} de tu carrito.\nTotal actual: ${total:.2f}\n\n¿Algo más?"

    # Eliminar el último producto
    if any(p in msg_low for p in ['último', 'ultimo', 'ese', 'eso', 'el último', 'el que agregué']):
        eliminado = user['carrito'].pop()
        total = sum(i.get('subtotal', 0) for i in user['carrito'])
        return f"✅ Eliminado {eliminado['descripcion']} de tu carrito.\nTotal actual: ${total:.2f}\n\n¿Necesitas algo más?"

    # Mostrar carrito numerado
    carrito_str = "\n".join([f"{i+1}. {item['descripcion']} - ${item.get('subtotal', 0):.2f}" for i, item in enumerate(user['carrito'])])
    return f"No entendí qué producto quieres eliminar. Estos son los productos en tu carrito:\n{carrito_str}\n\nDime el **número** o el **nombre** del que quieras quitar."

def flujo_mas_opciones(user):
    if not user.get('ultima_busqueda'):
        return "Dime qué estás buscando y te muestro opciones."

    productos = buscar_productos(user['ultima_busqueda'], top_k=6)

    if user.get('ultimos_productos'):
        codigos_vistos = {p['codigo'] for p in user['ultimos_productos']}
        productos = [p for p in productos if p['codigo'] not in codigos_vistos]

    if not productos:
        return "No encontré más opciones diferentes. ¿Quieres probar con otra descripción?"

    user['ultimos_productos'] = productos[:3]
    contexto = formatear_productos_para_llm(productos[:3], user.get('m2'))

    respuesta = generar_respuesta_llm(user, "Muéstrame más opciones", contexto)
    if respuesta:
        return respuesta

    return formatear_productos_para_llm(productos[:3], user.get('m2'))

def flujo_sucursales(user, msg):
    """Busca y muestra sucursales usando ChromaDB"""
    resultados = buscar_sucursales(msg)

    if not resultados:
        texto_fallback = """
Nuestras sucursales:

CULIACÁN - TRES RIOS
Blvd. Enrique Sánchez Alonso #1515, Desarrollo Urbano Tres Ríos
Tel. (667) 752 20 78

CULIACÁN - IGNACIO RAMÍREZ
Ignacio Ramírez #981 pte, Jorge Almada
Tel. (667) 752 02 61

LOS MOCHIS - BIENESTAR
Bienestar #633, Col. Bienestar
Tel. (668) 812 58 13

LOS MOCHIS - INDEPENDENCIA
Av. Independencia #2049 pte, Jardines del Country
Tel. (668) 176 96 13

Horario: Lunes a Viernes 8am-8pm, Sábados 9am-6pm
"""
        respuesta = generar_respuesta_llm(user, msg, f"El cliente pregunta por ubicación. Dale esta información:\n{texto_fallback}")
        if respuesta:
            return respuesta
        return f"¡Claro! Aquí están nuestras sucursales:\n{texto_fallback}"

    texto = "Claro, aquí están nuestras sucursales:\n\n"
    for i, s in enumerate(resultados[:3], 1):
        texto += f"{i}. {s['nombre']}\n"
        texto += f"   📍 {s['direccion']}\n"
        texto += f"   📞 {s['telefono']}\n"
        texto += f"   🕒 {s['horario']}\n\n"

    respuesta = generar_respuesta_llm(user, msg, texto)
    if respuesta:
        return respuesta

    return texto

# ============================================================================
# DETECTOR DE INTENCIONES
# ============================================================================
def detectar_intencion(msg, user):
    msg_low = msg.lower().strip()

    if any(p in msg_low for p in ['ubicacion', 'ubicados', 'direccion', 'sucursal', 'donde estan', 'tienda fisica', 'como llegar']):
        return 'SUCURSALES'

    if not user.get('historial') and any(p in msg_low for p in ['hola', 'buenas', 'buen dia', 'saludos']):
        return 'SALUDAR'

    if any(p in msg_low for p in ['gracias', 'adios', 'bye', 'hasta luego', 'eso es todo']):
        return 'DESPEDIR'

    if any(p in msg_low for p in ['total', 'cuanto es', 'cuenta', 'pagar', 'pdf', 'cotizacion']):
        return 'CHECKOUT'

    if any(p in msg_low for p in ['mas opciones', 'otras opciones', 'ver más', 'otros']):
        return 'MAS_OPCIONES'

    if any(p in msg_low for p in ['quitar', 'eliminar', 'quita', 'borrar', 'saca', 'no quiero ese', 'cancela ese']):
        return 'ELIMINAR_PRODUCTO'

    if any(p in msg_low for p in ['nueva cotizacion', 'empezar de nuevo', 'borrar', 'limpiar']):
        return 'REINICIAR'

    if user.get('ultimos_productos'):
        if re.match(r'^\s*[123]\s*$', msg_low) or any(f"el {i}" in msg_low for i in ['1','2','3']):
            return 'SELECCIONAR'

    return 'BUSCAR'

# ============================================================================
# PROCESADOR PRINCIPAL (CORREGIDO)
# ============================================================================
def procesar(user_id, nombre, mensaje):
    user = memoria.get(user_id)

    if 'historial' not in user:
        user['historial'] = []

    if nombre and not user['nombre']:
        user['nombre'] = nombre

    # === SOLUCIÓN PRINCIPAL: DETECTAR PRODUCTOS + PAGAR ===
    msg_lower = mensaje.lower()
    quiere_pagar = any(p in msg_lower for p in ['pagar', 'pago', 'cotizacion', 'cotización', 'total', 'cuanto es'])
    menciona_productos = any(p in msg_lower for p in MAP_COLECCIONES.keys())

    if quiere_pagar and menciona_productos:
        # El usuario menciona productos Y quiere pagar
        # Primero extraer y agregar los productos
        productos_agregados = extraer_y_agregar_productos(user, mensaje)

        if productos_agregados:
            print(f"🛒 Agregados {len(productos_agregados)} productos automáticamente")

        # Ahora mostrar el checkout
        if user.get('carrito') and len(user['carrito']) > 0:
            respuesta = flujo_total(user, user_id)
            user['historial'].append({'role': 'user', 'content': mensaje, 'time': datetime.now().isoformat()})
            user['historial'].append({'role': 'assistant', 'content': respuesta, 'time': datetime.now().isoformat()})
            memoria.save()
            return respuesta

    # === SOLUCIÓN SECUNDARIA: SOLO PAGAR (sin mencionar productos) ===
    if quiere_pagar and not menciona_productos:
        # El usuario solo quiere pagar, pero el carrito puede estar vacío
        # Buscar en el historial si mencionó productos antes
        if not user.get('carrito'):
            for h in reversed(user.get('historial', [])[-10:]):
                if h['role'] == 'user':
                    productos_hist = extraer_y_agregar_productos(user, h['content'])
                    if productos_hist:
                        print(f"🛒 Recuperados {len(productos_hist)} productos del historial")
                        break

        if user.get('carrito') and len(user['carrito']) > 0:
            respuesta = flujo_total(user, user_id)
            user['historial'].append({'role': 'user', 'content': mensaje, 'time': datetime.now().isoformat()})
            user['historial'].append({'role': 'assistant', 'content': respuesta, 'time': datetime.now().isoformat()})
            memoria.save()
            return respuesta
    # =========================

    user['historial'].append({
        'role': 'user',
        'content': mensaje,
        'time': datetime.now().isoformat()
    })
    if len(user['historial']) > 10:
        user['historial'] = user['historial'][-10:]

    intencion = detectar_intencion(mensaje, user)
    print(f"[{user_id}] Intención: {intencion}")

    if intencion == 'SUCURSALES':
        respuesta = flujo_sucursales(user, mensaje)

    elif intencion == 'SALUDAR':
        respuesta = generar_respuesta_llm(user, mensaje, "El cliente está saludando. Salúdalo amablemente y pregúntale qué necesita.")
        if not respuesta:
            respuesta = f"¡Hola {user['nombre']}! ¿En qué puedo ayudarte hoy? Estamos buscando materiales para construcción."

    elif intencion == 'BUSCAR':
        respuesta = flujo_buscar(user, mensaje)

    elif intencion == 'SELECCIONAR':
        respuesta = flujo_seleccionar(user, mensaje)

    elif intencion == 'CHECKOUT':
        respuesta = flujo_total(user, user_id)

    elif intencion == 'MAS_OPCIONES':
        respuesta = flujo_mas_opciones(user)

    elif intencion == 'REINICIAR':
        user['carrito'] = []
        user['ultimos_productos'] = []
        user['m2'] = 0
        respuesta = generar_respuesta_llm(user, mensaje, "El cliente quiere empezar una cotización nueva.")
        if not respuesta:
            respuesta = "¡Listo! Empezamos una cotización nueva. ¿Qué material necesitas?"

    elif intencion == 'ELIMINAR_PRODUCTO':
        respuesta = flujo_eliminar_producto(user, mensaje)

    elif intencion == 'DESPEDIR':
        respuesta = generar_respuesta_llm(user, mensaje, "El cliente se está despidiendo. Despídete amablemente.")
        if not respuesta:
            if user['carrito']:
                total = sum(i['subtotal'] for i in user['carrito'])
                respuesta = f"¡Perfecto! Tu cotización quedó en ${total:.2f}. Estoy aquí cuando me necesites. ¡Que tengas excelente día!"
            else:
                respuesta = "¡Con gusto! Estoy aquí cuando me necesites. ¡Que tengas excelente día!"
        user['carrito'] = []
        user['ultimos_productos'] = []
        user['m2'] = 0

    else:
        respuesta = flujo_buscar(user, mensaje)

    user['historial'].append({
        'role': 'assistant',
        'content': respuesta,
        'time': datetime.now().isoformat()
    })

    memoria.save()
    log_conversacion(user_id, user.get('nombre', nombre), mensaje, respuesta, user)
    
    return respuesta

# ============================================================================
# LOG DE CONVERSACIONES
# ============================================================================
def log_conversacion(user_id, nombre, mensaje_usuario, respuesta, user):
    with open("conversaciones_completas.jsonl", "a") as f:
        registro = {
            "timestamp": datetime.now().isoformat(),
            "telefono": user_id,
            "nombre": nombre,
            "mensaje_usuario": mensaje_usuario,
            "respuesta_bot": respuesta,
            "estado": user.get("estado"),  # aunque no lo uses ahora, por si acaso
            "carrito": user.get("carrito"),
            "ultimos_productos": [p.get("codigo") for p in user.get("ultimos_productos", [])],
            "m2": user.get("m2"),
        }
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")

# ============================================================================
# API
# ============================================================================
@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json() or {}
    tel = str(data.get('telefono', '0'))
    nom = data.get('nombre', 'Cliente')
    msg = data.get('mensaje', '')

    # FILTRO ANTI-DEMONIO
    if not msg or len(msg.strip()) < 2:
        return jsonify({'respuesta': ''})

    if len(tel) < 10 or tel == '0':
        print(f"⚠️ Petición ignorada - teléfono inválido: {tel}")
        return jsonify({'respuesta': ''})

    print(f"\n[{datetime.now():%H:%M}] {nom} ({tel}): {msg[:50]}")
    resp = procesar(tel, nom, msg)
    print(f"[{datetime.now():%H:%M}] Bot: {resp[:80]}...")

    return jsonify({'respuesta': resp})

@app.route('/pdf/<user_id>')
def descargar_pdf(user_id):
    """Endpoint para descargar el PDF más reciente del usuario"""
    try:
        pattern = f"cotizaciones/cot_{user_id}_*.pdf"
        pdfs = glob.glob(pattern)

        if pdfs:
            pdfs.sort(key=os.path.getmtime, reverse=True)
            return send_file(pdfs[0], as_attachment=True, download_name=f'cotizacion_{user_id}.pdf')
        else:
            return "No hay cotizaciones disponibles para este usuario", 404
    except Exception as e:
        return f"Error al descargar PDF: {e}", 500

@app.route('/health')
def health():
    return {
        'status': 'ok',
        'colecciones': list(colecciones.keys()),
        'usuarios': len(memoria.datos)
    }

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🟢 Servidor listo en http://0.0.0.0:5001")
    print("   Health check: http://localhost:5001/health")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=5001, debug=True)