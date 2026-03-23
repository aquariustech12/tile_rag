#!/usr/bin/env python3
"""
VAMA Agent v3.1 - Versión definitiva
Estado limpio, prioridad de categorías, checkout robusto
Puerto: 5001
"""

import os
import sys
import json
import re
import math
import glob
import sqlite3
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
DB_PATH = "vama.db"
OLLAMA_HOST = 'http://127.0.0.1:11434'
MODELO = "qwen2.5:14b"

print("=" * 60)
print("🚀 VAMA Agent v3.1 - Versión definitiva")
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
# MAPEO DE COLECCIONES (con prioridad)
# ============================================================================
MAP_COLECCIONES = {
    # Superficies
    'piso': ['nacionales', 'importados'],
    'azulejo': ['nacionales', 'importados'],
    'porcelanato': ['nacionales', 'importados'],
    'muro': ['muros', 'nacionales', 'importados'],
    # Grifería
    'grifo': ['griferia'],
    'monomando': ['griferia'],
    'llave': ['griferia'],
    'regadera': ['griferia'],
    # Lavabos y sanitarios
    'lavabo': ['lavabos'],
    'wc': ['sanitarios'],
    'inodoro': ['sanitarios'],
    'taza': ['sanitarios'],
    'sanitario': ['sanitarios'],
    # Muebles
    'mueble': ['muebles'],
    'gabinete': ['muebles'],
    # Tinacos
    'tinaco': ['tinacos'],
    'cisterna': ['tinacos'],
    # Espejos
    'espejo': ['espejos'],
    # Tarjas
    'tarja': ['tarjas'],
    'fregadero': ['tarjas'],
    # Pegamentos (polvos)
    'pegamento': ['polvos'],
    'pega': ['polvos'],
    'adhesivo': ['polvos'],
    'cemix': ['polvos'],
    'mortero': ['polvos'],
    'cemento': ['polvos'],
    'concreto': ['polvos'],
    'pegazulejo': ['polvos']
}
    
# Definir categorías que son "superficie" (usan m²)
CATEGORIAS_SUPERFICIE = ['nacionales', 'importados', 'muros']

# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================
def detectar_intenciones_producto(msg):
    msg = msg.lower()
    intenciones = []
    for palabra, cols in MAP_COLECCIONES.items():
        if palabra in msg:
            intenciones.append(palabra)
    # Si hay intenciones, eliminar 'piso' si también hay otra más específica
    if len(intenciones) > 1 and 'piso' in intenciones:
        intenciones.remove('piso')
    # Si no hay intenciones, devolver ['piso'] solo si la consulta no contiene palabras de polvos
    if not intenciones:
        # Si contiene alguna palabra típica de polvos, devolver ['polvos']
        if any(p in msg for p in ['cemento', 'mortero', 'adhesivo', 'pegamento', 'cemix']):
            return ['polvos']
        else:
            return ['piso']
    return intenciones

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

def calcular_cantidad(m2_proyecto, m2_caja):
    if m2_proyecto and m2_caja and m2_caja > 0:
        return math.ceil(m2_proyecto / m2_caja)
    return 1

def generar_pdf(user_id, user):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'COTIZACIÓN VAMA', 0, 1, 'C')
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 10, f'Fecha: {datetime.now().strftime("%d/%m/%Y %H:%M")}', 0, 1)
        pdf.cell(0, 10, f'Cliente: {user.get("nombre", "Cliente")}', 0, 1)
        pdf.cell(0, 10, f'Teléfono: {user_id}', 0, 1)
        pdf.ln(5)

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

        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, f'TOTAL: ${total:.2f}', 0, 1, 'R')

        pdf.ln(10)
        pdf.set_font('Arial', '', 8)
        pdf.cell(0, 5, 'VAMA - Materiales y Acabados', 0, 1, 'C')
        pdf.cell(0, 5, 'Esta cotización tiene validez de 7 días', 0, 1, 'C')

        os.makedirs('cotizaciones', exist_ok=True)
        filename = f"cotizaciones/cot_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(filename)
        return filename
    except Exception as e:
        print(f"Error generando PDF: {e}")
        return None

def embellecer_con_llm(texto_base, user):
    if not ollama_client:
        return None
    # Solo embellece saludos y despedidas, no listas ni resúmenes
    prompt = f"""Eres VAMA, un vendedor amable. El siguiente texto es un saludo o despedida. 
Hazlo más cálido y profesional, sin cambiar el contenido.

Texto: {texto_base}

Responde solo con el texto mejorado:"""
    try:
        r = ollama_client.generate(
            model=MODELO,
            prompt=prompt,
            options={"temperature": 0.4, "num_predict": 100}
        )
        return r['response'].strip()
    except Exception as e:
        return None

# ============================================================================
# GESTIÓN DE ESTADO CON SQLITE
# ============================================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS usuarios (
            telefono TEXT PRIMARY KEY,
            nombre TEXT,
            estado TEXT DEFAULT 'inicio',
            m2 INTEGER DEFAULT 0,
            ultima_busqueda TEXT,
            carrito TEXT,
            ultimos_productos TEXT,
            historial TEXT
        )
    ''')
    conn.commit()
    conn.close()

def get_user(telefono):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT nombre, estado, m2, ultima_busqueda, carrito, ultimos_productos, historial FROM usuarios WHERE telefono=?", (telefono,))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            'nombre': row[0] or '',
            'estado': row[1] or 'inicio',
            'm2': row[2] or 0,
            'ultima_busqueda': row[3] or '',
            'carrito': json.loads(row[4]) if row[4] else [],
            'ultimos_productos': json.loads(row[5]) if row[5] else [],
            'historial': json.loads(row[6]) if row[6] else []
        }
    else:
        return {
            'nombre': '',
            'estado': 'inicio',
            'm2': 0,
            'ultima_busqueda': '',
            'carrito': [],
            'ultimos_productos': [],
            'historial': []
        }

def save_user(telefono, data):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT OR REPLACE INTO usuarios (telefono, nombre, estado, m2, ultima_busqueda, carrito, ultimos_productos, historial)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        telefono,
        data.get('nombre', ''),
        data.get('estado', 'inicio'),
        data.get('m2', 0),
        data.get('ultima_busqueda', ''),
        json.dumps(data.get('carrito', [])),
        json.dumps(data.get('ultimos_productos', [])),
        json.dumps(data.get('historial', []))
    ))
    conn.commit()
    conn.close()

init_db()

# ============================================================================
# FUNCIONES DE NEGOCIO
# ============================================================================
def formatear_opciones(productos, m2=None):
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
        texto += f"\nPara tus {m2} m², ¿cuál te interesa? (dime el número o el nombre)"
    else:
        texto += "\n¿Cuál te interesa? (dime el número o el nombre)"
    return texto

def agregar_producto(user, producto):
    # Verificar si el producto ya está en el carrito
    for item in user['carrito']:
        if item['codigo'] == producto['codigo']:
            return f"Ya tienes {item['descripcion']} en tu carrito. ¿Quieres agregar más unidades o prefieres otro?"
    es_superficie = producto.get('coleccion') in CATEGORIAS_SUPERFICIE
    if es_superficie and user.get('m2') and producto['m2_caja']:
        cantidad = calcular_cantidad(user['m2'], producto['m2_caja'])
    else:
        cantidad = 1
    item = {
        'codigo': producto['codigo'],
        'descripcion': producto['descripcion'],
        'cantidad': cantidad,
        'precio_unitario': producto['precio'],
        'subtotal': (producto['precio'] or 0) * cantidad
    }
    user['carrito'].append(item)
    user['ultimos_productos'] = []
    user['estado'] = 'cotizando'
    total = sum(i['subtotal'] for i in user['carrito'])
    return (f"¡Listo! He añadido {cantidad} unidad(es) de {producto['descripcion']} a tu carrito.\n"
            f"Subtotal: ${item['subtotal']:.2f}\n"
            f"Total acumulado: ${total:.2f}\n\n"
            f"¿Necesitas algo más o quieres ver el total?")

def generar_resumen_cotizacion(user, telefono):
    if not user['carrito']:
        return "Aún no tienes productos en tu cotización."
    total = sum(i.get('subtotal', 0) for i in user['carrito'])
    pdf_file = generar_pdf(telefono, user) if total > 0 else None
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
        pdf_url = f"{base_url}/pdf/{telefono}"
        lineas.append(f"📄 Descarga tu cotización aquí: {pdf_url}\n")
    lineas.append("¿Te parece bien? ¿Confirmamos disponibilidad?")
    return "\n".join(lineas)

def obtener_sucursales(query):
    return """
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

def seleccionar_por_numero(user, mensaje):
    m = re.search(r'[123]', mensaje)
    if m:
        idx = int(m.group()) - 1
        if 0 <= idx < len(user['ultimos_productos']):
            return user['ultimos_productos'][idx]
    return None

def seleccionar_por_nombre(user, mensaje):
    msg_lower = mensaje.lower()
    mejores = []
    for p in user['ultimos_productos']:
        desc = p['descripcion'].lower()
        score = 0
        if msg_lower in desc:
            score += 10
        if desc in msg_lower:
            score += 5
        # Bonus por palabras numéricas (kg, m2, etc.)
        for palabra in msg_lower.split():
            if palabra in desc:
                score += 1
        # Bonus especial para "25 kg"
        if '25 kg' in msg_lower and '25 kg' in desc:
            score += 3
        if score > 0:
            mejores.append((score, p))
    if mejores:
        mejores.sort(key=lambda x: x[0], reverse=True)
        return mejores[0][1]
    return None

#=============================================================================
# PROCESADOR PRINCIPAL
# ============================================================================
def procesar(telefono, nombre, mensaje):
    user = get_user(telefono)
    if nombre and not user['nombre']:
        user['nombre'] = nombre

    user['historial'].append({'role': 'user', 'content': mensaje, 'time': datetime.now().isoformat()})
    if len(user['historial']) > 10:
        user['historial'] = user['historial'][-10:]

    msg_low = mensaje.lower()
    texto_respuesta = None

    print(f"\n[DEBUG] Mensaje: {mensaje}")
    print(f"[DEBUG] estado = {user['estado']}")
    print(f"[DEBUG] carrito = {[p['descripcion'] for p in user['carrito']]}")

    # 0. Reinicio (prioridad máxima)
    if any(p in msg_low for p in ['nueva cotizacion', 'nueva cotización', 'cotizacion nueva', 'cotización nueva', 'reiniciar', 'empezar de nuevo', 'borrar', 'limpiar']):
        print("[DEBUG] Ram: reiniciar")
        user['carrito'] = []
        user['ultimos_productos'] = []
        user['m2'] = 0
        user['estado'] = 'inicio'
        texto_respuesta = "¡Listo! Empezamos una cotización nueva. ¿Qué material necesitas?"
        save_user(telefono, user)
        user['historial'].append({'role': 'assistant', 'content': texto_respuesta, 'time': datetime.now().isoformat()})
        save_user(telefono, user)
        return texto_respuesta

    # 1. Esperando confirmación
    if user['estado'] == 'esperando_confirmacion':
        if any(p in msg_low for p in ['si', 'sí', 'ok', 'dale', 'confirmo', 'confirma', 'disponibilidad']):
            user['carrito'] = []
            user['ultimos_productos'] = []
            user['m2'] = 0
            user['estado'] = 'inicio'
            texto_respuesta = "✅ ¡Gracias por tu compra! Hemos enviado tu pedido a la sucursal de Culiacán. En breve recibirás un mensaje con los detalles de pago y entrega.\n\n¿Necesitas algo más?"
            print("[DEBUG] Ram: confirmar_compra")
        elif any(p in msg_low for p in ['no', 'cancelar', 'espera']):
            user['estado'] = 'inicio'
            texto_respuesta = "Entiendo, cancelamos la compra. ¿Quieres empezar una nueva cotización o necesitas ayuda con algo más?"
            print("[DEBUG] Ram: cancelar_compra")
        # Si no es ni sí ni no, pasamos a otras condiciones (no generamos respuesta aquí)

    if texto_respuesta is None:
        # 2. Ubicación
        if any(p in msg_low for p in ['ubicacion', 'ubicados', 'direccion', 'sucursal', 'donde estan']):
            texto_respuesta = obtener_sucursales(mensaje)
            print("[DEBUG] Ram: ubicacion")

        # 3. Frases de rechazo o finalización
        elif any(p in msg_low for p in ['no quiero eso', 'no me interesa', 'cancelar', 'olvídalo']):
            user['ultimos_productos'] = []
            texto_respuesta = "Entiendo, busquemos otra cosa. ¿Qué te gustaría en su lugar?"
            print("[DEBUG] Ram: rechazo")

        elif any(p in msg_low for p in ['es todo', 'eso es todo', 'nada más']):
            # Si el carrito no está vacío, vamos al checkout
            if user['carrito']:
                texto_respuesta = generar_resumen_cotizacion(user, telefono)
                user['estado'] = 'esperando_confirmacion'
                print("[DEBUG] Ram: es_todo -> checkout")
            else:
                texto_respuesta = "Entiendo. Si necesitas algo más, aquí estoy."
                print("[DEBUG] Ram: es_todo sin carrito")

        # 4. Saludo con carrito pendiente
        elif any(p in msg_low for p in ['hola', 'buenas', 'buen dia', 'saludos']) and user['carrito']:
            total = sum(i.get('subtotal', 0) for i in user['carrito'])
            texto_respuesta = f"¡Hola de nuevo! Veo que tenías una cotización pendiente por ${total:.2f}. ¿Quieres continuar con esa compra o prefieres empezar una nueva?"
            user['estado'] = 'esperando_confirmacion'
            print("[DEBUG] Ram: saludo_con_carrito")

        # 5. Saludo normal
        elif any(p in msg_low for p in ['hola', 'buenas', 'buen dia', 'saludos']):
            texto_respuesta = f"¡Hola {user['nombre']}! ¿En qué puedo ayudarte hoy? Estamos buscando materiales para construcción."
            print("[DEBUG] Ram: saludo_normal")

        # 6. Pedir total / pagar / resumen
        elif any(p in msg_low for p in ['total', 'cuanto es', 'cuenta', 'pagar', 'pdf', 'cotizacion', 'resumen']):
            if not user['carrito']:
                texto_respuesta = "Aún no tienes productos en tu cotización. ¿Qué material necesitas?"
            else:
                texto_respuesta = generar_resumen_cotizacion(user, telefono)
                user['estado'] = 'esperando_confirmacion'
            print("[DEBUG] Ram: total/pagar")

        # 7. Selección (si hay productos mostrados)
        elif user['ultimos_productos']:
            producto = seleccionar_por_numero(user, mensaje) or seleccionar_por_nombre(user, mensaje)
            if producto:
                print("[DEBUG] Ram: selección (producto válido)")
                texto_respuesta = agregar_producto(user, producto)
            else:
                # No seleccionó, limpiamos y buscamos de nuevo
                print("[DEBUG] Ram: selección fallida, limpiar y buscar")
                user['ultimos_productos'] = []
                m2 = detectar_m2(mensaje)
                if m2:
                    user['m2'] = m2
                user['ultima_busqueda'] = mensaje
                productos = buscar_productos(mensaje)
                if not productos:
                    texto_respuesta = "Lo siento, no encontré productos con esas características. ¿Podrías darme más detalles? (color, medida, tipo)"
                else:
                    user['ultimos_productos'] = productos[:3]
                    texto_respuesta = formatear_opciones(productos[:3], user['m2'])

        # 8. Búsqueda (default)
        else:
            print("[DEBUG] Ram: búsqueda")
            user['ultimos_productos'] = []
            m2 = detectar_m2(mensaje)
            if m2:
                user['m2'] = m2
            user['ultima_busqueda'] = mensaje
            productos = buscar_productos(mensaje)
            if not productos:
                texto_respuesta = "Lo siento, no encontré productos con esas características. ¿Podrías darme más detalles? (color, medida, tipo)"
            else:
                user['ultimos_productos'] = productos[:3]
                texto_respuesta = formatear_opciones(productos[:3], user['m2'])

    if not texto_respuesta:
        texto_respuesta = "Lo siento, no entendí tu mensaje. ¿Puedes repetirlo?"

    save_user(telefono, user)

    # Embellecer solo si es un saludo o despedida (no tocar listas ni resúmenes)
    es_saludo = any(p in texto_respuesta for p in ['¡Hola', 'buen día', 'saludos'])
    es_despedida = any(p in texto_respuesta for p in ['gracias', 'adiós', 'hasta luego'])
    if (es_saludo or es_despedida) and not any(p in texto_respuesta for p in ['Opción', 'Rendimiento', 'Resumen']):
        mejorado = embellecer_con_llm(texto_respuesta, user)
        if mejorado:
            texto_respuesta = mejorado

    user['historial'].append({'role': 'assistant', 'content': texto_respuesta, 'time': datetime.now().isoformat()})
    save_user(telefono, user)

    return texto_respuesta

# ============================================================================
# ENDPOINTS FLASK
# ============================================================================
@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json() or {}
    tel = str(data.get('telefono', '0'))
    nom = data.get('nombre', 'Cliente')
    msg = data.get('mensaje', '')

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
        'db_size': os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0
    }

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🟢 Servidor v3.1 listo en http://0.0.0.0:5001")
    print("   Health check: http://localhost:5001/health")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=5001, debug=True)