#!/usr/bin/env python3
"""
VAMA API - Endpoints para n8n
Buscar productos, generar PDF, sucursales, y GESTIÓN DE USUARIOS
"""

import os
import sys
import json
import re
import math
import glob
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify, send_file
import chromadb
from chromadb.utils import embedding_functions
from fpdf import FPDF
import ollama
import requests

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
CHROMA_PATH = "./chroma_db_v3"
OLLAMA_HOST = 'http://127.0.0.1:11434'
MODELO = "qwen2.5:14b"
DATABASE = "/home/julian/tile_rag/usuarios.db"  # Ruta exacta que proporcionaste

print("=" * 60)
print("🚀 VAMA API - Servicio de datos")
print("=" * 60)

# ============================================================================
# INICIALIZAR BASE DE DATOS SQLITE
# ============================================================================
def init_db():
    """Crea la tabla de usuarios si no existe"""
    with sqlite3.connect(DATABASE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS usuarios (
                telefono TEXT PRIMARY KEY,
                nombre TEXT,
                estado TEXT DEFAULT 'inicio',
                m2 INTEGER DEFAULT 0,
                carrito TEXT DEFAULT '[]',
                ultimos_productos TEXT DEFAULT '[]',
                ultima_busqueda TEXT DEFAULT ''
            )
        """)
        conn.commit()
    print(f"✅ Base de datos inicializada: {DATABASE}")

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
# ENDPOINTS DE USUARIOS (NUEVOS - Reemplazan SQLite de n8n)
# ============================================================================

@app.route('/user/<telefono>', methods=['GET'])
def get_user(telefono):
    """Obtiene el estado de un usuario por teléfono"""
    try:
        with sqlite3.connect(DATABASE) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute('SELECT * FROM usuarios WHERE telefono = ?', [telefono])
            row = cur.fetchone()

            if row:
                user = dict(row)
                return jsonify(user)
            else:
                return jsonify({"error": "Usuario no encontrado"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/user', methods=['POST'])
def create_or_update_user():
    """Crea o actualiza un usuario"""
    try:
        data = request.get_json() or {}
        telefono = data.get('telefono')

        if not telefono:
            return jsonify({"error": "telefono es requerido"}), 400

        # Preparar datos (del body o valores por defecto)
        nombre = data.get('nombre', '')
        estado = data.get('estado', 'inicio')
        m2 = data.get('m2', 0)
        carrito = data.get('carrito', [])
        ultimos_productos = data.get('ultimos_productos', [])
        ultima_busqueda = data.get('ultima_busqueda', '')
        respuesta = data.get('respuesta', '')

        # Asegurar que carrito y ultimos_productos sean strings JSON si vienen como arrays
        if isinstance(carrito, list):
            carrito_json = json.dumps(carrito)
        else:
            carrito_json = carrito
            
        if isinstance(ultimos_productos, list):
            ultimos_productos_json = json.dumps(ultimos_productos)
        else:
            ultimos_productos_json = ultimos_productos

        with sqlite3.connect(DATABASE) as conn:
            cur = conn.execute('SELECT 1 FROM usuarios WHERE telefono = ?', [telefono])
            exists = cur.fetchone() is not None

            if exists:
                # UPDATE
                conn.execute("""
                    UPDATE usuarios 
                    SET estado = ?, m2 = ?, carrito = ?, ultimos_productos = ?, ultima_busqueda = ?
                    WHERE telefono = ?
                """, [estado, m2, carrito_json, ultimos_productos_json, ultima_busqueda, telefono])
            else:
                # INSERT
                conn.execute("""
                    INSERT INTO usuarios (telefono, nombre, estado, m2, carrito, ultimos_productos, ultima_busqueda)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, [telefono, nombre, estado, m2, carrito_json, ultimos_productos_json, ultima_busqueda])

            conn.commit()
            
            # ✅ DEVOLVER LOS DATOS DEL USUARIO, NO SOLO STATUS
            return jsonify({
                "status": "ok",
                "action": "update" if exists else "create",
                "telefono": telefono,
                "nombre": nombre,
                "mensaje": data.get('mensaje', ''),  # ← AGREGAR ESTA LÍNEA
                "estado": estado,
                "m2": m2,
                "carrito": carrito if isinstance(carrito, list) else json.loads(carrito) if carrito else [],
                "ultimos_productos": ultimos_productos if isinstance(ultimos_productos, list) else json.loads(ultimos_productos) if ultimos_productos else [],
                "ultima_busqueda": ultima_busqueda,
                "respuesta": respuesta
            })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/user/<telefono>/reset', methods=['POST'])
def reset_user(telefono):
    """Resetea el estado de un usuario a valores iniciales"""
    try:
        with sqlite3.connect(DATABASE) as conn:
            conn.execute("""
                UPDATE usuarios 
                SET estado='inicio', carrito='[]', ultimos_productos='[]', m2=0 
                WHERE telefono=?
            """, [telefono])
            conn.commit()
            return jsonify({"status": "reset", "telefono": telefono})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================================
# MAPEO DE COLECCIONES (igual que antes)
# ============================================================================
MAP_COLECCIONES = {
    'piso': ['nacionales', 'importados'],
    'azulejo': ['nacionales', 'importados'],
    'porcelanato': ['nacionales', 'importados'],
    'muro': ['muros', 'nacionales', 'importados'],
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
    'cemix': ['polvos'],
    'mortero': ['polvos'],
    'cemento': ['polvos'],
    'concreto': ['polvos']
}

# ============================================================================
# FUNCIONES DE UTILIDAD (igual que antes)
# ============================================================================
def detectar_intenciones_producto(msg):
    msg = msg.lower()
    intenciones = []
    for palabra, cols in MAP_COLECCIONES.items():
        if palabra in msg:
            intenciones.append(palabra)
    if len(intenciones) > 1 and 'piso' in intenciones:
        intenciones.remove('piso')
    if not intenciones:
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

def generar_pdf(user_id, carrito, nombre="Cliente"):
    """Genera PDF de cotización y devuelve la ruta del archivo"""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'COTIZACIÓN VAMA', 0, 1, 'C')
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 10, f'Fecha: {datetime.now().strftime("%d/%m/%Y %H:%M")}', 0, 1)
        pdf.cell(0, 10, f'Cliente: {nombre}', 0, 1)
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

        for item in carrito:
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

# ============================================================================
# ENDPOINTS EXISTENTES
# ============================================================================
@app.route('/buscar', methods=['POST'])
def buscar():
    data = request.get_json() or {}
    query = data.get('query', '')
    m2 = data.get('m2', 0)
    
    # Guardar datos del usuario para devolverlos
    telefono = data.get('telefono', '')
    nombre = data.get('nombre', '')
    mensaje = data.get('mensaje', '')
    carrito = data.get('carrito', [])
    estado = data.get('estado', 'inicio')
    
    if not query:
        return jsonify({'error': 'query required'}), 400
        
    productos = buscar_productos(query)
    for p in productos:
        if p['coleccion'] in ['nacionales', 'importados', 'muros'] and m2 > 0 and p['m2_caja'] > 0:
            p['cantidad_sugerida'] = math.ceil(m2 / p['m2_caja'])
        else:
            p['cantidad_sugerida'] = 1
            
    # ✅ Devolver productos + datos del usuario
    return jsonify({
        'productos': productos,
        'telefono': telefono,
        'nombre': nombre,
        'mensaje': mensaje,
        'm2': m2,
        'carrito': carrito if isinstance(carrito, list) else [],
        'estado': estado,
        'ultimos_productos': []  # Se llenará en Format Búsqueda
    })

@app.route('/generar_pdf', methods=['POST'])
def crear_pdf():
    data = request.get_json() or {}
    user_id = data.get('user_id')
    carrito = data.get('carrito')
    nombre = data.get('nombre', 'Cliente')
    if not user_id or not carrito:
        return jsonify({'error': 'user_id y carrito requeridos'}), 400
    filename = generar_pdf(user_id, carrito, nombre)
    if filename:
        base_url = "https://agente3.aegis-oceanit.work"
        pdf_url = f"{base_url}/pdf_download/{user_id}"
        return jsonify({'pdf_url': pdf_url})
    else:
        return jsonify({'error': 'Error generando PDF'}), 500

@app.route('/pdf_download/<user_id>')
def descargar_pdf(user_id):
    pattern = f"cotizaciones/cot_{user_id}_*.pdf"
    pdfs = glob.glob(pattern)
    if pdfs:
        pdfs.sort(key=os.path.getmtime, reverse=True)
        return send_file(pdfs[0], as_attachment=True, download_name=f'cotizacion_{user_id}.pdf')
    else:
        return "No hay cotizaciones disponibles para este usuario", 404

@app.route('/sucursales', methods=['GET'])
def sucursales():
    sucursales = [
        {"nombre": "CULIACÁN - TRES RIOS", "direccion": "Blvd. Enrique Sánchez Alonso #1515, Desarrollo Urbano Tres Ríos", "telefono": "(667) 752 20 78", "horario": "Lun-Vie 8am-8pm, Sáb 9am-6pm"},
        {"nombre": "CULIACÁN - IGNACIO RAMÍREZ", "direccion": "Ignacio Ramírez #981 pte, Jorge Almada", "telefono": "(667) 752 02 61", "horario": "Lun-Vie 8am-8pm, Sáb 9am-6pm"},
        {"nombre": "LOS MOCHIS - BIENESTAR", "direccion": "Bienestar #633, Col. Bienestar", "telefono": "(668) 812 58 13", "horario": "Lun-Vie 8am-8pm, Sáb 9am-6pm"},
        {"nombre": "LOS MOCHIS - INDEPENDENCIA", "direccion": "Av. Independencia #2049 pte, Jardines del Country", "telefono": "(668) 176 96 13", "horario": "Lun-Vie 8am-8pm, Sáb 9am-6pm"}
    ]
    return jsonify(sucursales)

@app.route('/llm/intent', methods=['POST'])
def llm_intent():
    data = request.get_json()
    mensaje = data.get('mensaje', '')
    estado = data.get('estado', 'inicio')
    ultimos_productos = data.get('ultimos_productos', [])
    carrito = data.get('carrito', [])
    m2 = data.get('m2', 0)

    prompt = f"""Eres un asistente de ventas. Tu única tarea es clasificar la intención del usuario y extraer entidades. No generes respuestas conversacionales.

Mensaje del usuario: "{mensaje}"
Estado actual: {estado}
Últimos productos mostrados: {[p.get('descripcion', p) for p in ultimos_productos]}
Carrito tiene {len(carrito)} productos.
Metros cuadrados en contexto: {m2}

Devuelve ÚNICAMENTE un JSON con estos campos:
- intent: uno de ["saludo", "buscar", "seleccion", "sucursales", "envio", "checkout", "quitar"]
- entidades: objeto con (producto_tipo, color, m2, opcion_numero, codigo_producto) según corresponda

IMPORTANTE:
- Si el mensaje se refiere a un número de opción (ej: "opción 1", "quiero la 2", "la primera") y hay productos mostrados, intent = "seleccion" y opcion_numero = ese número.
- Si el mensaje pide productos nuevos (ej: "y en color gris"), intent = "buscar".
- No incluyas campo "respuesta_sugerida".

Ejemplo: {{"intent": "seleccion", "entidades": {{"opcion_numero": 1}}}}
"""
    try:
        response = ollama_client.generate(model=MODELO, prompt=prompt, stream=False)
        resultado = json.loads(response['response'])
        return jsonify(resultado)
    except Exception as e:
        return jsonify({"intent": "buscar", "entidades": {}}), 500

@app.route('/llm/embellish', methods=['POST'])
def embellish():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({"embellished": ""})
    prompt = f"Reescribe el siguiente mensaje de un asesor de ventas para que sea cálido, amigable y natural, pero sin añadir explicaciones ni prefijos. Devuelve solo el mensaje reescrito:\n\n{text}"
    try:
        response = ollama_client.generate(model=MODELO, prompt=prompt, stream=False)
        # Limpiar posibles comillas o prefijos residuales
        emb = response['response'].strip()
        # Si el LLM aún agrega algo como "Claro, aquí tienes...", lo quitamos
        if emb.startswith('"') and emb.endswith('"'):
            emb = emb[1:-1]
        return jsonify({"embellished": emb})
    except Exception as e:
        print(f"Error embellish: {e}")
        return jsonify({"embellished": text})

@app.route('/health')
def health():
    return {'status': 'ok', 'colecciones': list(colecciones.keys())}

if __name__ == '__main__':
    init_db()  # Inicializar base de datos
    print("\n" + "=" * 60)
    print("🟢 API VAMA lista en http://0.0.0.0:5001")
    print("   Health: http://localhost:5001/health")
    print("   User endpoints:")
    print("     GET  /user/<telefono>")
    print("     POST /user")
    print("     POST /user/<telefono>/reset")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=5001, debug=True)
