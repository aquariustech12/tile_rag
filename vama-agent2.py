#!/usr/bin/env python3
"""
VAMA 3.4 - BETA 2
- Filtro anti-alucinaciones para 3b/14b
- PDF de cotización
- Dashboard en tiempo real
- Logging corregido
"""
import sys, os, re, math, pickle, json, time
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from fpdf import FPDF

# OLLAMA
import ollama
ollama_client = ollama.Client(host='http://127.0.0.1:11434')
MODELO = "qwen2.5:14b"  # Cambiar a 3b si es necesario

# CHROMA
import chromadb
from chromadb.utils import embedding_functions
os.environ["TOKENIZERS_PARALLELISM"] = "false"
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-m3", device="cpu"
)
chroma = chromadb.PersistentClient(path="chroma_db_v3")

# 12 COLECCIONES
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

# Crear directorios necesarios
os.makedirs('cotizaciones', exist_ok=True)
os.makedirs('logs', exist_ok=True)

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
                'contador': 0,
                'ultima_visita': None
            }
        else:
            defaults = {
                'nombre': '',
                'carrito': [],
                'ultimos_productos': [],
                'm2': 0,
                'ultimo_mensaje': '',
                'contador': 0,
                'ultima_visita': None
            }
            for key, val in defaults.items():
                if key not in self.datos[uid]:
                    self.datos[uid][key] = val
        return self.datos[uid]

memoria = Memoria()

def log_conversacion(datos):
    """Guarda conversación en archivo diario"""
    try:
        fecha = datetime.now().strftime('%Y%m%d')
        archivo = f'logs/conversaciones_{fecha}.log'
        
        with open(archivo, 'a', encoding='utf-8') as f:
            f.write(json.dumps(datos, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"Error en log: {e}")

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

def hablar(prompt_contexto, temperatura=0.05, max_tokens=60):
    """FILTRO ANTI-ALUCINACIONES - Versión emergencia"""
    try:
        # Prompt base ultra-restrictivo
        prompt_completo = f"""Eres VAMA, sistema técnico de cotizaciones de materiales de construcción.
REGLAS ABSOLUTAS - INCUMPLIR ES ERROR CRÍTICO:
1. PROHIBIDO: emociones, entusiasmo, "me encanta", "ilumina", "emoción"
2. PROHIBIDO: "¿cómo estás?", "¿te gustaría?", "¿quieres saber?"
3. PROHIBIDO: metáforas, "ojos", "brillan", "hermoso", "perfecto"
4. OBLIGATORIO: datos concretos, precios, cantidades, códigos
5. ESTILO: Seco, directo, profesional. Ejemplo: "Productos disponibles. ¿Cuál prefiere?"

{prompt_contexto}"""
        
        r = ollama_client.generate(
            model=MODELO,
            prompt=prompt_completo,
            options={
                "temperature": temperatura,
                "num_predict": max_tokens,
                "stop": ["Cliente:", "Usuario:", "\n\n", "me encanta", "ilumina", "emoción", "feliz", "¿cómo estás", "¿te gustaría"]
            }
        )
        
        respuesta = r['response'].strip().replace("**", "").replace("*", "").replace("¡", "").replace("!", "")
        
        # FILTRO HARDCODED - Palabras prohibidas
        prohibidas = [
            "me encanta", "me ilumina", "mis ojos", "emociona", "feliz", 
            "iluminan", "proyectos", "espacios", "renovar", "alegría",
            "¿cómo estás", "¿te gustaría", "¿quieres saber", "me fascina",
            "me emociona", "brillan", "hermoso", "perfecto", "ideal"
        ]
        
        for frase in prohibidas:
            if frase.lower() in respuesta.lower():
                print(f"⚠️ ALUCINACIÓN FILTRADA: {respuesta[:60]}...")
                # Respuesta de emergencia según contexto
                if "presenta" in prompt_contexto.lower() or "opciones" in prompt_contexto.lower():
                    return "Aquí tiene las opciones disponibles. Indique 1, 2 o 3. Equipo VAMA."
                elif "agregaste" in prompt_contexto.lower() or "confirm" in prompt_contexto.lower():
                    return "Producto agregado. ¿Algo más o ver total? Equipo VAMA."
                elif "total" in prompt_contexto.lower():
                    return "Resumen generado. ¿Confirma disponibilidad? Equipo VAMA."
                else:
                    return "¿Necesita algo más? Equipo VAMA."
        
        return respuesta if respuesta else "¿Necesita información? Equipo VAMA."
        
    except Exception as e:
        print(f"LLM error: {e}")
        return "¿Necesita ayuda? Equipo VAMA."

def generar_pdf(user_id, user):
    """Genera PDF de cotización"""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'COTIZACIÓN VAMA', 0, 1, 'C')
        
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 10, f'Fecha: {datetime.now().strftime("%d/%m/%Y %H:%M")}', 0, 1)
        pdf.cell(0, 10, f'Cliente: {user["nombre"] or "Cliente"}', 0, 1)
        pdf.cell(0, 10, f'Teléfono: {user_id}', 0, 1)
        pdf.ln(5)
        
        # Tabla
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(80, 8, 'Producto', 1)
        pdf.cell(25, 8, 'Cantidad', 1)
        pdf.cell(35, 8, 'P.Unitario', 1)
        pdf.cell(35, 8, 'Subtotal', 1)
        pdf.ln()
        
        total = 0
        pdf.set_font('Arial', '', 9)
        
        for item in user['carrito']:
            desc = item['descripcion'][:35] if len(item['descripcion']) > 35 else item['descripcion']
            pdf.cell(80, 8, desc, 1)
            pdf.cell(25, 8, str(item['cantidad']), 1, 0, 'C')
            
            if item['precio']:
                pdf.cell(35, 8, f"${item['precio']:.2f}", 1, 0, 'R')
                pdf.cell(35, 8, f"${item['subtotal']:.2f}", 1, 0, 'R')
                total += item['subtotal']
            else:
                pdf.cell(35, 8, "P/C", 1, 0, 'C')
                pdf.cell(35, 8, "P/C", 1, 0, 'C')
            pdf.ln()
        
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, f'TOTAL: ${total:.2f}' if total > 0 else 'TOTAL: Por confirmar', 0, 1, 'R')
        
        pdf.ln(10)
        pdf.set_font('Arial', '', 8)
        pdf.cell(0, 5, 'VAMA - Materiales y Acabados', 0, 1, 'C')
        pdf.cell(0, 5, 'Esta cotización tiene validez de 7 días', 0, 1, 'C')
        
        filename = f"cotizaciones/cot_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(filename)
        return filename
        
    except Exception as e:
        print(f"Error PDF: {e}")
        return None

def flujo_presentar_productos(user, productos, es_follow_up=False):
    lista = formatear_lista(productos)
    user['ultimos_productos'] = productos
    
    if not user['m2'] and not es_follow_up:
        return "¿Para cuántos m2 necesitas el material? Así calculo las cajas exactas. Equipo VAMA."
    
    prompt = f"""Presenta opciones de productos al cliente.
Productos: {lista[:200]}
Instrucción: Indique número de opción deseada (1, 2 o 3)."""
    
    texto = hablar(prompt, temperatura=0.05, max_tokens=50)
    return f"{texto}\n\n{lista}\n\n¿Cuál te interesa? (dime 1, 2 o 3) Equipo VAMA."

def flujo_agregar_producto(user, msg):
    if not user['ultimos_productos']:
        return "Primero indique qué productos busca. ¿Qué material necesita? Equipo VAMA."
    
    seleccion = None
    
    num = re.search(r'\b([123])\b', msg)
    if num:
        idx = int(num.group(1)) - 1
        if 0 <= idx < len(user['ultimos_productos']):
            seleccion = user['ultimos_productos'][idx]
    
    if not seleccion:
        for p in user['ultimos_productos']:
            if p['codigo'].lower() in msg.lower():
                seleccion = p
                break
    
    if not seleccion:
        return flujo_presentar_productos(user, user['ultimos_productos'], es_follow_up=True)
    
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
    
    prompt = f"""Confirma agregado de producto: {cantidad} unidades de {seleccion['descripcion']}.
Subtotal: {subtotal_str}.
Pregunte si desea agregar más productos o ver total."""
    
    return hablar(prompt, max_tokens=60)

def flujo_total(user, user_id=None, generar_pdf_flag=False):
    if not user['carrito']:
        return "Cotización vacía. ¿Qué material necesita? Equipo VAMA."
    if generar_pdf_flag:
        pdf_file = generar_pdf(user_id_global, user) # Necesita user id

    lineas = []
    total = 0
    for item in user['carrito']:
        if item['precio']:
            lineas.append(f"- {item['descripcion'][:40]}: {item['cantidad']} x ${item['precio']:.2f} = ${item['subtotal']:.2f}")
            total += item['subtotal']
        else:
            lineas.append(f"- {item['descripcion'][:40]}: {item['cantidad']} (precio por confirmar)")
    
    resumen = "\n".join(lineas)
    total_str = f"${total:.2f}" if total > 0 else "Por confirmar"
    
    # Generar PDF si se solicita
    pdf_msg = ""
    if generar_pdf_flag and user_id:
        pdf_file = generar_pdf(user_id, user)
        if pdf_file:
            pdf_msg = f"\n📄 PDF: {request.host_url}pdf/{user_id}"
    
    prompt = f"""Presente resumen final de cotización. Total: {total_str}.
Solicite confirmación de disponibilidad."""
    
    texto = hablar(prompt, max_tokens=60)
    return f"{texto}{pdf_msg}\n\n{resumen}\n\n**TOTAL: {total_str}**\n\n¿Confirma disponibilidad? Equipo VAMA."

def flujo_saludo(user, es_visita_nueva=False):
    nombre = user['nombre'] or 'Cliente'
    
    if es_visita_nueva or user.get('contador', 0) > 0:
        if user['carrito']:
            return f"¡Hola de nuevo {nombre}! Tiene {len(user['carrito'])} productos en cotización. ¿Continuamos? Equipo VAMA."
        else:
            return f"¡Hola de nuevo {nombre}! ¿En qué puedo ayudarle hoy? Equipo VAMA."
    
    return f"¡Hola {nombre}! Soy VAMA, asistente de materiales. ¿Qué necesita cotizar? Equipo VAMA."

def procesar(user_id, nombre, mensaje):
    global user_id_global
    user_id_global = user_id
    
    user = memoria.get(user_id)
    
    ahora = datetime.now().isoformat()
    es_visita_nueva = False
    if user.get('ultima_visita'):
        try:
            ultima = datetime.fromisoformat(user['ultima_visita'])
            if (datetime.now() - ultima).total_seconds() > 1800:
                es_visita_nueva = True
        except:
            pass
    user['ultima_visita'] = ahora
    
    if nombre and not user['nombre']:
        user['nombre'] = nombre
    
    msg = mensaje.strip()
    msg_low = msg.lower()
    
    # Saludo
    es_saludo = any(w in msg_low for w in ['hola', 'buenas', 'buen dia', 'ola', 'de nuevo', 'hey'])
    if es_saludo:
        return flujo_saludo(user, es_visita_nueva)
    
    user['contador'] += 1
    
    # Reset
    reset_palabras = ['nueva cotizacion', 'nueva cotización', 'reiniciar', 'empezar de nuevo', 
                      'borrar todo', 'iniciar nueva', 'nueva', 'empezar', 'reset', 'limpiar',
                      'iniciar', 'otra cotizacion', 'nuevo', 'desde cero']
    if any(w in msg_low for w in reset_palabras):
        user['carrito'] = []
        user['ultimos_productos'] = []
        user['m2'] = 0
        return "Listo, nueva cotización. ¿Qué material necesita? Equipo VAMA."
    
    # PDF explícito
    if "pdf" in msg_low or "cotizacion en pdf" in msg_low:
        if user['carrito']:
            return flujo_total(user, generar_pdf_flag=True)
        else:
            return "Cotización vacía. Agregue productos primero. Equipo VAMA."
    
    # M2
    m2 = detectar_m2(msg)
    if m2:
        user['m2'] = m2
        if user['ultimos_productos']:
            return flujo_presentar_productos(user, user['ultimos_productos'], es_follow_up=True)
    
    # Total
    es_total = any(w in msg_low for w in ['total', 'es todo', 'ya es todo', 'eso es todo', 
                                          'terminamos', 'seria todo', 'cuanto es', 'cuánto es', 'cerrar', 'finalizar'])
    if es_total:
        quiere_pdf = "pdf" in msg_low or "cotizacion" in msg_low
        return flujo_total(user, user_id, generar_pdf_flag=quiere_pdf)
    
    # Selección
    es_seleccion = user['ultimos_productos'] and (
        re.search(r'\b[123]\b', msg) or 
        any(p['codigo'].lower() in msg_low for p in user['ultimos_productos'])
    )
    if es_seleccion:
        return flujo_agregar_producto(user, msg)
    
    # Mostrar de nuevo
    es_mostrar = user['ultimos_productos'] and any(w in msg_low for w in ['muestra', 'ver', 'de nuevo', 'otra vez', 'cuales'])
    if es_mostrar:
        return flujo_presentar_productos(user, user['ultimos_productos'], es_follow_up=True)
    
    # Buscar nuevo
    ints = detectar_intenciones(msg)
    color = detectar_color(msg)
    
    if not es_seleccion and not es_mostrar:
        user['ultimos_productos'] = []
    
    productos = buscar(msg, intenciones=ints, color=color, top_k=3)
    
    if not productos:
        return f"No encontré {', '.join(ints)}. ¿Otras características? Equipo VAMA."
    
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
    
    # Logging
    log_conversacion({
        'timestamp': datetime.now().isoformat(),
        'telefono': tel,
        'nombre': nom,
        'mensaje': msg,
        'respuesta': resp,
        'modelo': MODELO
    })
    
    print(f"[R] {resp[:80]}...")
    return jsonify({'respuesta': resp})

@app.route('/pdf/<telefono>', methods=['GET'])
def descargar_pdf(telefono):
    """Endpoint para descargar último PDF del cliente"""
    try:
        # Buscar PDF más reciente del cliente
        import glob
        pdfs = glob.glob(f"cotizaciones/cot_{telefono}_*.pdf")
        if pdfs:
            pdf_reciente = max(pdfs, key=os.path.getctime)
            return send_file(pdf_reciente, as_attachment=True)
        return jsonify({'error': 'No hay PDF para este cliente'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard', methods=['GET'])
def dashboard():
    """Dashboard simple en tiempo real"""
    try:
        # Estadísticas básicas
        total_users = len(memoria.datos)
        total_cotizaciones = sum(1 for u in memoria.datos.values() if u.get('carrito'))
        
        # Leer logs de hoy
        fecha = datetime.now().strftime('%Y%m%d')
        archivo = f'logs/conversaciones_{fecha}.log'
        interacciones_hoy = 0
        if os.path.exists(archivo):
            with open(archivo, 'r') as f:
                interacciones_hoy = len(f.readlines())
        
        html = f"""
        <html>
        <head>
            <title>VAMA Dashboard</title>
            <meta http-equiv="refresh" content="10">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .stat {{ background: #f0f0f0; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .num {{ font-size: 2em; color: #333; }}
                h1 {{ color: #555; }}
            </style>
        </head>
        <body>
            <h1>📊 VAMA Dashboard - {datetime.now().strftime('%d/%m/%Y %H:%M')}</h1>
            <div class="stat">
                <div class="num">{total_users}</div>
                <div>Usuarios totales</div>
            </div>
            <div class="stat">
                <div class="num">{total_cotizaciones}</div>
                <div>Cotizaciones activas</div>
            </div>
            <div class="stat">
                <div class="num">{interacciones_hoy}</div>
                <div>Interacciones hoy</div>
            </div>
            <hr>
            <h3>Logs recientes:</h3>
            <pre>{os.popen(f'tail -10 {archivo} 2>/dev/null || echo "Sin logs"').read()}</pre>
        </body>
        </html>
        """
        return html
    except Exception as e:
        return f"Error: {e}", 500

if __name__ == '__main__':
    print(f"🚀 VAMA 3.4 BETA 2 | Modelo: {MODELO}")
    print(f"📁 PDFs: ./cotizaciones/ | Logs: ./logs/")
    print(f"📊 Dashboard: http://localhost:5001/dashboard")
    print(f"📄 PDF endpoint: /pdf/<telefono>")
    app.run(host='0.0.0.0', port=5001, debug=False)
