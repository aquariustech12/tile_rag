#!/usr/bin/env python3
"""
VAMA 3.5 - VERSIÓN NATURAL
- Prompt humano pero preciso
- Memoria conversacional real
- Validaciones de conexión
- Fine-tuning vectorizado automático
- Happy path para Google boys
"""
import sys, os, re, math, pickle, json, time
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from fpdf import FPDF

# OLLAMA
import ollama
import requests
ollama_client = ollama.Client(host='http://127.0.0.1:11434')
MODELO = "qwen2.5:14b"  # o 3b si prefieres

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
os.makedirs('dataset', exist_ok=True)

# ============================================================================
# VALIDACIONES DE CONEXIÓN
# ============================================================================
def validar_conexiones():
    """Valida que todo esté funcionando antes de arrancar"""
    print("\n🔍 VALIDANDO CONEXIONES...")
    errores = []
    exitos = []
    
    # 1. Validar Ollama
    try:
        r = requests.get('http://127.0.0.1:11434/api/tags', timeout=5)
        if r.status_code == 200:
            models = r.json().get('models', [])
            modelo_base = MODELO.split(':')[0]
            if any(modelo_base in m['name'] for m in models):
                exitos.append(f"✅ Ollama: {MODELO} disponible")
            else:
                errores.append(f"⚠️ Modelo {MODELO} no encontrado en Ollama")
        else:
            errores.append("❌ Ollama no responde en http://127.0.0.1:11434")
    except Exception as e:
        errores.append(f"❌ Error conectando a Ollama: {e}")
    
    # 2. Validar ChromaDB
    try:
        colecciones = chroma.list_collections()
        if len(colecciones) >= 5:
            exitos.append(f"✅ ChromaDB: {len(colecciones)} colecciones")
        else:
            errores.append(f"⚠️ Pocas colecciones en ChromaDB: {len(colecciones)}")
    except Exception as e:
        errores.append(f"❌ Error en ChromaDB: {e}")
    
    # 3. Validar embeddings
    try:
        test_embed = embed_fn(["texto de prueba"])
        if test_embed and len(test_embed[0]) > 100:
            exitos.append("✅ Embeddings funcionando")
        else:
            errores.append("⚠️ Embeddings no funcionan correctamente")
    except Exception as e:
        errores.append(f"❌ Error en embeddings: {e}")
    
    # Mostrar resultados
    for e in exitos:
        print(e)
    for e in errores:
        print(e)
    
    if errores:
        print("\n⚠️ Hay problemas, pero el sistema intentará continuar")
    else:
        print("\n✅ Todas las conexiones OK")
    
    return errores

# ============================================================================
# MEMORIA
# ============================================================================
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
                'total': 0,
                'ultimo_mensaje': '',
                'contador': 0,
                'ultima_visita': None,
                'historial': []  # NUEVO: para memoria conversacional
            }
        else:
            defaults = {
                'nombre': '',
                'carrito': [],
                'ultimos_productos': [],
                'm2': 0,
                'total': 0,
                'ultimo_mensaje': '',
                'contador': 0,
                'ultima_visita': None,
                'historial': []
            }
            for key, val in defaults.items():
                if key not in self.datos[uid]:
                    self.datos[uid][key] = val
        return self.datos[uid]

memoria = Memoria()

# ============================================================================
# LOGS Y DATASET
# ============================================================================
def log_conversacion(datos):
    """Guarda conversación en archivo diario"""
    try:
        fecha = datetime.now().strftime('%Y%m%d')
        archivo = f'logs/conversaciones_{fecha}.log'
        
        with open(archivo, 'a', encoding='utf-8') as f:
            f.write(json.dumps(datos, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"Error en log: {e}")

def guardar_para_finetuning(user_id, user, mensaje, respuesta):
    """Guarda conversaciones exitosas para futuro fine-tuning"""
    try:
        # Solo guardar si hubo interacción útil
        if len(user.get('carrito', [])) > 0 or len(user.get('ultimos_productos', [])) > 0:
            archivo = f'dataset/conversaciones_completas.jsonl'
            
            with open(archivo, 'a', encoding='utf-8') as f:
                registro = {
                    'user_id': user_id,
                    'timestamp': datetime.now().isoformat(),
                    'historial': user.get('historial', [])[-6:],
                    'carrito': user.get('carrito', []),
                    'ultima_pregunta': mensaje,
                    'ultima_respuesta': respuesta
                }
                f.write(json.dumps(registro, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"Error guardando dataset: {e}")

# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================
def extraer_precio(meta):
    for k in ['precio_caja', 'precio_unitario', 'precio_m2', 'precio_final', 'precio']:
        try:
            v = float(meta.get(k, 0))
            if v > 10:
                return v
        except:
            continue
    return None

def buscar(query, intenciones=None, color=None, top_k=10):
    """Busca productos en ChromaDB con filtros mejorados"""
    
    # Determinar colecciones a buscar
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
            res = col.query(query_texts=[query], n_results=top_k * 3)  # Pedir más para poder filtrar
            
            for meta in res['metadatas'][0]:
                if color and color.lower() not in meta.get('color', '').lower():
                    continue
                
                # Extraer precio
                precio = None
                for k in ['precio_caja', 'precio_unitario', 'precio_m2', 'precio_final', 'precio']:
                    try:
                        v = float(meta.get(k, 0))
                        if v > 10:
                            precio = v
                            break
                    except:
                        continue
                
                resultados.append({
                    'codigo': meta.get('codigo', 'N/A'),
                    'descripcion': meta.get('descripcion', ''),
                    'formato': meta.get('formato', ''),
                    'color': meta.get('color', ''),
                    'm2_caja': float(meta.get('metraje_caja', 0) or 0),
                    'precio': precio,
                    'coleccion': col_name
                })
        except Exception as e:
            print(f"Error en {col_name}: {e}")
            continue

    # --- INSERCIÓN 1: LIMPIEZA DE DUPLICADOS ---
    vistos = set()
    unicos = []
    for p in resultados:
        cod = p['codigo']
        if cod not in vistos:
            unicos.append(p)
            vistos.add(cod)
    resultados = unicos # Esto evita que salgan 3 monomandos iguales
    # ------------------------------------------

    # ========================================================================
    # FILTROS ESPECÍFICOS POR TIPO DE BÚSQUEDA
    # ========================================================================
    q = query.lower()
    
    # 1. Si busca PEGAMENTO - FILTRO MEGA ESTRICTO
    if any(word in q for word in ['pegamento', 'pega', 'adhesivo', 'pegazulejo', 'polvo', 'polvos', 'cemix']):
        # Solo conservar productos de la colección 'polvos'
        resultados = [r for r in resultados if r['coleccion'] == 'polvos']
        
        # Filtrar por palabras clave en descripción (solo pegamentos)
        palabras_pegamento = ['pegamento', 'pega', 'adhesivo', 'pegazulejo', 'cemix', 'pegapiso', 'pegamarmol']
        resultados = [r for r in resultados if any(
            word in r['descripcion'].lower() 
            for word in palabras_pegamento
        )]
        
        # Excluir explícitamente cualquier cosa que no sea pegamento
        palabras_excluir = ['lavadero', 'granito', 'piso', 'muro', 'azulejo', 'tinaco', 'monomando', 'perfil']
        resultados = [r for r in resultados if not any(
            excl in r['descripcion'].lower() 
            for excl in palabras_excluir
        )]
        
        # DEBUG: Mostrar qué encontró
        print(f"DEBUG - Búsqueda pegamento: {len(resultados)} resultados")
        for r in resultados[:3]:
            print(f"  - {r['descripcion'][:50]}")
    
    # 2. Si busca MONOMANDO - solo grifería
    elif any(word in q for word in ['monomando', 'grifo', 'llave', 'regadera']):
        resultados = [r for r in resultados if r['coleccion'] == 'griferia']
        resultados = [r for r in resultados if any(
            word in r['descripcion'].lower()
            for word in ['monomando', 'grifo', 'llave', 'mezcladora']
        )]
    
    # 3. Si busca TINACO - solo tinacos
    elif any(word in q for word in ['tinaco', 'cisterna']):
        resultados = [r for r in resultados if r['coleccion'] == 'tinacos']
    
    # 4. Si busca PISO - excluir muros y baños
    elif any(word in q for word in ['piso', 'porcelanato', 'ceramica']):
        resultados = [r for r in resultados if r['coleccion'] in ['nacionales', 'importados']]
        resultados = [r for r in resultados if not any(
            excl in r['descripcion'].lower()
            for excl in ['muro', 'azulejo', 'lavabo', 'wc', 'inodoro']
        )]
    
    # Ordenar por relevancia (palabras clave en descripción)
    resultados.sort(key=lambda x: (
        sum(word in x['descripcion'].lower() for word in q.split()),
        x['precio'] is not None,
        x['descripcion']
    ), reverse=True)
    
    # Limitar a top_k
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
    """Formatea productos con narrativa para que el LLM tenga argumentos de venta"""
    if not productos:
        return "No encontré productos con esas características."
    
    lineas = []
    for i, p in enumerate(productos[:3], 1):
        # Limpieza de precio para el texto
        precio_texto = f"${p['precio']:.2f}" if p['precio'] else "precio por confirmar"
        
        # Construimos una descripción enriquecida (Narrativa)
        # Usamos .get() por si algún metadato viene vacío, que no truene el código
        descripcion_rica = (
            f"{i}. {p['descripcion']} (Cod: {p['codigo']}). "
            f"Este material es de la marca {p.get('proveedor', 'nuestra línea')} "
            f"en formato {p.get('formato', 'estándar')}. "
            f"Tiene un acabado {p.get('acabado', 'de primera')} "
            f"con un estilo de {p.get('tipologia', 'diseño único')}. "
            f"El precio es de {precio_texto}."
        )
        lineas.append(descripcion_rica)
    
    # Unimos todo para que el LLM lo reciba como un bloque de contexto
    return "\n".join(lineas)

def calcular_cantidad(m2_proyecto, m2_caja):
    if m2_proyecto and m2_caja:
        return max(1, math.ceil(m2_proyecto / m2_caja))
    return 1

def actualizar_historial(user, role, content):
    """Mantiene historial de últimos mensajes"""
    if 'historial' not in user:
        user['historial'] = []
    
    user['historial'].append({
        'role': role,
        'content': content,
        'time': datetime.now().isoformat()
    })
    
    # Mantener solo últimos 10 mensajes
    if len(user['historial']) > 10:
        user['historial'] = user['historial'][-10:]
    
    # Formato legible para el prompt
    historial_texto = ""
    for h in user['historial'][-6:]:  # Últimos 3 intercambios
        quien = "Cliente" if h['role'] == 'user' else "Vendedor"
        historial_texto += f"{quien}: {h['content']}\n"
    
    user['ultimos_mensajes'] = historial_texto
    return historial_texto

# ============================================================================
# GENERACIÓN DE RESPUESTA CON PROMPT NATURAL
# ============================================================================
def generar_respuesta_natural(user, msg, productos=None):
    """Prompt natural pero preciso"""
    
    # Construir contexto del carrito
    carrito_str = ""
    if user.get('carrito'):
        carrito_str = "Carrito actual:\n"
        for item in user['carrito']:
            precio_str = f"${item['subtotal']:.2f}" if item['subtotal'] else "pendiente"
            carrito_str += f"- {item['descripcion']}: {item['cantidad']} unidades - {precio_str}\n"
    
    # Construir productos disponibles
    productos_str = ""
    if productos:
        productos_str = "Productos encontrados ahora:\n"
        for i, p in enumerate(productos[:3], 1):
            precio = f"${p['precio']:.2f}" if p['precio'] else "precio por confirmar"
            productos_str += f"{i}. {p['codigo']} - {p['descripcion']} ({p['formato']}) - {precio}\n"
    
    # Obtener historial reciente
    historial = user.get('ultimos_mensajes', 'Sin historial previo')
    
    # Detectar si es primera interacción del día
    es_primera = user.get('contador', 0) < 2
    
    prompt = f"""Eres VAMA, un vendedor amable y profesional de una tienda de materiales para construcción.

HISTORIAL RECIENTE DE LA CONVERSACIÓN (USA ESTO PARA DAR CONTINUIDAD):
{historial}

CONTEXTO ACTUAL:
- Cliente: {user.get('nombre', 'Cliente')}
- Metros cuadrados del proyecto: {user.get('m2', 'aún no especificado')}
{carrito_str}
{productos_str}

MENSAJE DEL CLIENTE: "{msg}"

INSTRUCCIONES PARA RESPONDER:
1. SÉ NATURAL: Usa frases como "¡Claro!", "Con gusto", "Déjame ver", "¿Te parece bien?", "Por supuesto".
2. USA EL CONTEXTO: Si el cliente ya eligió un producto, NO vuelvas a mostrar opciones a menos que pida cambiar.
3. MEMORIA: Si pregunta "¿me recuerdas?" o similar, responde con lo último que estaba cotizando.
4. CÁLCULOS: Si hay m2, calcula las cajas necesarias y menciónalo amablemente.
5. PRECISOS: Usa SIEMPRE los datos reales de productos y carrito. NUNCA inventes.
6. {('SALUDO: Es primera interacción del día, saluda calurosamente.' if es_primera else 'CONTINUACIÓN: Retoma naturalmente donde quedaron.')}
7. DESPEDIDA: Al final, pregunta si necesita algo más o si quiere ver el total.

RESPUESTA (solo el mensaje para el cliente, sin explicaciones):"""
    
    try:
        r = ollama_client.generate(
            model=MODELO,
            prompt=prompt,
            options={
                "temperature": 0.4,  # Suficiente para ser natural pero controlado
                "num_predict": 150,
                "stop": ["Cliente:", "\n\n"]
            }
        )
        
        respuesta = r['response'].strip()
        
        # Limpieza mínima (solo caracteres problemáticos)
        respuesta = respuesta.replace("**", "").replace("__", "")
        
        return respuesta
        
    except Exception as e:
        print(f"Error LLM: {e}")
        return "Disculpa, estoy teniendo problemas. ¿Puedes repetir?"

# ============================================================================
# FLUJOS PRINCIPALES
# ============================================================================
def flujo_presentar_productos(user, productos, es_follow_up=False):
    """Muestra productos de forma natural PERO SIEMPRE muestra la lista"""
    user['ultimos_productos'] = productos
    lista = formatear_lista(productos)
    
    # Construir prompt para el LLM
    if not user.get('m2'):
        prompt = f"El cliente {user.get('nombre', '')} está viendo opciones de {productos[0]['descripcion'][:30]}. Presenta estas opciones de forma natural y pregúntale para cuántos m² los necesita. Usa su nombre si lo tienes."
    else:
        prompt = f"El cliente {user.get('nombre', '')} ya indicó que necesita {user['m2']}m². Presenta estas opciones de forma natural, usa su nombre, y pídele que elija una (1, 2 o 3)."
    
    # Intentar que el LLM responda
    try:
        respuesta_llm = generar_respuesta_natural(user, prompt, productos)
        
        # Si el LLM incluyó la lista, bien; si no, la agregamos
        if "1." in respuesta_llm or "2." in respuesta_llm:
            return respuesta_llm
        else:
            return f"{respuesta_llm}\n\n{lista}"
    except:
        # Si falla el LLM, usamos nuestro mensaje base con la lista
        if not user.get('m2'):
            return f"Claro, encontré estas opciones:\n\n{lista}\n\n¿Para cuántos m² los necesitas?"
        else:
            return f"Claro, para tus {user['m2']}m² tengo estas opciones:\n\n{lista}\n\n¿Cuál te interesa? (responde 1, 2 o 3)"

def flujo_agregar_producto(user, msg, productos):
    seleccion = None
    
    # Intentar por número
    num = re.search(r'\b([123])\b', msg)
    if num:
        idx = int(num.group(1)) - 1
        if 0 <= idx < len(productos):
            seleccion = productos[idx]
    
    # Intentar por código
    if not seleccion:
        for p in productos:
            if p['codigo'].lower() in msg.lower():
                seleccion = p
                break
    
    if not seleccion:
        return flujo_presentar_productos(user, productos, es_follow_up=True)
    
    cantidad = calcular_cantidad(user.get('m2', 0), seleccion['m2_caja'])
    item = {
        'codigo': seleccion['codigo'],
        'descripcion': seleccion['descripcion'],
        'precio': seleccion['precio'],
        'cantidad': cantidad,
        'subtotal': (seleccion['precio'] or 0) * cantidad
    }
    user['carrito'].append(item)
    
    # Actualizar total
    total = 0
    for i in user['carrito']:
        if i['subtotal']:
            total += i['subtotal']
    user['total'] = total
    
    precio_str = f"${seleccion['precio']:.2f}" if seleccion['precio'] else "precio por confirmar"
    subtotal_str = f"${item['subtotal']:.2f}" if seleccion['precio'] else "por confirmar"
    
    return f"¡Listo! Agregué {cantidad} unidad(es) de {seleccion['descripcion']} a {precio_str} c/u. Subtotal parcial: {subtotal_str}. ¿Necesitas algo más o quieres ver el total?"

def flujo_total(user, user_id=None):
    if not user['carrito']:
        return "Tu cotización está vacía. ¿Qué material necesitas?"
    
    lineas = []
    total = 0
    for item in user['carrito']:
        if item['precio']:
            lineas.append(f"• {item['descripcion']}: {item['cantidad']} x ${item['precio']:.2f} = ${item['subtotal']:.2f}")
            total += item['subtotal']
        else:
            lineas.append(f"• {item['descripcion']}: {item['cantidad']} (precio por confirmar)")
    
    resumen = "\n".join(lineas)
    total_str = f"${total:.2f}" if total > 0 else "Por confirmar"
    
    # Guardar total en memoria
    user['total'] = total
    
    # Generar PDF si hay total
    pdf_msg = ""
    if total > 0 and user_id:
        pdf_file = generar_pdf(user_id, user)
        if pdf_file:
            pdf_msg = f"\n\n📄 Puedes descargar tu cotización aquí: {request.host_url}pdf/{user_id}"

    # ========================================================================
    # REGISTRO DE COTIZACIÓN CERRADA (SEGURO)
    # ========================================================================
    try:
        if total > 0 and user_id:
            registro = {
                'fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'cliente': user.get('nombre', 'Cliente'),
                'telefono': user_id,
                'total': total,
                'productos': len(user['carrito']),
                'items': user['carrito']
            }
            
            # Guardar en archivo JSON
            archivo_registro = 'cotizaciones_cerradas.json'
            
            # Leer existente o crear nuevo
            if os.path.exists(archivo_registro):
                with open(archivo_registro, 'r', encoding='utf-8') as f:
                    registros = json.load(f)
            else:
                registros = []
            
            registros.append(registro)
            
            with open(archivo_registro, 'w', encoding='utf-8') as f:
                json.dump(registros, f, indent=2, ensure_ascii=False)
            
            print(f"📝 Cotización cerrada guardada: ${total} - {user.get('nombre', 'Cliente')}")
    except Exception as e:
        print(f"Error guardando registro de cotización: {e}")
        
    return f"**Resumen de tu cotización:**\n\n{resumen}\n\n**TOTAL: {total_str}**{pdf_msg}\n\n¿Te parece bien? ¿Confirmamos disponibilidad?"

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
            desc = item['descripcion'][:35] + "..." if len(item['descripcion']) > 35 else item['descripcion']
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
        pdf.cell(0, 10, f'TOTAL: ${total:.2f}', 0, 1, 'R')
        
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

# ============================================================================
# PROCESADOR PRINCIPAL
# ============================================================================
def procesar(user_id, nombre, mensaje):
    user = memoria.get(user_id)
    
    # Registrar visita
    ahora = datetime.now().isoformat()
    user['ultima_visita'] = ahora
    
    if nombre and not user['nombre']:
        user['nombre'] = nombre
    
    msg = mensaje.strip()
    msg_low = msg.lower()
    
    # Guardar mensaje en historial
    actualizar_historial(user, 'user', msg)
    
    # ========================================================================
    # SALUDO - SIEMPRE NATURAL
    # ========================================================================
    es_saludo = any(w in msg_low for w in ['hola', 'buenas', 'buen dia', 'ola', 'de nuevo', 'hey', 'saludos'])
    if es_saludo:
        user['contador'] += 1
        
        # Verificar si tiene carrito para personalizar saludo
        if user['carrito']:
            prompt = f"Saluda a {user.get('nombre', 'cliente')} por su nombre, dile que tiene {len(user['carrito'])} productos en cotización y pregúntale si quiere continuar o agregar algo más."
        else:
            prompt = f"Saluda a {user.get('nombre', 'cliente')} por su nombre de forma cálida y pregúntale qué necesita cotizar hoy."
        
        respuesta = generar_respuesta_natural(user, prompt)
        actualizar_historial(user, 'assistant', respuesta)
        guardar_para_finetuning(user_id, user, msg, respuesta)
        return respuesta
    
    user['contador'] += 1
    
    # ========================================================================
    # NUEVA COTIZACIÓN - CON LLM NATURAL
    # ========================================================================
    reset_palabras = ['nueva cotizacion', 'nueva cotización', 'reiniciar', 'empezar de nuevo', 
                      'borrar todo', 'iniciar nueva', 'reset', 'limpiar', 'desde cero']
    if any(w in msg_low for w in reset_palabras):
        user['carrito'] = []
        user['ultimos_productos'] = []
        user['m2'] = 0
        user['total'] = 0
        
        prompt = f"El cliente {user.get('nombre', '')} quiere empezar una cotización completamente nueva. Confírmało amablemente, usa su nombre si lo tienes, y pregúntale qué material necesita cotizar ahora."
        respuesta = generar_respuesta_natural(user, prompt)
        actualizar_historial(user, 'assistant', respuesta)
        return respuesta
    
    # ========================================================================
    # MEMORIA EXPLÍCITA
    # ========================================================================
    if any(w in msg_low for w in ['recuerdas', 'acuerdas', 'qué cotizaba', 'qué tenía', 'lo último']):
        if user['carrito']:
            items = ", ".join([f"{item['descripcion'][:30]}" for item in user['carrito'][-2:]])
            prompt = f"El cliente pregunta si recuerdas lo que estaba cotizando. Tenía: {items}. Responde amablemente usando su nombre y pregúnta si quiere agregar algo más o ver el total."
        elif user['ultimos_productos']:
            prompt = f"El cliente pregunta si recuerdas lo que estábamos viendo. Estábamos viendo opciones de {user['ultimos_productos'][0]['descripcion'][:30]}. Responde amablemente y pregúnta si le interesa alguno."
        else:
            prompt = f"El cliente pregunta si recuerdas algo de él. No tiene nada en cotización hoy. Responde amablemente y pregúnta qué necesita."
        
        respuesta = generar_respuesta_natural(user, prompt)
        actualizar_historial(user, 'assistant', respuesta)
        return respuesta
    
    # ========================================================================
    # PDF
    # ========================================================================
    if "pdf" in msg_low or "cotizacion en pdf" in msg_low:
        if user['carrito']:
            respuesta = flujo_total(user, user_id)
        else:
            respuesta = "Aún no tienes productos en tu cotización. ¿Qué material necesitas?"
        actualizar_historial(user, 'assistant', respuesta)
        return respuesta
    
    # ========================================================================
    # DETECTAR M2 - GUARDAR SIEMPRE
    # ========================================================================
    m2 = detectar_m2(msg)
    if m2:
        user['m2'] = m2
    
    # ========================================================================
    # TOTAL
    # ========================================================================
    es_total = any(w in msg_low for w in ['total', 'es todo', 'ya es todo', 'eso es todo', 
                                          'terminamos', 'seria todo', 'cuanto es', 'cuánto es', 'cerrar', 'finalizar'])
    
    # ========================================================================
    # INTERCEPTOR DE MUEBLES Y ACCESORIOS (Evita que Chroma busque pisos)
    # ========================================================================
    muebles_p = ['lavamanos', 'tinaco', 'taza', 'sanitario', 'wc', 'mezcladora', 
                 'ovalin', 'mueble', 'regadera', 'monomando', 'espejo']
    
    if any(w in msg_low for w in muebles_p):
        # En lugar de ir a buscar azulejos a la base de datos, el Qwen responde
        prompt = f"""
        El cliente {user.get('nombre', 'Julián')} pregunta por: '{msg}'. 
        No tenemos este producto específico en el catálogo de recubrimientos de ChromaDB.
        Responde como vendedor experto de VAMA. 
        Confirma que manejamos esos accesorios, dile que tenemos varios modelos 
        y pregúntale qué estilo o medida está buscando para darle opciones.
        """
        respuesta = generar_respuesta_natural(user, prompt)
        actualizar_historial(user, 'assistant', respuesta)
        return respuesta # <--- Cortamos aquí para que NO ofrezca pisos

    # ========================================================================
    # --- INSERCIÓN 2: HAPPY PATH (CIERRE Y CORTESÍA) ---
    # ========================================================================
    palabras_cierre = ['pago', 'pagar', 'cuenta', 'clabe', 'transferencia', 'gracias', 'ok', 'bye', 'esperando']
    
    if any(w in msg_low for w in palabras_cierre):
        # Usamos .get('total') para que no truene con KeyError si no hay total aún
        total_monto = user.get('total', 'tu cotización')
        
        prompt = f"""
        El cliente dice '{msg}'. Ya tiene {len(user.get('carrito', []))} productos.
        Si pide la cuenta: Banco BBVA, CLABE 012345678901234567.
        Si agradece: Sé amable, despídete y dile que esperas su comprobante.
        NO OFREZCAS PISOS NI PRODUCTOS NUEVOS.
        """
        respuesta = generar_respuesta_natural(user, prompt)
        actualizar_historial(user, 'assistant', respuesta)
        return respuesta # Aquí cortamos el flujo para que NO busque en Chroma
    # ---------------------------------------------------

    # ========================================================================
    # --- INSERCIÓN 3: SEGURO DE MEMORIA ---
    # ========================================================================
    # Si el cliente no está pidiendo nada nuevo y ya hay carrito, no busques más.
    if len(msg_low.split()) < 4 and user.get('carrito') and not any(w in msg_low for w in ['quiero', 'busca', 'dame']):
        prompt = f"El cliente solo está saludando o siguiendo la plática: '{msg}'. Responde algo breve y amable sin ofrecer productos."
        respuesta = generar_respuesta_natural(user, prompt)
        actualizar_historial(user, 'assistant', respuesta)
        return respuesta
    # --------------------------------------

    # ========================================================================
    # INTERCEPTOR DE PAGO (CORREGIDO SIN ERRORES)
    # ========================================================================
    palabras_pago = ['pago', 'pagar', 'cuenta', 'clabe', 'transferencia', 'depósito', 'tarjeta', 'link']
    
    if any(w in msg_low for w in palabras_pago):
        if user.get('carrito'):
            # Usamos .get() para evitar el KeyError si 'total' no existe aún
            total_monto = user.get('total', 'la cotización actual')
            
            prompt = f"""
            El cliente {user.get('nombre', 'Julián')} está listo para pagar: {total_monto}.
            Responde como vendedor profesional de VAMA:
            1. Proporciona estos datos: Banco: BBVA, Cuenta: 0123 4567 8901, CLABE: 012345678901234567.
            2. Pídele amablemente que envíe el comprobante por aquí mismo para liberar el pedido.
            3. Despídete confirmando que en cuanto recibas el pago, se programa la entrega en 24-48 horas.
            """
            respuesta = generar_respuesta_natural(user, prompt)
            actualizar_historial(user, 'assistant', respuesta)
            return respuesta

    # 2. >>> PEGA AQUÍ EL BLOQUE DE DISPONIBILIDAD <<<
    palabras_cierre = ['disponibilidad', 'confirma', 'cuándo llega', 'cuando llega', 'existencias', 'stock', 'cuanto tarda']
    if any(w in msg_low for w in palabras_cierre):
        if user['carrito']:
            productos_lista = "\n".join([f"- {p['descripcion']}" for p in user['carrito']])
            prompt = f"El cliente {user.get('nombre', 'Julián')} quiere confirmar disponibilidad para: {productos_lista}. Responde que sí hay stock, que entregamos en 24-48h y pregunta cómo prefiere pagar."
            respuesta = generar_respuesta_natural(user, prompt)
            actualizar_historial(user, 'assistant', respuesta)
            return respuesta

    # ========================================================================
    # SEGURO ANTI-VENTA INFINITA (Si ya hay carrito, no ofrezcas pisos gratis)
    # ========================================================================
    # Si el mensaje es corto o de seguimiento ("estoy esperando", "qué paso", "mándalo")
    # y ya tenemos productos en el carrito, NO busques más pisos.
    palabras_seguimiento = ['esperando', 'paso', 'mandalo', 'dale', 'manda', 'ok', 'listo']
    
    if any(w in msg_low for w in palabras_seguimiento) and user.get('carrito'):
        prompt = f"El cliente dice '{msg}' y está esperando finalizar su compra de {user.get('total', 'su pedido')}. Recuérdale los datos de pago o pregúntale si tiene alguna duda con la cuenta que le pasaste."
        respuesta = generar_respuesta_natural(user, prompt)
        actualizar_historial(user, 'assistant', respuesta)
        return respuesta

    # 3. BÚSQUEDA GENERAL (Si no fue mueble ni fue cierre, entonces busca pisos)
    ints = detectar_intenciones(msg)

    if es_total:
        respuesta = flujo_total(user, user_id)
        actualizar_historial(user, 'assistant', respuesta)
        return respuesta
    
    # ========================================================================
    # SELECCIÓN DE PRODUCTO (solo si es claramente una elección)
    # ========================================================================
    if user['ultimos_productos']:
        # Detectar si es SOLO un número (1, 2, 3) o un código
        es_numero = re.search(r'^\s*[123]\s*$', msg_low)
        es_codigo = any(p['codigo'].lower() in msg_low for p in user['ultimos_productos'])
        
        # También detectar frases como "el 1", "el primero", etc.
        es_seleccion_texto = any(f" {i} " in f" {msg_low} " or f"el {i}" in msg_low for i in ['1','2','3'])
        
        if es_numero or es_codigo or es_seleccion_texto:
            respuesta = flujo_agregar_producto(user, msg, user['ultimos_productos'])
            actualizar_historial(user, 'assistant', respuesta)
            return respuesta
    
    # ========================================================================
    # MÁS OPCIONES
    # ========================================================================
    if any(w in msg_low for w in ['mas opciones', 'más opciones', 'otras opciones', 'ver más', 'otros', 'otro', 'diferente']):
        # Buscar sin filtro de color para ampliar resultados
        ints = detectar_intenciones(msg)
        productos = buscar(msg, intenciones=ints, color=None, top_k=5)
        
        if productos and len(productos) > 0:
            # Si son los mismos productos, intentar búsqueda más amplia
            if user['ultimos_productos'] and productos[0]['codigo'] == user['ultimos_productos'][0]['codigo']:
                # Buscar en todas las colecciones sin restricciones
                productos = buscar(msg, intenciones=None, color=None, top_k=5)
            
            respuesta = flujo_presentar_productos(user, productos)
        else:
            respuesta = "Lo siento, no encontré más opciones. ¿Quieres probar con otro color, formato o tipo de material?"
        
        actualizar_historial(user, 'assistant', respuesta)
        return respuesta
    
    # ========================================================================
    # MOSTRAR DE NUEVO
    # ========================================================================
    if user['ultimos_productos'] and any(w in msg_low for w in ['muestra', 'ver', 'de nuevo', 'otra vez', 'cuales', 'opciones', '?' ]):
        respuesta = flujo_presentar_productos(user, user['ultimos_productos'])
        actualizar_historial(user, 'assistant', respuesta)
        return respuesta
    
    # ========================================================================
    # BUSCAR NUEVOS PRODUCTOS
    # ========================================================================
    ints = detectar_intenciones(msg)
    color = detectar_color(msg)
    productos = buscar(msg, intenciones=ints, color=color, top_k=3)
    
    if not productos:
        respuesta = f"Disculpa, no encontré {', '.join(ints) if ints else 'productos'} con esas características. ¿Puedes darme más detalles? (color, medida, uso)"
        actualizar_historial(user, 'assistant', respuesta)
        return respuesta
    
    respuesta = flujo_presentar_productos(user, productos)
    actualizar_historial(user, 'assistant', respuesta)
    return respuesta

# ============================================================================
# API
# ============================================================================
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
        total_users = len(memoria.datos)
        total_cotizaciones = sum(1 for u in memoria.datos.values() if u.get('carrito'))
        
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
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .stat {{ background: white; padding: 20px; margin: 10px 0; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .num {{ font-size: 2.5em; color: #2c3e50; font-weight: bold; }}
                h1 {{ color: #34495e; }}
                pre {{ background: #eee; padding: 10px; border-radius: 5px; overflow-x: auto; }}
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
            <pre>{os.popen(f'tail -20 {archivo} 2>/dev/null || echo "Sin logs"').read()}</pre>
            <hr>
            <h3>Dataset para fine-tuning:</h3>
            <pre>{os.popen('tail -5 dataset/conversaciones_completas.jsonl 2>/dev/null || echo "Generando..."').read()}</pre>
        </body>
        </html>
        """
        return html
    except Exception as e:
        return f"Error: {e}", 500

@app.route('/health', methods=['GET'])
def health():
    """Endpoint de salud para monitoreo"""
    return jsonify({
        'status': 'ok',
        'modelo': MODELO,
        'timestamp': datetime.now().isoformat(),
        'usuarios': len(memoria.datos)
    })

# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("🚀 VAMA 3.5 - VERSIÓN NATURAL")
    print("=" * 60)
    print(f"📁 Modelo: {MODELO}")
    print(f"📁 PDFs: ./cotizaciones/")
    print(f"📁 Logs: ./logs/")
    print(f"📁 Dataset: ./dataset/")
    print(f"📊 Dashboard: http://localhost:5001/dashboard")
    print(f"📄 PDF endpoint: /pdf/<telefono>")
    print(f"🩺 Health: http://localhost:5001/health")
    print("=" * 60)
    
    # Validar conexiones
    errores = validar_conexiones()
    
    print("\n🔧 Iniciando servidor...")
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)