#ingest_tiles_catalog.py
"""
VAMA - Sistema de Ingesta de Catálogo a ChromaDB
Correcciones críticas aplicadas
"""

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os
import shutil
import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CHROMA_PATH = "chroma_db_v3"  # Nueva versión para no mezclar datos corruptos
DATA_PATH = "data"
BATCH_SIZE = 100  # Para ingesta por lotes y evitar memory leaks

# ============================================================================
# CONFIGURACIÓN DE PRECIOS Y REGLAS DE NEGOCIO
# ============================================================================

# Rangos válidos por categoría para detección de errores (MXN)
RANGOS_PRECIOS = {
    'piso': {'min_m2': 30, 'max_m2': 5000, 'default_m2': 150},
    'muro': {'min_m2': 25, 'max_m2': 3000, 'default_m2': 120},
    'sanitario': {'min_unit': 200, 'max_unit': 50000, 'default_unit': 1500},
    'griferia': {'min_unit': 150, 'max_unit': 20000, 'default_unit': 800},
    'instalacion': {'min_unit': 50, 'max_unit': 2000, 'default_unit': 200},
    'accesorios': {'min_unit': 20, 'max_unit': 5000, 'default_unit': 150},
    'mobiliario': {'min_unit': 500, 'max_unit': 50000, 'default_unit': 2500},
    'otros': {'min_m2': 10, 'max_m2': 10000, 'default_m2': 100}
}

# Palabras clave para detectar precios por pieza vs por m2
PALABRAS_PIEZA = ['pieza', 'piezas', 'c/u', 'cada uno', 'pza', 'pz', 'unitario', 'juego']
PALABRAS_M2 = ['m2', 'm²', 'metro cuadrado', 'metros cuadrados', 'por m2']

# ============================================================================
# LIMPIEZA Y CORRECCIÓN DE DATOS
# ============================================================================

def limpiar_db_anterior():
    """Limpia base de datos anterior de forma segura"""
    if os.path.exists(CHROMA_PATH):
        logger.info(f"🗑️  Eliminando DB anterior: {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH)
    os.makedirs(CHROMA_PATH, exist_ok=True)

def corregir_precio(val, contexto: Dict = None) -> Tuple[float, str]:
    """
    Corrige y valida precios con contexto de categoría.
    Retorna: (precio_corregido, metodo_usado)
    """
    if pd.isna(val) or val in [0, '0', '', None, 'nan', 'NaN', 'NULL', '-']: 
        return 0.0, "invalido"
    
    try:
        # Convertir a string y limpiar
        if isinstance(val, str):
            # Quitar símbolos de moneda y separadores de miles
            s = val.replace('$', '').replace('MXN', '').replace('USD', '')
            s = s.replace(',', '').replace(' ', '').strip()
            
            # Manejar paréntesis (números negativos o anotaciones)
            s = s.replace('(', '-').replace(')', '')
            
            # Si contiene múltiples puntos decimales, probablemente es separador de miles
            if s.count('.') > 1:
                s = s.replace('.', '')
            
            # Convertir a float
            try:
                num = float(s)
            except ValueError:
                # Intentar extraer número con regex
                numeros = re.findall(r'[\d.]+', s)
                if numeros:
                    num = float(numeros[0])
                else:
                    return 0.0, "no_parseable"
        else:
            num = float(val)
        
        # Detectar y corregir decimales desplazados
        # Si es un precio de piso y es menor a 10, probablemente falta un cero o está en miles
        categoria = contexto.get('categoria', 'otros') if contexto else 'otros'
        
        # Lógica de corrección específica por categoría
        if categoria in ['sanitario', 'griferia', 'mobiliario']:
            # Estos van por pieza, no por m2
            if num < 50 and num > 0:
                # Probablemente está en miles (4.8 = 4800) o faltan ceros
                if num < 10:
                    num = num * 1000  # 4.8 -> 4800
                else:
                    num = num * 100   # 40 -> 4000
                return num, "corregido_x100/x1000"
            
            # Si es muy alto, probablemente es error
            if num > 100000:
                num = num / 100
                return num, "corregido_div100"
        else:
            # Pisos y muros por m2
            if num < 5 and num > 0:
                # Probablemente está en cientos (2.96 = 296)
                num = num * 100
                return num, "corregido_x100"
            
            if num > 10000:
                # Probablemente es precio por caja, dividir por metraje típico
                num = num / 1.44
                return num, "corregido_div_caja"
        
        return num, "original"
        
    except Exception as e:
        logger.warning(f"No se pudo convertir precio '{val}': {e}")
        return 0.0, "error"

def detectar_unidad_medida(row: pd.Series, descripcion: str) -> str:
    """
    Detecta si el precio es por m2 o por pieza basado en múltiples señales
    """
    desc_upper = str(descripcion).upper()
    
    # 1. Revisar columnas específicas de unidad
    for col in row.index:
        col_upper = str(col).upper()
        val = str(row[col]).upper()
        
        if any(u in col_upper for u in ['UNIDAD', 'UMEDIDA', 'U.M.', 'UM']):
            if any(p in val for p in ['M2', 'M²', 'MT2']):
                return 'm2'
            elif any(p in val for p in ['PZ', 'PZA', 'PIEZA', 'JGO', 'JUEGO']):
                return 'pieza'
    
    # 2. Revisar descripción
    if any(p in desc_upper for p in ['X M2', 'POR M2', '/M2', 'EL M2', 'M2 ']):
        return 'm2'
    if any(p in desc_upper for p in ['PZA', 'PZ ', 'PIEZA', 'JUEGO', 'SET', 'KIT']):
        return 'pieza'
    
    # 3. Inferir por categoría detectada
    cat_clues = clasificar_categoria(descripcion)
    if cat_clues['categoria'] in ['sanitario', 'griferia', 'mobiliario', 'accesorios']:
        return 'pieza'
    
    return 'm2'  # Default para pisos/muros

def extraer_metraje_caja(row: pd.Series, categoria: str) -> float:
    """
    Extrae metraje por caja de forma inteligente
    """
    # Buscar en columnas comunes
    candidatos = ['Metraje', 'Metraje caja', 'M2 por caja', 'M2 caja', 'Contenido']
    
    for col in candidatos:
        matches = [c for c in row.index if col.upper() in c.upper()]
        for m in matches:
            try:
                val = str(row[m]).replace(',', '').replace('m2', '').replace('m²', '').strip()
                # Buscar patrón número x número (ej: 1.44 o 1.92)
                nums = re.findall(r'\d+\.?\d*', val)
                if nums:
                    metraje = float(nums[0])
                    if 0.5 <= metraje <= 10:  # Rango razonable
                        return metraje
            except:
                continue
    
    # Inferir por formato
    formato = str(row.get('Formato', '')).upper()
    formatos_comunes = {
        '30X60': 1.08,
        '60X60': 1.44,
        '60X120': 1.44,
        '45X45': 1.42,
        '20X60': 1.08,
        '30X30': 0.81,
        '80X80': 1.28,
        '120X120': 2.88
    }
    
    for fmt, m2 in formatos_comunes.items():
        if fmt in formato.replace(' ', '').replace('X', 'X').replace('x', 'X'):
            return m2
    
    # Defaults por categoría
    if categoria == 'piso':
        return 1.44
    elif categoria == 'muro':
        return 1.5
    return 1.0

# ============================================================================
# CLASIFICACIÓN MEJORADA
# ============================================================================

def clasificar_categoria(descripcion: str) -> Dict:
    """Clasificación robusta con extracción de atributos"""
    if not descripcion or pd.isna(descripcion):
        desc_upper = ""
    else:
        desc_upper = str(descripcion).upper()
    
    categoria = "otros"
    subcategoria = ""
    tipologia = ""
    acabado = ""
    color = ""
    
    # Diccionario de palabras clave expandido
    keywords = {
        'sanitario': ['WC', 'ONE PIECE', 'TANQUE', 'LAVABO', 'OVALIN', 'SANITARIO', 
                      'INODORO', 'TAZA', 'RETRETE', 'PEDESTAL', 'MINGITORIO'],
        'griferia': ['MEZCLADORA', 'GRIFO', 'MONOMANDO', 'GRIFERIA', 'LLAVE', 
                     'MEZCLADOR', 'DOSIFICADOR', 'DUCHA', 'REGADERA'],
        'piso': ['PISO', 'PISOS', 'LOSETA', 'BALDOSA'],
        'muro': ['MURO', 'MURAL', 'FACHALETA', 'AZULEJO', 'PANEL', 'REVES', 
                 'REVESTIMIENTO', 'MOSAICO'],
        'instalacion': ['PEGAMENTO', 'ADHESIVO', 'BOQUILLA', 'POLVO', 'CEMENTO', 
                        'MORTERO', 'PASTA', 'SELLADOR', 'IMPERMEABILIZANTE'],
        'accesorios': ['PERFIL', 'LISTELO', 'ZOCLO', 'ESQUINERO', 'TRANSICION',
                       'ROMPEVIENTO', 'DESAGÜE', 'REJILLA', 'SIFÓN'],
        'mobiliario': ['MUEBLE', 'ESPEJO', 'TINA', 'TARJA', 'LAVAMANOS', 'VANITY']
    }
    
    # Detectar categoría principal
    for cat, words in keywords.items():
        if any(w in desc_upper for w in words):
            categoria = cat
            break
    
    # Subcategorías específicas
    if categoria == 'sanitario':
        if 'ONE PIECE' in desc_upper or 'ONEPIECE' in desc_upper:
            subcategoria = 'one_piece'
        elif 'TANQUE' in desc_upper:
            subcategoria = 'tanque'
        elif 'LAVABO' in desc_upper or 'OVALIN' in desc_upper:
            subcategoria = 'lavabo'
    
    elif categoria == 'griferia':
        if 'LAVABO' in desc_upper:
            subcategoria = 'lavabo'
        elif any(r in desc_upper for r in ['REGADERA', 'DUCHA', 'MONOMANDO']):
            subcategoria = 'regadera'
        elif 'COCINA' in desc_upper or 'TARJA' in desc_upper:
            subcategoria = 'cocina'
    
    # Tipología de material
    tipologias = {
        'marmol': ['MARMOL', 'MÁRMOL', 'MARBLE'],
        'madera': ['MADERA', 'WOOD', 'ROBLE', 'NOGAL', 'ENCINO'],
        'piedra': ['PIEDRA', 'STONE', 'GRANITO', 'CUARZO'],
        'cemento': ['CEMENTO', 'CONCRETO', 'MICROCEMENTO'],
        'porcelanico': ['PORCELANICO', 'PORCELÁNICO', 'PORCELAIN'],
        'ceramico': ['CERAMICO', 'CERÁMICO', 'CERAMICA'],
        'monocolor': ['MONOCOLOR', 'LISO']
    }
    
    for tipo, words in tipologias.items():
        if any(w in desc_upper for w in words):
            tipologia = tipo
            break
    
    # Acabado
    acabados = {
        'mate': ['MATE', 'MATTE', 'SATIN', 'SATINADO'],
        'brillante': ['BRILLANTE', 'BRILLO', 'GLOSSY', 'PULIDO', 'POLISHED'],
        'rustico': ['RUSTICO', 'RÚSTICO', 'TEXTURIZADO', 'ANTIDERRAPANTE'],
        'natural': ['NATURAL', 'RAW']
    }
    
    for acab, words in acabados.items():
        if any(w in desc_upper for w in words):
            acabado = acab
            break
    
    # Color - lista expandida
    colores_map = {
        'blanco': ['BLANCO', 'WHITE', 'BLANCA'],
        'negro': ['NEGRO', 'BLACK', 'NEGROS', 'EBANO'],
        'gris': ['GRIS', 'GRAY', 'GREY', 'PLATA', 'CEMENTO'],
        'beige': ['BEIGE', 'HUESO', 'MARFIL', 'IVORY', 'CREMA'],
        'cafe': ['CAFE', 'MARRON', 'BROWN', 'NOGAL', 'ROBLE', 'MOKA'],
        'azul': ['AZUL', 'BLUE', 'NAVY', 'TURQUESA'],
        'verde': ['VERDE', 'GREEN', 'ESMERALDA', 'JADE'],
        'rojo': ['ROJO', 'RED', 'BURDEO'],
        'terracota': ['TERRACOTA', 'COBRE', 'LADRILLO']
    }
    
    for col, words in colores_map.items():
        if any(w in desc_upper for w in words):
            color = col
            break
    
    return {
        "categoria": categoria,
        "subcategoria": subcategoria,
        "tipologia": tipologia,
        "acabado": acabado,
        "color": color
    }

# ============================================================================
# EXTRACCIÓN DE PRECIOS MEJORADA
# ============================================================================

def extraer_precio_efectivo(row: pd.Series, descripcion: str) -> Dict:
    """
    Extrae el mejor precio disponible con detección de unidad de medida
    """
    precios = {}
    categoria = clasificar_categoria(descripcion)['categoria']
    unidad = detectar_unidad_medida(row, descripcion)
    
    # Mapeo de columnas de precio por prioridad
    columnas_precio = [
        ('oferta_final', ['OFERTA FINAL', 'PRECIO OFERTA', 'OFERTA', 'PROMO']),
        ('con_complementos', ['CON COMPLEMENTO', 'CON COMPLEMENTOS', 'COMPLEMENTO']),
        ('regular', ['PRECIO REGULAR', 'REGULAR', 'PRECIO LISTA', 'LISTA']),
        ('sistema', ['PRECIO SISTEMA', 'SISTEMA']),
        ('m2_base', ['PRECIO M2', 'PRECIO/M2', 'M2', 'PRECIO METRO'])
    ]
    
    for key, posibles_nombres in columnas_precio:
        for col in row.index:
            col_upper = str(col).upper()
            if any(p in col_upper for p in posibles_nombres):
                val = row[col]
                precio, metodo = corregir_precio(val, {'categoria': categoria})
                if precio > 0:
                    precios[key] = {'valor': precio, 'metodo': metodo, 'columna': col}
                    break
    
    # Seleccionar mejor precio por prioridad
    precio_final = 0.0
    fuente = "default"
    
    for key in ['oferta_final', 'con_complementos', 'regular', 'sistema', 'm2_base']:
        if key in precios:
            precio_final = precios[key]['valor']
            fuente = key
            break
    
    # Si no hay precio, usar default según categoría
    if precio_final == 0:
        rangos = RANGOS_PRECIOS.get(categoria, RANGOS_PRECIOS['otros'])
        if unidad == 'pieza':
            precio_final = rangos['default_unit']
        else:
            precio_final = rangos['default_m2']
        fuente = "default_categoria"
    
    # Calcular precio por caja si aplica
    metraje_caja = extraer_metraje_caja(row, categoria)
    
    if unidad == 'm2':
        precio_caja = precio_final * metraje_caja
    else:
        # Si ya es por pieza, el precio_caja es el mismo
        precio_caja = precio_final
        # Pero para mantener consistencia, calculamos precio_m2 si es posible
        if metraje_caja > 0:
            precio_m2_equiv = precio_final / metraje_caja
        else:
            precio_m2_equiv = precio_final
    
    return {
        "precio_efectivo_m2": precio_final if unidad == 'm2' else precio_m2_equiv,
        "precio_unitario": precio_final if unidad == 'pieza' else precio_caja,
        "precio_caja": precio_caja,
        "metraje_caja": metraje_caja,
        "unidad_medida": unidad,
        "fuente_precio": fuente,
        "metodo_correccion": precios.get(fuente, {}).get('metodo', 'default')
    }

# ============================================================================
# CONSTRUCCIÓN DE DOCUMENTOS
# ============================================================================

def construir_documento_semantico(row: pd.Series, clasificacion: Dict, precios: Dict) -> str:
    """Construye texto enriquecido para embeddings"""
    partes = []
    
    # Descripción base
    desc = str(row.get('Descripcion', '')).strip()
    if desc and desc.lower() != 'nan':
        partes.append(desc)
    
    # Atributos estructurados
    atributos = []
    
    if clasificacion['categoria']:
        atributos.append(f"Categoría: {clasificacion['categoria']}")
    if clasificacion['subcategoria']:
        atributos.append(f"Tipo: {clasificacion['subcategoria']}")
    if clasificacion['tipologia']:
        atributos.append(f"Material: {clasificacion['tipologia']}")
    if clasificacion['acabado']:
        atributos.append(f"Acabado: {clasificacion['acabado']}")
    if clasificacion['color']:
        atributos.append(f"Color: {clasificacion['color']}")
    
    # Formato y dimensiones
    formato = str(row.get('Formato', '')).strip()
    if formato and formato.lower() != 'nan':
        atributos.append(f"Formato: {formato}")
    
    if precios['metraje_caja']:
        atributos.append(f"Metraje: {precios['metraje_caja']}m2 por caja")
    
    # Proveedor
    proveedor = str(row.get('Proveedor', '')).strip()
    if proveedor and proveedor.lower() != 'nan':
        atributos.append(f"Marca: {proveedor}")
    
    # Uso recomendado basado en categoría
    usos = []
    cat = clasificacion['categoria']
    if cat == 'piso':
        usos.extend(['piso', 'suelo'])
        if clasificacion['tipologia'] in ['porcelanico', 'marmol']:
            usos.extend(['interior', 'exterior', 'baño', 'cocina'])
    elif cat == 'muro':
        usos.extend(['muro', 'pared', 'baño', 'cocina'])
    elif cat == 'sanitario':
        usos.extend(['baño', 'sanitario', 'wc'])
    elif cat == 'griferia':
        usos.extend(['griferia', 'baño', 'cocina'])
    
    if usos:
        atributos.append(f"Uso: {', '.join(list(set(usos)))}")
    
    # Unir todo
    texto_atributos = " | ".join(atributos)
    documento = f"{desc}. {texto_atributos}" if atributos else desc
    
    return documento

# ============================================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================================

def procesar_archivo(filepath: str, collection) -> Tuple[int, List[str]]:
    """Procesa un archivo CSV completo"""
    filename = os.path.basename(filepath)
    logger.info(f"📄 Procesando: {filename}")
    
    errores = []
    productos_indexados = 0
    
    try:
        # Detectar encoding
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
            try:
                df = pd.read_csv(filepath, encoding=encoding, dtype=str)
                break
            except UnicodeDecodeError:
                continue
        
        # Limpiar nombres de columnas
        df.columns = df.columns.str.strip().str.upper()
        
        # Verificar columna mínima
        desc_cols = [c for c in df.columns if 'DESCRIP' in c]
        if not desc_cols:
            logger.warning(f"⚠️  {filename}: No tiene columna de descripción")
            return 0, ["sin_descripcion"]
        
        desc_col = desc_cols[0]
        
        # Filtrar filas inválidas
        mask = (
            (df[desc_col].notna()) &
            (df[desc_col].str.len() > 3) &
            (~df[desc_col].str.contains('^(PROVEEDOR|CATEGORIA|ESTATUS|DESCRIPCION)$', case=False, na=False))
        )
        df = df[mask].reset_index(drop=True)
        
        logger.info(f"   ✓ {len(df)} productos válidos encontrados")
        
        # Procesar en lotes para eficiencia
        batch_docs = []
        batch_metadatas = []
        batch_ids = []
        
        for idx, row in df.iterrows():
            try:
                descripcion = str(row.get(desc_col, '')).strip()
                if not descripcion or len(descripcion) < 3:
                    continue
                
                # Extraer información
                clasificacion = clasificar_categoria(descripcion)
                precios = extraer_precio_efectivo(row, descripcion)
                documento = construir_documento_semantico(row, clasificacion, precios)
                
                # Extraer código
                codigo_cols = [c for c in row.index if any(x in c for x in ['CODIGO', 'COD', 'SKU', 'CLAVE', 'MODELO'])]
                codigo = str(row.get(codigo_cols[0], '')).strip() if codigo_cols else f"AUTO_{idx}"
                if not codigo or codigo.lower() in ['nan', '']:
                    codigo = f"AUTO_{idx}"
                
                # Determinar tipo de producto desde nombre de archivo
                tipo_prod = "nacional"
                fname_lower = filename.lower()
                if "importado" in fname_lower:
                    tipo_prod = "importado"
                elif "descontinuado" in fname_lower:
                    tipo_prod = "descontinuado"
                elif any(x in fname_lower for x in ["promo", "oferta"]):
                    tipo_prod = "promocion"
                elif any(x in fname_lower for x in ["polvo", "instalacion", "pegamento"]):
                    tipo_prod = "instalacion"
                
                # Construir metadata
                proveedor = str(row.get([c for c in row.index if 'PROVEEDOR' in c][0], '')).strip() if any('PROVEEDOR' in c for c in row.index) else ''
                formato = str(row.get([c for c in row.index if 'FORMATO' in c][0], '')).strip() if any('FORMATO' in c for c in row.index) else ''
                
                metadata = {
                    "codigo": codigo[:50],
                    "descripcion": descripcion[:200],
                    "proveedor": proveedor[:50] if proveedor and proveedor.lower() != 'nan' else '',
                    "precio_m2": float(precios['precio_efectivo_m2']),
                    "precio_unitario": float(precios['precio_unitario']),
                    "precio_caja": float(precios['precio_caja']),
                    "metraje_caja": float(precios['metraje_caja']),
                    "unidad_medida": precios['unidad_medida'],
                    "formato": formato[:30] if formato and formato.lower() != 'nan' else '',
                    "categoria": clasificacion['categoria'],
                    "subcategoria": clasificacion['subcategoria'],
                    "tipologia": clasificacion['tipologia'],
                    "acabado": clasificacion['acabado'],
                    "color": clasificacion['color'],
                    "tipo_producto": tipo_prod,
                    "archivo_fuente": filename,
                    "fuente_precio": precios['fuente_precio'],
                    "fecha_ingesta": datetime.now().isoformat()
                }
                
                product_id = f"{filename.replace('.csv', '')}_{idx}_{codigo[:20]}"
                
                # Agregar a batch
                batch_docs.append(documento)
                batch_metadatas.append(metadata)
                batch_ids.append(product_id)
                
                # Insertar cuando el batch esté lleno
                if len(batch_docs) >= BATCH_SIZE:
                    collection.add(
                        documents=batch_docs,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                    productos_indexados += len(batch_docs)
                    batch_docs, batch_metadatas, batch_ids = [], [], []
                
            except Exception as e:
                errores.append(f"{filename} fila {idx}: {str(e)[:100]}")
                continue
        
        # Insertar batch final
        if batch_docs:
            collection.add(
                documents=batch_docs,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            productos_indexados += len(batch_docs)
        
        logger.info(f"   ✅ Indexados: {productos_indexados}")
        return productos_indexados, errores
        
    except Exception as e:
        logger.error(f"   ❌ Error grave en {filename}: {e}")
        return 0, [str(e)]

# ============================================================================
# REPORTE DE CALIDAD DE DATOS
# ============================================================================

def generar_reporte_calidad(collection):
    """Genera estadísticas de calidad de los datos indexados"""
    logger.info("\n" + "="*60)
    logger.info("📊 REPORTE DE CALIDAD DE DATOS")
    logger.info("="*60)
    
    # Conteos por categoría
    all_meta = collection.get()
    
    categorias = {}
    unidades = {}
    fuentes_precio = {}
    precios_cero = 0
    precios_extremos = {'bajos': 0, 'altos': 0}
    
    for meta in all_meta['metadatas']:
        cat = meta.get('categoria', 'otros')
        categorias[cat] = categorias.get(cat, 0) + 1
        
        unidad = meta.get('unidad_medida', 'desconocida')
        unidades[unidad] = unidades.get(unidad, 0) + 1
        
        fuente = meta.get('fuente_precio', 'desconocida')
        fuentes_precio[fuente] = fuentes_precio.get(fuente, 0) + 1
        
        precio = meta.get('precio_m2', 0) or meta.get('precio_unitario', 0)
        if precio == 0:
            precios_cero += 1
        elif precio < 10:
            precios_extremos['bajos'] += 1
        elif precio > 50000:
            precios_extremos['altos'] += 1
    
    logger.info(f"\n📦 Total productos: {len(all_meta['ids'])}")
    
    logger.info(f"\n📂 Por categoría:")
    for cat, count in sorted(categorias.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"   {cat}: {count}")
    
    logger.info(f"\n📏 Unidades de medida:")
    for unidad, count in unidades.items():
        logger.info(f"   {unidad}: {count}")
    
    logger.info(f"\n💰 Fuentes de precio:")
    for fuente, count in fuentes_precio.items():
        logger.info(f"   {fuente}: {count}")
    
    logger.info(f"\n⚠️  Alertas de precios:")
    logger.info(f"   Precios en cero: {precios_cero}")
    logger.info(f"   Precios sospechosamente bajos (<$10): {precios_extremos['bajos']}")
    logger.info(f"   Precios extremadamente altos (>$50k): {precios_extremos['altos']}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Función principal de ingesta"""
    logger.info("🚀 INICIANDO SISTEMA DE INGESTA VAMA")
    
    # 1. Limpiar y preparar
    limpiar_db_anterior()
    
    # 2. Conectar a ChromaDB
    logger.info("🔮 Cargando modelo de embeddings BGE-M3...")
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-m3",
        device="cuda"
    )
    
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name="tiles_catalog_v3",
        embedding_function=embedding_func,
        metadata={
            "hnsw:space": "cosine",
            "descripcion": "Catálogo VAMA v3 con correcciones de precios",
            "fecha_creacion": datetime.now().isoformat()
        }
    )
    
    # 3. Procesar archivos
    total_productos = 0
    todos_errores = []
    
    archivos_csv = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv')]
    logger.info(f"📁 Encontrados {len(archivos_csv)} archivos CSV")
    
    for archivo in sorted(archivos_csv):
        filepath = os.path.join(DATA_PATH, archivo)
        count, errores = procesar_archivo(filepath, collection)
        total_productos += count
        todos_errores.extend(errores)
    
    # 4. Reporte final
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ INGESTA COMPLETADA")
    logger.info(f"{'='*60}")
    logger.info(f"Total productos indexados: {total_productos}")
    logger.info(f"Errores menores: {len(todos_errores)}")
    
    if todos_errores[:10]:
        logger.info("Primeros errores:")
        for err in todos_errores[:5]:
            logger.info(f"   - {err}")
    
    # 5. Reporte de calidad
    generar_reporte_calidad(collection)
    
    logger.info(f"\n📍 Base de datos: {CHROMA_PATH}")
    logger.info(f"🔍 Colección: tiles_catalog_v3")
    logger.info("🎉 Listo para usar!")

if __name__ == "__main__":
    main()