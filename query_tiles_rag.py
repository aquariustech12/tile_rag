#!/usr/bin/env python3
"""
VAMA - Agente Vendedor de Baños para WhatsApp
Cotizador inteligente con RAG (Retrieval Augmented Generation)
"""

import os
import sys
import json
import re
import math
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

# Configuración de logging para debugging en producción
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vama_agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suprimir warnings de transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import chromadb
from chromadb.utils import embedding_functions
import ollama

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

CHROMA_PATH = "chroma_db_v2"
MODEL_LLM = "qwen3:30b-a3b-fp16"
EMBEDDING_MODEL = "BAAI/bge-m3"

# Precios de referencia para validación (MXN)
PRECIOS_REFERENCIA = {
    'wc_one_piece': {'min': 800, 'max': 8000, 'default': 1499},
    'monomando_lavabo': {'min': 400, 'max': 5000, 'default': 899},
    'monomando_regadera': {'min': 350, 'max': 4000, 'default': 799},
    'adhesivo_20kg': {'min': 150, 'max': 400, 'default': 219},
    'piso_porcelanico': {'min': 50, 'max': 2000, 'default': 299},
    'muro_ceramico': {'min': 40, 'max': 800, 'default': 159}
}

# Factores de cálculo
FACTOR_MURO_PERIMETRO = 3.2  # Perímetro * altura estándar 2.4m menos 20% puertas/ventanas
RENDIMIENTO_ADHESIVO = 4.5   # m² por saco de 20kg

# ============================================================================
# ESTRUCTURAS DE DATOS
# ============================================================================

class CategoriaProducto(Enum):
    PISO = "piso"
    MURO = "muro"
    SANITARIO = "sanitario"
    GRIFERIA = "griferia"
    ACCESORIO = "accesorio"
    ADHESIVO = "adhesivo"

@dataclass
class Producto:
    id: str
    descripcion: str
    proveedor: str
    precio_m2: float
    metraje_caja: float
    formato: str
    categoria: str
    tipologia: str
    acabado: str
    color: str
    codigo: str
    tipo_producto: str
    score: float
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class CotizacionItem:
    nombre: str
    cantidad: int
    unidad: str
    precio_unitario: float
    nota: str = ""
    categoria: str = ""
    sku: str = ""
    
    @property
    def total(self) -> float:
        return self.cantidad * self.precio_unitario
    
    def to_dict(self) -> Dict:
        return {
            'nombre': self.nombre,
            'cantidad': self.cantidad,
            'unidad': self.unidad,
            'precio_unitario': round(self.precio_unitario, 2),
            'total': round(self.total, 2),
            'nota': self.nota,
            'categoria': self.categoria,
            'sku': self.sku
        }

@dataclass
class DimensionesBaño:
    largo: float
    ancho: float
    alto: float = 2.4
    
    @property
    def m2_piso(self) -> float:
        return self.largo * self.ancho
    
    @property
    def m2_muro(self) -> float:
        """Calcula m² reales de muro (perímetro * alto - 20% huecos)"""
        perimetro = 2 * (self.largo + self.ancho)
        m2_total = perimetro * self.alto
        return m2_total * 0.8  # -20% para puertas, ventanas, etc.

# ============================================================================
# CONEXIÓN A BASE DE DATOS VECTORIAL
# ============================================================================

class ChromaDBManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        logger.info("🔌 Inicializando conexión a ChromaDB...")
        
        try:
            self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL,
                device="cuda"
            )
            
            self.client = chromadb.PersistentClient(path=CHROMA_PATH)
            self.collection = self.client.get_collection(
                name="tiles_catalog_v2",
                embedding_function=self.embedding_func
            )
            
            count = self.collection.count()
            logger.info(f"✅ Conectado. {count} productos en catálogo")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"❌ Error conectando a ChromaDB: {e}")
            raise

# ============================================================================
# AGENTE VENDEDOR VAMA
# ============================================================================

class VAMAAgent:
    def __init__(self):
        self.db = ChromaDBManager()
        self.m2_proyecto: Optional[float] = None
        self.dimensiones: Optional[DimensionesBaño] = None
        self.setup_completo: bool = False
        self.color_preferido: str = "blanco"
        self.presupuesto_max: Optional[float] = None
        self.historial_busquedas: List[Dict] = []
        
    # -------------------------------------------------------------------------
    # EXTRACCIÓN DE PARÁMETROS DEL USUARIO
    # -------------------------------------------------------------------------
    
    def extraer_dimensiones(self, texto: str) -> Optional[DimensionesBaño]:
        """Extrae dimensiones del baño con múltiples patrones"""
        texto_lower = texto.lower()
        
        # Patrón 1: "2x3 metros" o "2 x 3 m"
        patron_dim = r'(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*(?:m|metros|mt)'
        match = re.search(patron_dim, texto_lower)
        if match:
            largo = float(match.group(1))
            ancho = float(match.group(2))
            # Detectar altura opcional "x 2.4 alto"
            alt_match = re.search(r'(?:x\s*(\d+(?:\.\d+)?)\s*(?:alto|height))', texto_lower)
            alto = float(alt_match.group(1)) if alt_match else 2.4
            return DimensionesBaño(largo, ancho, alto)
        
        # Patrón 2: "8m2", "8 m2", "8 metros cuadrados"
        patron_m2 = r'(\d+(?:\.\d+)?)\s*(?:m2|m²|metros?\s+cuadrados?|mt2)'
        match = re.search(patron_m2, texto_lower)
        if match:
            m2 = float(match.group(1))
            # Asumimos baño rectangular típico, estimamos lados
            ancho = math.sqrt(m2 / 2)  # Proporción 2:1
            largo = m2 / ancho
            return DimensionesBaño(round(largo, 2), round(ancho, 2))
        
        return None
    
    def extraer_color(self, texto: str) -> str:
        """Extrae preferencia de color del texto"""
        colores = {
            'blanco': ['blanco', 'white', 'blanca'],
            'beige': ['beige', ' hueso', 'marfil', 'ivory'],
            'gris': ['gris', 'gray', 'grises', 'concreto'],
            'negro': ['negro', 'black', 'negros'],
            'madera': ['madera', 'wood', 'roble', 'nogal'],
            'azul': ['azul', 'blue', 'turquesa'],
            'verde': ['verde', 'green', 'esmeralda']
        }
        
        texto_lower = texto.lower()
        for color, keywords in colores.items():
            if any(k in texto_lower for k in keywords):
                return color
        return "blanco"  # Default
    
    def detectar_setup_completo(self, texto: str) -> bool:
        """Detecta si el usuario quiere cotización completa"""
        keywords = [
            "setup completo", "todo el baño", "baño completo", 
            "remodelar baño", "hacer mi baño", "equipar baño",
            "cotizar todo", "todo incluido", "llave en mano"
        ]
        return any(k in texto.lower() for k in keywords)
    
    def extraer_presupuesto(self, texto: str) -> Optional[float]:
        """Extrae presupuesto máximo mencionado"""
        patrones = [
            r'(?:presupuesto|máximo|max|hasta|tope)\s*(?:de\s*)?\$?\s*(\d{1,6}(?:,\d{3})*(?:\.\d{2})?)',
            r'\$?\s*(\d{1,6}(?:,\d{3})*(?:\.\d{2})?)\s*(?:pesos|mxn)?\s*(?:máximo|max|tope)'
        ]
        for patron in patrones:
            match = re.search(patron, texto.lower())
            if match:
                monto_str = match.group(1).replace(',', '')
                return float(monto_str)
        return None

    # -------------------------------------------------------------------------
    # BÚSQUEDA Y VALIDACIÓN DE PRODUCTOS
    # -------------------------------------------------------------------------
    
    def corregir_metraje(self, valor) -> float:
        """Corrige metraje de caja si viene en formato incorrecto"""
        try:
            m = float(valor)
            # Si el valor es > 100, probablemente sea cm² o error
            if m > 100:
                return m / 10000  # Convertir cm² a m²
            if m > 10:
                return m / 100   # Convertir cm a m (por si acaso)
            return m
        except (ValueError, TypeError):
            return 1.44  # Default común 60x60
    
    def validar_precio(self, precio: float, categoria: str) -> Tuple[float, bool]:
        """
        Valida que el precio esté en rangos razonables.
        Retorna (precio_corregido, es_valido)
        """
        refs = PRECIOS_REFERENCIA.get(categoria, {'min': 1, 'max': 999999, 'default': 100})
        
        if precio < refs['min'] or precio > refs['max']:
            logger.warning(f"Precio sospechoso ${precio} para {categoria}. Usando default ${refs['default']}")
            return refs['default'], False
        return precio, True
    
    def buscar(self, query: str, where_filter: Dict = None, top_k: int = 5, categoria_hint: str = None) -> List[Producto]:
        """
        Búsqueda vectorial con fallback y logging
        """
        try:
            logger.info(f"🔍 Buscando: '{query}' | Filtro: {where_filter}")
            
            # Intentar búsqueda con filtro
            results = self.db.collection.query(
                query_texts=[query], 
                n_results=top_k*3,  # Pedir más para filtrar después
                where=where_filter
            )
            
            productos = []
            for doc, meta, dist, pid in zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0], 
                results['ids'][0]
            ):
                try:
                    # Extraer y corregir metraje
                    metraje_raw = meta.get('metraje_caja', 1.44)
                    metraje = self.corregir_metraje(metraje_raw)
                    
                    # Extraer precio y validar
                    precio_raw = float(meta.get('precio_m2', 0) or 0)
                    categoria = meta.get('categoria', 'desconocido')
                    precio, _ = self.validar_precio(precio_raw, categoria)
                    
                    p = Producto(
                        id=pid,
                        descripcion=meta.get('descripcion', 'Sin descripción'),
                        proveedor=meta.get('proveedor', 'Sin proveedor'),
                        precio_m2=precio,
                        metraje_caja=metraje,
                        formato=meta.get('formato', 'N/A') or 'N/A',
                        categoria=categoria,
                        tipologia=meta.get('tipologia', 'N/A'),
                        acabado=meta.get('acabado', 'Estándar') or 'Estándar',
                        color=meta.get('color', 'N/A'),
                        codigo=meta.get('codigo', 'N/A'),
                        tipo_producto=meta.get('tipo_producto', 'nacional'),
                        score=1 - (dist / 2)
                    )
                    
                    if p.precio_m2 > 0:
                        productos.append(p)
                        
                except Exception as e:
                    logger.error(f"Error procesando producto {pid}: {e}")
                    continue
            
            # Ordenar por score y retornar top_k
            productos.sort(key=lambda x: x.score, reverse=True)
            resultado = productos[:top_k]
            
            logger.info(f"✅ Encontrados {len(resultado)} productos válidos")
            return resultado
            
        except Exception as e:
            logger.error(f"❌ Error en búsqueda: {e}")
            return []

    # -------------------------------------------------------------------------
    # CÁLCULOS DE COTIZACIÓN
    # -------------------------------------------------------------------------
    
    def calcular_piso(self, p: Producto, m2: float) -> CotizacionItem:
        """Calcula cotización de piso con matemáticas exactas"""
        cajas = math.ceil(m2 / p.metraje_caja)
        m2_reales = round(cajas * p.metraje_caja, 2)
        precio_caja = round(p.precio_m2 * p.metraje_caja, 2)
        total = round(cajas * precio_caja, 2)
        
        return CotizacionItem(
            nombre=f"{p.descripcion}",
            cantidad=cajas,
            unidad=f"cajas ({p.metraje_caja}m² c/u)",
            precio_unitario=precio_caja,
            nota=f"Precio por m²: ${p.precio_m2:.2f} | Cubre {m2_reales}m²",
            categoria="piso",
            sku=p.codigo
        )
    
    def calcular_muro(self, m2: float, color: str = "blanco", estilo_piso: str = "") -> Optional[CotizacionItem]:
        """
        Busca y calcula muro coordinado con el piso
        """
        # Construir query inteligente según estilo
        query_terms = [color, "muro", "baño"]
        if "mate" in estilo_piso.lower():
            query_terms.append("mate")
        elif "brillante" in estilo_piso.lower() or "pulido" in estilo_piso.lower():
            query_terms.append("brillante")
        
        query = " ".join(query_terms)
        
        muros = self.buscar(
            query, 
            where_filter={"categoria": "muro"}, 
            top_k=3
        )
        
        if not muros:
            # Fallback: buscar cualquier muro blanco
            muros = self.buscar(
                "azulejo blanco muro", 
                where_filter={"categoria": "muro"}, 
                top_k=1
            )
        
        if not muros:
            return None
        
        m = muros[0]  # Tomar el mejor match
        cajas = math.ceil(m2 / m.metraje_caja)
        precio_caja = round(m.precio_m2 * m.metraje_caja, 2)
        
        return CotizacionItem(
            nombre=f"{m.descripcion}",
            cantidad=cajas,
            unidad=f"cajas ({m.metraje_caja}m²)",
            precio_unitario=precio_caja,
            nota=f"Precio por m²: ${m.precio_m2:.2f} | Color: {m.color}",
            categoria="muro",
            sku=m.codigo
        )
    
    def calcular_pegamento(self, m2_total: float) -> CotizacionItem:
        """Calcula adhesivo necesario"""
        sacos = math.ceil(m2_total / RENDIMIENTO_ADHESIVO)
        precio_ref = PRECIOS_REFERENCIA['adhesivo_20kg']['default']
        
        return CotizacionItem(
            nombre="Adhesivo Porcelánico Gris 20kg",
            cantidad=sacos,
            unidad="sacos de 20kg",
            precio_unitario=precio_ref,
            nota=f"Rendimiento: ~{RENDIMIENTO_ADHESIVO}m² por saco | Total m²: {round(m2_total, 2)}",
            categoria="adhesivo",
            sku="ADH-CEMIX-20K"
        )
    
    def buscar_sanitario(self, tipo: str = "one piece") -> Optional[CotizacionItem]:
        """Busca WC con validación de precios"""
        query = f"wc {tipo} blanco"
        sans = self.buscar(query, where_filter={"categoria": "sanitario"}, top_k=2)
        
        if not sans:
            # Fallback a búsqueda general
            sans = self.buscar("inodoro blanco", where_filter={"categoria": "sanitario"}, top_k=1)
        
        if not sans:
            # Crear item con precio de referencia si no hay en DB
            precio = PRECIOS_REFERENCIA['wc_one_piece']['default']
            return CotizacionItem(
                nombre="WC One Piece Estándar (Consultar modelo)",
                cantidad=1,
                unidad="pieza",
                precio_unitario=precio,
                nota="Incluye taza y tanque integrados. Confirmar modelo disponible.",
                categoria="sanitario",
                sku="POR-CONFIRMAR"
            )
        
        s = sans[0]
        # Validar precio - en sanitarios precio_m2 es precio unitario
        precio, es_valido = self.validar_precio(s.precio_m2, 'wc_one_piece')
        
        if not es_valido:
            logger.warning(f"Precio de sanitario corregido: ${s.precio_m2} -> ${precio}")
        
        return CotizacionItem(
            nombre=f"{s.descripcion}",
            cantidad=1,
            unidad="pieza",
            precio_unitario=precio,
            nota="Incluye taza, tanque integrados y asiento",
            categoria="sanitario",
            sku=s.codigo
        )
    
    def buscar_mezcladora(self, tipo: str = "lavabo") -> Optional[CotizacionItem]:
        """Busca grifería monomando"""
        queries = {
            'lavabo': "monomando lavabo manija",
            'regadera': "monomando regadera ducha",
            'cocina': "monomando fregadero cocina"
        }
        
        query = queries.get(tipo, f"monomando {tipo}")
        grif = self.buscar(query, where_filter={"categoria": "griferia"}, top_k=2)
        
        if not grif:
            # Precio de referencia
            pref_key = 'monomando_lavabo' if tipo == 'lavabo' else 'monomando_regadera'
            precio = PRECIOS_REFERENCIA[pref_key]['default']
            return CotizacionItem(
                nombre=f"Monomando para {tipo.title()} (Consultar modelo)",
                cantidad=1,
                unidad="pieza",
                precio_unitario=precio,
                nota="Incluye instalación básica y flexibles",
                categoria="griferia",
                sku="POR-CONFIRMAR"
            )
        
        g = grif[0]
        precio, es_valido = self.validar_precio(g.precio_m2, f'monomando_{tipo}')
        
        return CotizacionItem(
            nombre=f"{g.descripcion}",
            cantidad=1,
            unidad="pieza",
            precio_unitario=precio,
            nota=f"Incluye instalación básica | Marca: {g.proveedor}",
            categoria="griferia",
            sku=g.codigo
        )

    # -------------------------------------------------------------------------
    # GENERACIÓN DE COTIZACIÓN COMPLETA
    # -------------------------------------------------------------------------
    
    def generar_cotizacion_completa(self, dimensiones: DimensionesBaño) -> List[Dict]:
        """
        Genera cotización con 3 opciones de piso coordinadas
        """
        m2_piso = dimensiones.m2_piso
        m2_muro = dimensiones.m2_muro
        
        logger.info(f"Generando cotización para {m2_piso}m² piso, {m2_muro}m² muro")
        
        # Buscar 3 opciones de piso en el color preferido
        query_piso = f"piso {self.color_preferido} baño porcelanico"
        pisos = self.buscar(
            query_piso, 
            where_filter={"categoria": "piso"}, 
            top_k=3
        )
        
        if len(pisos) < 3:
            # Completar con opciones sin filtro de color
            adicionales = self.buscar(
                "piso porcelanico baño", 
                where_filter={"categoria": "piso"}, 
                top_k=5
            )
            # Evitar duplicados
            existentes_ids = {p.id for p in pisos}
            for p in adicionales:
                if p.id not in existentes_ids and len(pisos) < 3:
                    pisos.append(p)
        
        cotizaciones = []
        
        for p in pisos:
            items = []
            
            # 1. PISO
            item_piso = self.calcular_piso(p, m2_piso)
            items.append(item_piso)
            
            # 2. MURO (coordinado con el color del piso)
            item_muro = self.calcular_muro(
                m2_muro, 
                color=self.color_preferido,
                estilo_piso=p.acabado
            )
            if item_muro:
                items.append(item_muro)
            
            # 3. PEGAMENTO (piso + muro)
            m2_total_piso = item_piso.cantidad * p.metraje_caja
            m2_total_muro = item_muro.cantidad * 1.68 if item_muro else 0  # Metraje típico muro
            m2_total = m2_total_piso + m2_total_muro
            items.append(self.calcular_pegamento(m2_total))
            
            # 4. SANITARIO (solo si setup completo)
            if self.setup_completo:
                san = self.buscar_sanitario()
                if san:
                    items.append(san)
                
                # 5. GRIFERÍA
                grif_lav = self.buscar_mezcladora("lavabo")
                if grif_lav:
                    items.append(grif_lav)
                
                grif_reg = self.buscar_mezcladora("regadera")
                if grif_reg:
                    items.append(grif_reg)
                
                # 6. ACCESORIOS OPCIONALES (solo en opción premium)
                if p.precio_m2 > 400:  # Si es piso caro, agregar accesorios
                    items.append(CotizacionItem(
                        nombre="Kit de accesorios (Toallero, jabonera, porta papel)",
                        cantidad=1,
                        unidad="kit",
                        precio_unitario=450,
                        nota="Acero inoxidable o cromo según disponibilidad",
                        categoria="accesorio",
                        sku="KIT-ACC-BASIC"
                    ))
            
            # Calcular totales
            subtotal = sum(i.total for i in items)
            
            cotizaciones.append({
                "opcion_num": len(cotizaciones) + 1,
                "piso": p,
                "items": items,
                "subtotal": round(subtotal, 2),
                "dimensiones": {
                    "m2_piso": round(m2_piso, 2),
                    "m2_muro": round(m2_muro, 2),
                    "alto": dimensiones.alto
                }
            })
        
        return cotizaciones

    # -------------------------------------------------------------------------
    # FORMATO DE RESPUESTAS
    # -------------------------------------------------------------------------
    
    def formatear_respuesta_texto(self, cotizaciones: List[Dict], dimensiones: DimensionesBaño) -> str:
        """Formato bonito para WhatsApp/CLI"""
        lineas = []
        lineas.append(f"🛁 *COTIZACIÓN VAMA - BAÑO {dimensiones.largo}×{dimensiones.ancho}m*")
        lineas.append(f"📐 Superficie: {dimensiones.m2_piso}m² piso | {dimensiones.m2_muro:.1f}m² muros")
        lineas.append("=" * 50)
        
        for cot in cotizaciones[:3]:
            p = cot["piso"]
            lineas.append(f"\n💎 *OPCIÓN {cot['opcion_num']}: {p.descripcion}*")
            lineas.append(f"🏢 {p.proveedor} | 🔖 {p.codigo}")
            lineas.append(f"📏 Formato: {p.formato} | ✨ Acabado: {p.acabado}")
            lineas.append("")
            
            for item in cot["items"]:
                emoji = {
                    'piso': '🟫', 'muro': '⬜', 'adhesivo': '🪣',
                    'sanitario': '🚽', 'griferia': '🚿', 'accesorio': '🧴'
                }.get(item.categoria, '•')
                
                lineas.append(f"{emoji} *{item.nombre}*")
                lineas.append(f"   {item.cantidad} {item.unidad}")
                lineas.append(f"   💰 ${item.precio_unitario:,.2f} c/u = ${item.total:,.2f}")
                if item.nota:
                    lineas.append(f"   📝 {item.nota}")
                lineas.append("")
            
            lineas.append(f"💵 *SUBTOTAL OPCIÓN {cot['opcion_num']}: ${cot['subtotal']:,.2f}*")
            lineas.append("─" * 50)
        
        # Comparativo
        lineas.append(f"\n📊 *COMPARATIVO RÁPIDO:*")
        for cot in cotizaciones[:3]:
            lineas.append(f"   Opción {cot['opcion_num']}: ${cot['subtotal']:,.2f}")
        
        # Sugerencias contextuales
        lineas.append(f"\n💡 *SIGUIENTES PASOS:*")
        lineas.append(f"   1️⃣ Elige una opción (1, 2 o 3)")
        lineas.append(f"   2️⃣ ¿Necesitas instalación? (+15-20%)")
        lineas.append(f"   3️⃣ ¿Deseas agregar mueble con lavabo?")
        
        if self.setup_completo:
            lineas.append(f"\n⚠️ *Nota:* Los precios de sanitarios y grifería son referenciales.")
            lineas.append(f"   Confirmaremos modelo exacto según existencias.")
        
        lineas.append(f"\n🤔 *PREGUNTAS PARA PERSONALIZAR:*")
        lineas.append(f"   • ¿Tienes preferencia de marca en grifería?")
        lineas.append(f"   • ¿El baño tiene ventana o necesitas extractor?")
        lineas.append(f"   • ¿Requieres tinaco o cisterna?")
        
        return "\n".join(lineas)
    
    def formatear_respuesta_json(self, cotizaciones: List[Dict]) -> Dict:
        """Formato estructurado para API/WhatsApp Business"""
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "tipo_cotizacion": "baño_completo" if self.setup_completo else "materiales",
            "dimensiones": {
                "m2_piso": self.dimensiones.m2_piso if self.dimensiones else None,
                "m2_muro": self.dimensiones.m2_muro if self.dimensiones else None
            },
            "opciones": [
                {
                    "numero": cot['opcion_num'],
                    "nombre_opcion": cot['piso'].descripcion,
                    "proveedor": cot['piso'].proveedor,
                    "sku_piso": cot['piso'].codigo,
                    "items": [item.to_dict() for item in cot['items']],
                    "subtotal": cot['subtotal'],
                    "moneda": "MXN"
                }
                for cot in cotizaciones
            ],
            "metadata": {
                "total_opciones": len(cotizaciones),
                "color_seleccionado": self.color_preferido,
                "setup_completo": self.setup_completo
            }
        }

    # -------------------------------------------------------------------------
    # PROCESAMIENTO DE CONSULTAS
    # -------------------------------------------------------------------------
    
    def procesar_mensaje(self, mensaje: str, formato_salida: str = "texto") -> Dict:
        """
        Punto de entrada principal para mensajes de clientes
        Retorna dict con 'respuesta' y 'data'
        """
        logger.info(f"📩 Mensaje recibido: {mensaje[:100]}...")
        
        # Extraer parámetros
        self.setup_completo = self.detectar_setup_completo(mensaje)
        self.color_preferido = self.extraer_color(mensaje)
        self.presupuesto_max = self.extraer_presupuesto(mensaje)
        
        dimensiones = self.extraer_dimensiones(mensaje)
        
        if not dimensiones:
            return {
                "respuesta": (
                    "¡Hola! 👋 Soy VAMA, tu asesor de baños.\n\n"
                    "Para cotizar necesito saber:\n"
                    "📐 ¿Qué tamaño tiene tu baño?\n"
                    "   • Ejemplos: '8m2', '2x3 metros', '2.5x1.8m'\n\n"
                    "Opcionalmente dime:\n"
                    "🎨 ¿Qué color prefieres? (blanco, beige, gris...)\n"
                    "💰 ¿Tienes presupuesto máximo?"
                ),
                "data": None,
                "requiere_info": True
            }
        
        self.dimensiones = dimensiones
        self.m2_proyecto = dimensiones.m2_piso
        
        # Generar cotización
        try:
            cotizaciones = self.generar_cotizacion_completa(dimensiones)
            
            if not cotizaciones:
                return {
                    "respuesta": (
                        "😔 No encontré productos que coincidan con tu búsqueda.\n"
                        "¿Podrías intentar con otro color o tamaño?\n"
                        "También puedo mostrarte opciones en otros tonos."
                    ),
                    "data": None,
                    "error": "sin_productos"
                }
            
            # Formatear salida
            if formato_salida == "json":
                respuesta_texto = self.formatear_respuesta_json(cotizaciones)
            else:
                respuesta_texto = self.formatear_respuesta_texto(cotizaciones, dimensiones)
            
            # Opcional: Mejorar con LLM si está disponible
            if formato_salida == "texto":
                respuesta_texto = self._mejorar_con_llm(respuesta_texto)
            
            return {
                "respuesta": respuesta_texto,
                "data": self.formatear_respuesta_json(cotizaciones) if formato_salida != "json" else respuesta_texto,
                "requiere_info": False
            }
            
        except Exception as e:
            logger.error(f"Error generando cotización: {e}", exc_info=True)
            return {
                "respuesta": (
                    "⚠️ Hubo un problema generando tu cotización.\n"
                    "Por favor intenta de nuevo o contacta a un asesor humano."
                ),
                "data": None,
                "error": str(e)
            }
    
    def _mejorar_con_llm(self, texto_cotizacion: str) -> str:
        """Opcional: Usa LLM para humanizar sin cambiar números"""
        try:
            prompt = f"""Eres VAMA, un vendedor experto y amigable de una tienda de acabados.
REESCRIBE la siguiente cotización manteniendo EXACTAMENTE:
- Todos los precios y totales
- Las cantidades de productos
- Los códigos SKU
- Las dimensiones

Solo mejora el tono para que sea más cálido y profesional, como un vendedor experimentado atendiendo por WhatsApp. Usa emojis apropiados.

COTIZACIÓN ORIGINAL:
{texto_cotizacion}

REESCRITA (solo mejora el estilo, nunca los números):"""

            resp = ollama.chat(
                model=MODEL_LLM,
                messages=[
                    {"role": "system", "content": "Eres un vendedor profesional de VAMA. NUNCA cambies precios o cantidades."},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.2, "num_predict": 2000}
            )
            
            return resp['message']['content']
            
        except Exception as e:
            logger.warning(f"No se pudo mejorar con LLM: {e}")
            return texto_cotizacion

# ============================================================================
# INTERFAZ DE USUARIO
# ============================================================================

def chat_cli():
    """Modo interactivo para pruebas en terminal"""
    print("\n" + "="*60)
    print("🏪 VAMA - Cotizador de Baños (Modo CLI)")
    print("="*60)
    print("Comandos especiales:")
    print("  'json' - Cambiar formato de salida a JSON")
    print("  'texto' - Cambiar formato a texto (default)")
    print("  'salir' - Terminar sesión")
    print("="*60 + "\n")
    
    agente = VAMAAgent()
    formato = "texto"
    
    while True:
        try:
            user_input = input("👤 Cliente: ").strip()
            
            if user_input.lower() in ['salir', 'exit', 'quit']:
                print("🤖 ¡Gracias por usar VAMA! Hasta pronto. 👋")
                break
            
            if user_input.lower() == 'json':
                formato = "json"
                print("🤖 Formato cambiado a JSON")
                continue
            
            if user_input.lower() == 'texto':
                formato = "texto"
                print("🤖 Formato cambiado a Texto")
                continue
            
            if not user_input:
                continue
            
            print("🤖 Calculando...")
            resultado = agente.procesar_mensaje(user_input, formato_salida=formato)
            
            if formato == "json":
                print(f"\n🤖 VAMA (JSON):\n{json.dumps(resultado, indent=2, ensure_ascii=False)}\n")
            else:
                print(f"\n🤖 VAMA:\n{resultado['respuesta']}\n")
                
        except KeyboardInterrupt:
            print("\n\n🤖 ¡Hasta luego! 👋")
            break
        except Exception as e:
            logger.error(f"Error en CLI: {e}")
            print(f"❌ Error: {e}")

# ============================================================================
# INTEGRACIÓN WEB/API (Preparado para FastAPI/Flask)
# ============================================================================

def crear_api_handler():
    """
    Ejemplo de uso con FastAPI:
    
    from fastapi import FastAPI
    app = FastAPI()
    agente = VAMAAgent()
    
    @app.post("/webhook/whatsapp")
    async def whatsapp_webhook(request: Request):
        data = await request.json()
        mensaje = data.get('message', '')
        resultado = agente.procesar_mensaje(mensaje, formato_salida="texto")
        return {"reply": resultado['respuesta']}
    """
    pass

# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    # Verificar dependencias
    try:
        import chromadb
        import ollama
    except ImportError as e:
        print(f"❌ Falta dependencia: {e}")
        print("Instala con: pip install chromadb ollama")
        sys.exit(1)
    
    chat_cli()