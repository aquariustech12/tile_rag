#!/usr/bin/env python3
"""
VAMA PRO - Asistente Inteligente con LLM + RAG
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import chromadb
from chromadb.utils import embedding_functions
import ollama
import re
import json
from typing import Dict, List, Optional

CHROMA_PATH = "chroma_db_v3"

# ============================================================================
# CONEXIÓN A DB (igual que antes)
# ============================================================================

print("🔌 Conectando...")
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-m3",
    device="cuda"
)
client = chromadb.PersistentClient(path=CHROMA_PATH)

cols = {
    "nacionales": client.get_collection("nacionales", embedding_function=embedding_func),
    "importados": client.get_or_create_collection("importados", embedding_function=embedding_func),
    "griferia": client.get_or_create_collection("griferia", embedding_function=embedding_func),
    "polvos": client.get_or_create_collection("polvos", embedding_function=embedding_func),
    "otras": client.get_or_create_collection("otras", embedding_function=embedding_func)
}

# Verificar Ollama
try:
    ollama.list()
    print("✅ Ollama listo")
except:
    print("❌ Ollama no responde")
    exit(1)

print(f"✅ {sum(c.count() for c in cols.values())} productos\n")

# ============================================================================
# CLASE DB (adaptada de tu código)
# ============================================================================

class DB:
    def __init__(self):
        self.cols = cols
    
    def buscar(self, query, colecciones=None, tipo=None, top_k=5):
        """Busca productos"""
        if colecciones is None:
            colecciones = ["nacionales", "importados"]
        
        resultados = []
        for nombre in colecciones:
            if nombre not in self.cols:
                continue
            
            col = self.cols[nombre]
            where_filter = {"tipo": tipo} if tipo and nombre in ["nacionales", "importados"] else None
            
            try:
                if where_filter:
                    r = col.query(query_texts=[query], n_results=top_k, where=where_filter)
                else:
                    r = col.query(query_texts=[query], n_results=top_k)
                
                for meta, dist, pid in zip(r["metadatas"][0], r["distances"][0], r["ids"][0]):
                    resultados.append({
                        "codigo": meta.get("codigo", ""),
                        "descripcion": meta.get("descripcion", ""),
                        "proveedor": meta.get("proveedor", ""),
                        "precio_m2": float(meta.get("precio_m2", 0)) if meta.get("precio_m2") else 0,
                        "precio_unitario": float(meta.get("precio_unitario", 0)) if meta.get("precio_unitario") else 0,
                        "metraje_caja": float(meta.get("metraje_caja", 1.44)) if meta.get("metraje_caja") else 1.44,
                        "formato": meta.get("formato", ""),
                        "color": meta.get("color", ""),
                        "es_promo": meta.get("es_promo", False),
                        "tipo": meta.get("tipo", ""),
                        "coleccion": nombre
                    })
            except:
                continue
        
        # Quitar duplicados
        vistos = set()
        unicos = []
        for r in resultados:
            if r["codigo"] not in vistos:
                vistos.add(r["codigo"])
                unicos.append(r)
        
        return unicos[:top_k]

# ============================================================================
# ESTADO DE CONVERSACIÓN (adaptado de tu código)
# ============================================================================

class EstadoConversacion:
    def __init__(self):
        self.productos_vistos = []
        self.producto_seleccionado = None
        self.m2_proyecto = None
        self.items_cotizacion = []
        self.ultima_categoria = None
        self.ultimos_productos = []
        self.categoria_activa = None
    
    def reset(self):
        self.__init__()
    
    def guardar_productos(self, productos):
        self.ultimos_productos = productos
    
    def seleccionar(self, idx):
        if 0 <= idx < len(self.ultimos_productos):
            self.producto_seleccionado = self.ultimos_productos[idx]
            return self.producto_seleccionado
        return None
    
    def agregar_item(self, producto, calculo, m2=None):
        item = {
            "producto": producto,
            "calculo": calculo,
            "m2": m2
        }
        self.items_cotizacion.append(item)
        if m2:
            self.m2_proyecto = m2
    
    def get_total(self):
        total = 0
        for item in self.items_cotizacion:
            total += item["calculo"].get("total", 0)
        return total
    
    def get_resumen(self):
        if not self.items_cotizacion:
            return "No hay items en la cotización."
        
        lineas = ["📋 RESUMEN DE COTIZACIÓN"]
        for i, item in enumerate(self.items_cotizacion, 1):
            prod = item["producto"]
            calc = item["calculo"]
            if "cajas" in calc:
                lineas.append(f"{i}. {prod['descripcion'][:30]} - {calc['cajas']} cajas = ${calc['total']:.2f}")
            else:
                lineas.append(f"{i}. {prod['descripcion'][:30]} - {calc.get('unidades', 1)} un = ${calc['total']:.2f}")
        
        lineas.append(f"\n💰 TOTAL: ${self.get_total():.2f}")
        return "\n".join(lineas)

ESTADO = EstadoConversacion()

# ============================================================================
# VAMA CON LLM - Versión mejorada
# ============================================================================

class VAMALLM:
    def __init__(self, db, estado):
        self.db = db
        self.estado = estado
        self.modelo = "qwen3:30b-a3b-fp16"  # Cambia si usas otro modelo
        self.historial = []
        
    def clasificar_intencion(self, mensaje: str) -> Dict:
        """Clasificador simple de intención"""
        m = mensaje.lower()
        
        # Detectar despedida
        if any(x in m for x in ["gracias", "adiós", "hasta luego", "eso es todo", "listo", "terminamos"]):
            return {"intencion": "despedida", "categoria": None, "es_complemento": False}
        
        # Detectar búsqueda
        if any(x in m for x in ["busco", "tienes", "opciones", "muéstrame", "muestrame", "ver", "catalogo"]):
            return {"intencion": "buscar", "categoria": self.detectar_categoria(m), "es_complemento": False}
        
        # Detectar cotización
        if any(x in m for x in ["cotizar", "precio", "cuanto", "cuesta", "m2", "m²"]) or re.search(r'\d+\s*(?:m2|m²)', m):
            return {"intencion": "cotizar", "categoria": self.detectar_categoria(m), "es_complemento": False}
        
        # Detectar agregar complemento
        if any(x in m for x in ["agrega", "tambien", "ademas", "y tambien", "también", "además"]) or \
           (self.estado.items_cotizacion and any(x in m for x in ["pegamento", "griferia", "grifo", "llave", "adhesivo"])):
            return {"intencion": "agregar", "categoria": self.detectar_categoria(m), "es_complemento": True}
        
        # Si no, es información general
        return {"intencion": "info", "categoria": self.detectar_categoria(m), "es_complemento": False}
    
    def detectar_categoria(self, mensaje: str) -> Optional[str]:
        """Detecta categoría del producto"""
        m = mensaje.lower()
        
        if any(x in m for x in ["piso", "porcelanato", "porcelánico", "loseta"]):
            return "pisos"
        elif any(x in m for x in ["muro", "azulejo", "pared", "revestimiento"]):
            return "muros"
        elif any(x in m for x in ["grifo", "llave", "monomando", "mezcladora", "regadera", "griferia"]):
            return "griferia"
        elif any(x in m for x in ["pegamento", "adhesivo", "cemento", "boquilla", "pega", "polvo"]):
            return "polvos"
        elif any(x in m for x in ["mueble", "espejo", "tinaco", "tarja", "lavabo"]):
            return "otras"
        
        return None
    
    def obtener_productos_rag(self, clas: Dict, mensaje: str) -> str:
        """Obtiene productos relevantes usando RAG"""
        categoria = clas.get('categoria')
        query = mensaje
        
        if not categoria:
            return ""
        
        # Buscar productos según categoría
        if categoria == "pisos":
            productos = self.db.buscar(query, tipo="piso", top_k=3)
        elif categoria == "muros":
            productos = self.db.buscar(query, tipo="muro", top_k=3)
        elif categoria in ["griferia", "polvos", "otras"]:
            productos = self.db.buscar(query, colecciones=[categoria], top_k=3)
        else:
            productos = self.db.buscar(query, top_k=3)
        
        if not productos:
            return ""
        
        # Formatear para el prompt
        info = "\nPRODUCTOS ENCONTRADOS:\n"
        for i, p in enumerate(productos, 1):
            if p['precio_m2'] > 0:
                precio_caja = p['precio_m2'] * p['metraje_caja']
                info += f"{i}. {p['descripcion'][:40]} - ${p['precio_m2']:.2f}/m² (${precio_caja:.2f}/caja)\n"
            else:
                info += f"{i}. {p['descripcion'][:40]} - ${p['precio_unitario']:.2f}/unidad\n"
        
        # Guardar productos en estado
        self.estado.ultimos_productos = productos
        self.estado.categoria_activa = categoria
        
        return info
    
    def calcular_cotizacion(self, producto, m2):
        """Calcula cajas necesarias"""
        import math
        
        if producto["precio_m2"] > 0:
            cajas = math.ceil(m2 / producto["metraje_caja"])
            m2_real = cajas * producto["metraje_caja"]
            precio_caja = producto["precio_m2"] * producto["metraje_caja"]
            total = cajas * precio_caja
            
            return {
                "cajas": cajas,
                "m2_real": m2_real,
                "precio_caja": precio_caja,
                "total": total,
                "precio_m2": producto["precio_m2"]
            }
        else:
            unidades = max(1, int(m2))
            return {
                "unidades": unidades,
                "total": unidades * producto["precio_unitario"],
                "precio_unitario": producto["precio_unitario"]
            }
    
    def generar_prompt(self, mensaje: str, clas: Dict, productos_info: str) -> str:
        """Genera prompt para el LLM"""
        
        # Contexto actual
        contexto = ""
        if self.estado.items_cotizacion:
            contexto += f"Cotización actual: {len(self.estado.items_cotizacion)} items, Total: ${self.estado.get_total():.2f}\n"
        
        if self.estado.m2_proyecto:
            contexto += f"Proyecto: {self.estado.m2_proyecto}m²\n"
        
        if self.estado.categoria_activa:
            contexto += f"Categoría activa: {self.estado.categoria_activa}\n"
        
        prompt = f"""Eres VAMA, experto en materiales de construcción (pisos, azulejos, grifería, pegamentos).

CONTEXTO ACTUAL:
{contexto.strip()}

INTENCIÓN DETECTADA: {clas['intencion']}
CATEGORÍA: {clas.get('categoria', 'no especificada')}
ES COMPLEMENTO: {clas.get('es_complemento', False)}

{productos_info if productos_info else ''}

HISTORIAL RECIENTE (últimos 3 mensajes):
{chr(10).join(self.historial[-3:]) if self.historial else 'Nueva conversación'}

CLIENTE: {mensaje}

INSTRUCCIONES:
1. Sé claro y directo, como el bot de Clip
2. Ofrece opciones numeradas cuando muestres productos
3. Si das información detallada, pregunta "¿Te sirvió esta información?" con opciones 1) Sí 2) No
4. Para cotizaciones, muestra cálculos claros
5. Usa emojis relevantes pero no exageres
6. Mantén el foco en materiales de construcción

EMOJIS POR CATEGORÍA:
- 🏠 Pisos/porcelanatos
- 🧱 Azulejos/muros
- 🚿 Grifería
- 🔧 Pegamentos/adhesivos
- 💰 Cotizaciones/presupuestos
- ❓ Preguntas/ayuda

RESPONDE EN ESPAÑOL, COMO VAMA:"""
        
        return prompt
    
    def procesar_mensaje(self, mensaje: str) -> str:
        """Procesa el mensaje usando LLM + RAG"""
        
        # Guardar en historial
        self.historial.append(f"Cliente: {mensaje}")
        
        # Clasificar intención
        clas = self.clasificar_intencion(mensaje)
        
        # Extraer medidas si las hay
        medidas = None
        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:m2|m²)', mensaje.lower())
        if match:
            medidas = float(match.group(1))
            self.estado.m2_proyecto = medidas
        
        # Obtener productos relevantes si es búsqueda o cotización
        productos_info = ""
        if clas['intencion'] in ['buscar', 'cotizar', 'agregar']:
            productos_info = self.obtener_productos_rag(clas, mensaje)
        
        # Manejar casos especiales primero
        if clas['intencion'] == 'despedida':
            if self.estado.items_cotizacion:
                total = self.estado.get_total()
                respuesta = f"¡Gracias por tu cotización! 🎉\n\nTotal: *${total:.2f}*\n\n¿Te gustaría:\n\n1) Recibir esta cotización por WhatsApp/email\n2) Guardarla para después\n3) Modificar algo\n4) Finalizar\n\nElige una opción (1-4):"
            else:
                respuesta = "¡Gracias por contactar a VAMA! Estamos aquí cuando lo necesites. 👋"
            
            self.historial.append(f"VAMA: {respuesta[:100]}...")
            return respuesta
        
        # Generar respuesta con LLM
        try:
            prompt = self.generar_prompt(mensaje, clas, productos_info)
            
            respuesta = ollama.generate(
                model=self.modelo,
                prompt=prompt,
                options={'temperature': 0.3}
            )['response'].strip()
            
            # Si la respuesta es larga, agregar pregunta de confirmación
            if len(respuesta) > 200 and clas['intencion'] in ['buscar', 'cotizar']:
                respuesta += "\n\n¿Te sirvió esta información?\n\n1) Sí\n2) No"
            
        except Exception as e:
            # Fallback a respuestas deterministas
            respuesta = self.fallback_determinista(mensaje, clas)
        
        # Guardar respuesta en historial
        self.historial.append(f"VAMA: {respuesta[:100]}...")
        
        return respuesta
    
    def fallback_determinista(self, mensaje: str, clas: Dict) -> str:
        """Respuesta de fallback cuando el LLM falla"""
        intencion = clas['intencion']
        
        if intencion == "buscar":
            cat = clas.get('categoria', 'productos')
            return f"🛍️ *{cat.upper()}*\n\nTe recomiendo buscar en nuestra tienda. ¿Qué tipo de {cat} te interesa?\n\nPuedo mostrarte opciones por:\n• Color\n• Precio\n• Estilo\n\n¿Qué prefieres?"
        
        elif intencion == "cotizar":
            return f"💰 *COTIZACIÓN*\n\nPara darte un precio exacto, necesito:\n1. Producto específico\n2. Metros cuadrados\n3. Color preferido\n\n¿Ya tienes algún producto en mente o prefieres que te muestre opciones?"
        
        elif intencion == "agregar":
            return f"➕ *AGREGAR COMPLEMENTO*\n\nPerfecto, puedo agregar complementos a tu cotización.\n\n¿Qué necesitas?\n1) 🔧 Pegamento/adhesivo\n2) 🚿 Grifería\n3) 🧰 Otros accesorios\n\nDime el número o descríbemelo:"
        
        else:
            return "🏪 ¡Hola! Soy VAMA, tu experto en materiales de construcción.\n\nPuedo ayudarte con:\n\n1) 🏠 Buscar pisos y porcelanatos\n2) 🧱 Ver azulejos para muros\n3) 🚿 Cotizar grifería\n4) 🔧 Agregar pegamentos y adhesivos\n5) 💰 Crear cotizaciones completas\n\n¿En qué puedo ayudarte hoy?"

# ============================================================================
# MAIN LOOP MEJORADO
# ============================================================================

def main_mejorado():
    print("="*60)
    print("🏪 VAMA PRO - Asistente Inteligente")
    print("="*60)
    print("¡Hola! Soy VAMA, tu experto en materiales de construcción.")
    print("Puedo ayudarte a buscar productos, crear cotizaciones y más.\n")
    print("Ejemplos:")
    print("- 'Necesito pisos blancos para 20m2'")
    print("- 'Muéstrame azulejos para baño'")
    print("- 'Agrega pegamento a mi cotización'")
    print("- 'Eso es todo' (para finalizar)")
    print("="*60 + "\n")
    
    # Inicializar
    db = DB()
    estado = EstadoConversacion()
    vama = VAMALLM(db, estado)
    
    while True:
        try:
            mensaje = input("👤 Cliente: ").strip()
            
            if mensaje.lower() in ["salir", "exit", "quit"]:
                print("\n🤖 ¡Gracias por visitar VAMA! Hasta pronto 👋")
                break
            
            if not mensaje:
                continue
            
            print("🤖 Pensando...")
            
            # Procesar mensaje
            respuesta = vama.procesar_mensaje(mensaje)
            
            # Mostrar respuesta
            print(f"\n🤖 VAMA:\n{respuesta}\n")
            
            # Manejar confirmaciones
            if "1) Sí" in respuesta and "2) No" in respuesta:
                while True:
                    confirmacion = input("👤 (1 o 2): ").strip()
                    if confirmacion == "1":
                        print("\n🤖 ¡Perfecto! ¿En qué más puedo ayudarte?")
                        break
                    elif confirmacion == "2":
                        print("\n🤖 ¿Qué parte no quedó clara? ¿Quieres que lo explique de otra forma?")
                        break
                    else:
                        print("🤖 Por favor elige 1) Sí o 2) No")
            
        except KeyboardInterrupt:
            print("\n🤖 ¡Hasta pronto! 👋")
            break
        except Exception as e:
            print(f"\n🤖 ❌ Error: {e}")
            print("🤖 Disculpa, tuve un problema. ¿Podrías repetir tu pregunta?")

if __name__ == "__main__":
    main_mejorado()