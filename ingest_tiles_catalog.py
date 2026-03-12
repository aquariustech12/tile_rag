#!/usr/bin/env python3
"""
VAMA - Ingesta respetando nombres originales del catálogo
Versión con colecciones separadas: lavabos, sanitarios, muebles, etc.
"""

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os
import argparse

CHROMA_PATH = "chroma_db_v3"
EMBEDDING_MODEL = "BAAI/bge-m3"

print("🔌 Conectando a ChromaDB...")
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL,
    device="cpu"
)
client = chromadb.PersistentClient(path=CHROMA_PATH)

def limpiar_precio(valor):
    """Convierte '$2,099.00' → 2099.0"""
    if pd.isna(valor) or valor in ["", " ", "$-   ", "-", "$ -"]:
        return 0.0
    s = str(valor).replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except:
        return 0.0

def normalizar_texto(texto):
    return str(texto).strip().lower() if pd.notna(texto) else ""

def es_muro(descripcion):
    """Detecta si es muro por el nombre"""
    desc = str(descripcion).upper()
    return "MURO" in desc or "AZULEJO" in desc or "FACHALETA" in desc

# ============================================================================
# NACIONALES (pisos y muros)
# ============================================================================

def ingest_nacionales(filepath):
    print(f"\n📦 Ingestando NACIONALES: {filepath}")
    
    try:
        client.delete_collection(name="nacionales")
    except:
        pass
    
    collection = client.create_collection(
        name="nacionales",
        embedding_function=embedding_func
    )
    
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    
    productos = []
    for _, row in df.iterrows():
        try:
            precio_oferta = limpiar_precio(row.get("Oferta Final M2 ", 0))
            precio_regular = limpiar_precio(row.get("Precio Regular M2", 0))
            precio_final = precio_oferta if precio_oferta > 0 else precio_regular
            
            if precio_final == 0:
                continue
            
            metraje = float(row.get("Metraje", 1.44))
            if metraje > 100:
                metraje = metraje / 100
            
            tipo = "muro" if es_muro(row.get("Descripcion", "")) else "piso"
            
            doc = f"{row['Descripcion']} {row.get('Color', '')} ${precio}"
            
            metadata = {
                "codigo": str(row.get("Codigo", "")),
                "descripcion": f"{row.get('Descripcion', '')} ${precio}",
                "proveedor": str(row.get("Proveedor", "")),
                "codigo": str(row.get("Codigo", "")),
                "descripcion": str(row.get("Descripcion", "")),
                "proveedor": str(row.get("Proveedor", "")),
                "tipo": tipo,
                "precio_m2": precio_final,
                "precio_caja": precio_final * metraje,
                "metraje_caja": metraje,
                "formato": str(row.get("Formato", "")),
                "color": normalizar_texto(row.get("Color", "")),
                "tipologia": normalizar_texto(row.get("Tipologia", "")),
                "acabado": normalizar_texto(row.get("Acabado", "")),
                "corte": str(row.get("Corte", "")),
                "es_promo": False
            }
            
            productos.append({
                "id": f"nac_{row.get('Codigo', '')}_{_}",
                "document": doc,
                "metadata": metadata
            })
            
        except Exception as e:
            continue
    
    if productos:
        for i in range(0, len(productos), 100):
            batch = productos[i:i+100]
            collection.add(
                ids=[p["id"] for p in batch],
                documents=[p["document"] for p in batch],
                metadatas=[p["metadata"] for p in batch]
            )
        print(f"   ✅ {len(productos)} productos en 'nacionales'")
    
    return "nacionales", len(productos)

# ============================================================================
# IMPORTADOS
# ============================================================================

def ingest_importados(filepath):
    print(f"\n📦 Ingestando IMPORTADOS: {filepath}")
    
    try:
        client.delete_collection(name="importados")
    except:
        pass
    
    collection = client.create_collection(
        name="importados",
        embedding_function=embedding_func
    )
    
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    
    productos = []
    for _, row in df.iterrows():
        try:
            precio_oferta = limpiar_precio(row.get("Oferta Final M2 ", 0))
            precio_regular = limpiar_precio(row.get("Precio Regular M2", 0))
            precio_final = precio_oferta if precio_oferta > 0 else precio_regular
            
            if precio_final == 0:
                continue
            
            metraje = float(row.get("Metraje", 1.44))
            if metraje > 100:
                metraje = metraje / 100
            
            tipo = "muro" if es_muro(row.get("Descripcion", "")) else "piso"
            
            doc = f"{row['Descripcion']} {row.get('Color', '')} {row.get('Tipologia', '')}"
            
            metadata = {
                "codigo": str(row.get("Codigo", "")),
                "descripcion": str(row.get("Descripcion", "")),
                "proveedor": str(row.get("Proveedor", "")),
                "tipo": tipo,
                "precio_m2": precio_final,
                "precio_caja": precio_final * metraje,
                "metraje_caja": metraje,
                "formato": str(row.get("Formato", "")),
                "color": normalizar_texto(row.get("Color", "")),
                "tipologia": normalizar_texto(row.get("Tipologia", "")),
                "acabado": normalizar_texto(row.get("Acabado", "")),
                "es_promo": False
            }
            
            productos.append({
                "id": f"imp_{row.get('Codigo', '')}_{_}",
                "document": doc,
                "metadata": metadata
            })
            
        except Exception as e:
            continue
    
    if productos:
        for i in range(0, len(productos), 100):
            batch = productos[i:i+100]
            collection.add(
                ids=[p["id"] for p in batch],
                documents=[p["document"] for p in batch],
                metadatas=[p["metadata"] for p in batch]
            )
        print(f"   ✅ {len(productos)} productos en 'importados'")
    
    return "importados", len(productos)

# ============================================================================
# GRIFERIA PURA (solo llaves, monomandos, regaderas)
# ============================================================================

def ingest_griferia(filepath):
    print(f"\n📦 Ingestando GRIFERIA PURA: {filepath}")
    
    try:
        client.delete_collection(name="griferia")
    except:
        pass
    
    collection = client.create_collection(
        name="griferia",
        embedding_function=embedding_func
    )
    
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    
    productos = []
    for _, row in df.iterrows():
        try:
            precio = limpiar_precio(row.get("Precio autorizado", 0))
            if precio == 0:
                precio = limpiar_precio(row.get("Precio sistema", 0))
            
            if precio == 0:
                continue
            
            doc = f"{row.get('Descripcion', '')} {row.get('Categoria', '')}"
            
            metadata = {
                "codigo": str(row.get("Codigo", "")),
                "descripcion": str(row.get("Descripcion", "")),
                "proveedor": str(row.get("Proveedor", "")),
                "categoria": str(row.get("Categoria", "")),
                "subcategoria": str(row.get("Subcategoria", "")),
                "precio_unitario": precio,
                "unidad": "pieza",
                "color": normalizar_texto(row.get("Color", "")),
                "acabado": normalizar_texto(row.get("Acabado", "")),
                "es_promo": False
            }
            
            productos.append({
                "id": f"grif_{row.get('Codigo', '')}_{_}",
                "document": doc,
                "metadata": metadata
            })
            
        except Exception as e:
            continue
    
    if productos:
        for i in range(0, len(productos), 100):
            batch = productos[i:i+100]
            collection.add(
                ids=[p["id"] for p in batch],
                documents=[p["document"] for p in batch],
                metadatas=[p["metadata"] for p in batch]
            )
        print(f"   ✅ {len(productos)} productos en 'griferia'")
    
    return "griferia", len(productos)

# ============================================================================
# LAVABOS
# ============================================================================

def ingest_lavabos(filepath):
    print(f"\n📦 Ingestando LAVABOS: {filepath}")
    
    try:
        client.delete_collection(name="lavabos")
    except:
        pass
    
    collection = client.create_collection(
        name="lavabos",
        embedding_function=embedding_func
    )
    
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    
    productos = []
    for _, row in df.iterrows():
        try:
            precio = limpiar_precio(row.get("Precio autorizado", 0))
            if precio == 0:
                precio = limpiar_precio(row.get("Precio sistema", 0))
            
            if precio == 0:
                continue
            
            doc = f"{row.get('Descripcion', '')} {row.get('Categoria', '')}"
            
            metadata = {
                "codigo": str(row.get("Codigo", "")),
                "descripcion": str(row.get("Descripcion", "")),
                "proveedor": str(row.get("Proveedor", "")),
                "categoria": str(row.get("Categoria", "")),
                "subcategoria": str(row.get("Subcategoria", "")),
                "precio_unitario": precio,
                "unidad": "pieza",
                "color": normalizar_texto(row.get("Color", "")),
                "acabado": normalizar_texto(row.get("Acabado", "")),
                "es_promo": False
            }
            
            productos.append({
                "id": f"lav_{row.get('Codigo', '')}_{_}",
                "document": doc,
                "metadata": metadata
            })
            
        except Exception as e:
            continue
    
    if productos:
        for i in range(0, len(productos), 100):
            batch = productos[i:i+100]
            collection.add(
                ids=[p["id"] for p in batch],
                documents=[p["document"] for p in batch],
                metadatas=[p["metadata"] for p in batch]
            )
        print(f"   ✅ {len(productos)} productos en 'lavabos'")
    
    return "lavabos", len(productos)

# ============================================================================
# SANITARIOS (WC, tazas, tanques, pedestales)
# ============================================================================

def ingest_sanitarios(filepath):
    print(f"\n📦 Ingestando SANITARIOS: {filepath}")
    
    try:
        client.delete_collection(name="sanitarios")
    except:
        pass
    
    collection = client.create_collection(
        name="sanitarios",
        embedding_function=embedding_func
    )
    
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    
    productos = []
    for _, row in df.iterrows():
        try:
            precio = limpiar_precio(row.get("Precio autorizado", 0))
            if precio == 0:
                precio = limpiar_precio(row.get("Precio sistema", 0))
            
            if precio == 0:
                continue
            
            doc = f"{row.get('Descripcion', '')} {row.get('Categoria', '')}"
            
            metadata = {
                "codigo": str(row.get("Codigo", "")),
                "descripcion": str(row.get("Descripcion", "")),
                "proveedor": str(row.get("Proveedor", "")),
                "categoria": str(row.get("Categoria", "")),
                "subcategoria": str(row.get("Subcategoria", "")),
                "precio_unitario": precio,
                "unidad": "pieza",
                "color": normalizar_texto(row.get("Color", "")),
                "acabado": normalizar_texto(row.get("Acabado", "")),
                "es_promo": False
            }
            
            productos.append({
                "id": f"san_{row.get('Codigo', '')}_{_}",
                "document": doc,
                "metadata": metadata
            })
            
        except Exception as e:
            continue
    
    if productos:
        for i in range(0, len(productos), 100):
            batch = productos[i:i+100]
            collection.add(
                ids=[p["id"] for p in batch],
                documents=[p["document"] for p in batch],
                metadatas=[p["metadata"] for p in batch]
            )
        print(f"   ✅ {len(productos)} productos en 'sanitarios'")
    
    return "sanitarios", len(productos)

# ============================================================================
# MUEBLES (de baño)
# ============================================================================

def ingest_muebles(filepath):
    print(f"\n📦 Ingestando MUEBLES: {filepath}")
    
    try:
        client.delete_collection(name="muebles")
    except:
        pass
    
    collection = client.create_collection(
        name="muebles",
        embedding_function=embedding_func
    )
    
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    
    productos = []
    for _, row in df.iterrows():
        try:
            precio = limpiar_precio(row.get("Precio autorizado", 0))
            if precio == 0:
                precio = limpiar_precio(row.get("Precio sistema", 0))
            
            if precio == 0:
                continue
            
            doc = f"{row.get('Descripcion', '')} {row.get('Categoria', '')}"
            
            metadata = {
                "codigo": str(row.get("Codigo", "")),
                "descripcion": str(row.get("Descripcion", "")),
                "proveedor": str(row.get("Proveedor", "")),
                "categoria": str(row.get("Categoria", "")),
                "subcategoria": str(row.get("Subcategoria", "")),
                "precio_unitario": precio,
                "unidad": "pieza",
                "color": normalizar_texto(row.get("Color", "")),
                "acabado": normalizar_texto(row.get("Acabado", "")),
                "es_promo": False
            }
            
            productos.append({
                "id": f"mue_{row.get('Codigo', '')}_{_}",
                "document": doc,
                "metadata": metadata
            })
            
        except Exception as e:
            continue
    
    if productos:
        for i in range(0, len(productos), 100):
            batch = productos[i:i+100]
            collection.add(
                ids=[p["id"] for p in batch],
                documents=[p["document"] for p in batch],
                metadatas=[p["metadata"] for p in batch]
            )
        print(f"   ✅ {len(productos)} productos en 'muebles'")
    
    return "muebles", len(productos)

# ============================================================================
# TINACOS
# ============================================================================

def ingest_tinacos(filepath):
    print(f"\n📦 Ingestando TINACOS: {filepath}")
    
    try:
        client.delete_collection(name="tinacos")
    except:
        pass
    
    collection = client.create_collection(
        name="tinacos",
        embedding_function=embedding_func
    )
    
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    
    productos = []
    for _, row in df.iterrows():
        try:
            precio = limpiar_precio(row.get("Precio autorizado", 0))
            if precio == 0:
                continue
            
            doc = f"{row.get('Descripcion', '')} {row.get('Categoría', '')}"
            
            metadata = {
                "codigo": str(row.get("Codigo", "")),
                "descripcion": str(row.get("Descripcion", "")),
                "proveedor": str(row.get("Proveedor", "")),
                "categoria": str(row.get("Categoría", "")),
                "subcategoria": str(row.get("Subcategoria", "")),
                "precio_unitario": precio,
                "unidad": str(row.get("Udm", "pieza")),
                "medida": str(row.get("Medida", "")),
                "color": normalizar_texto(row.get("Color", "")),
                "es_promo": False
            }
            
            productos.append({
                "id": f"tin_{row.get('Codigo', '')}_{_}",
                "document": doc,
                "metadata": metadata
            })
            
        except Exception as e:
            continue
    
    if productos:
        for i in range(0, len(productos), 100):
            batch = productos[i:i+100]
            collection.add(
                ids=[p["id"] for p in batch],
                documents=[p["document"] for p in batch],
                metadatas=[p["metadata"] for p in batch]
            )
        print(f"   ✅ {len(productos)} productos en 'tinacos'")
    
    return "tinacos", len(productos)

# ============================================================================
# ESPEJOS
# ============================================================================

def ingest_espejos(filepath):
    print(f"\n📦 Ingestando ESPEJOS: {filepath}")
    
    try:
        client.delete_collection(name="espejos")
    except:
        pass
    
    collection = client.create_collection(
        name="espejos",
        embedding_function=embedding_func
    )
    
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    
    productos = []
    for _, row in df.iterrows():
        try:
            precio = limpiar_precio(row.get("Precio autorizado", 0))
            if precio == 0:
                continue
            
            doc = f"{row.get('Descripcion', '')} {row.get('Categoría', '')}"
            
            metadata = {
                "codigo": str(row.get("Codigo", "")),
                "descripcion": str(row.get("Descripcion", "")),
                "proveedor": str(row.get("Proveedor", "")),
                "categoria": str(row.get("Categoría", "")),
                "subcategoria": str(row.get("Subcategoria", "")),
                "precio_unitario": precio,
                "unidad": str(row.get("Udm", "pieza")),
                "medida": str(row.get("Medida", "")),
                "color": normalizar_texto(row.get("Color", "")),
                "es_promo": False
            }
            
            productos.append({
                "id": f"esp_{row.get('Codigo', '')}_{_}",
                "document": doc,
                "metadata": metadata
            })
            
        except Exception as e:
            continue
    
    if productos:
        for i in range(0, len(productos), 100):
            batch = productos[i:i+100]
            collection.add(
                ids=[p["id"] for p in batch],
                documents=[p["document"] for p in batch],
                metadatas=[p["metadata"] for p in batch]
            )
        print(f"   ✅ {len(productos)} productos en 'espejos'")
    
    return "espejos", len(productos)

# ============================================================================
# TARJAS (fregaderos, lavaplatos)
# ============================================================================

def ingest_tarjas(filepath):
    print(f"\n📦 Ingestando TARJAS: {filepath}")
    
    try:
        client.delete_collection(name="tarjas")
    except:
        pass
    
    collection = client.create_collection(
        name="tarjas",
        embedding_function=embedding_func
    )
    
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    
    productos = []
    for _, row in df.iterrows():
        try:
            precio = limpiar_precio(row.get("Precio autorizado", 0))
            if precio == 0:
                continue
            
            doc = f"{row.get('Descripcion', '')} {row.get('Categoría', '')}"
            
            metadata = {
                "codigo": str(row.get("Codigo", "")),
                "descripcion": str(row.get("Descripcion", "")),
                "proveedor": str(row.get("Proveedor", "")),
                "categoria": str(row.get("Categoría", "")),
                "subcategoria": str(row.get("Subcategoria", "")),
                "precio_unitario": precio,
                "unidad": str(row.get("Udm", "pieza")),
                "medida": str(row.get("Medida", "")),
                "color": normalizar_texto(row.get("Color", "")),
                "es_promo": False
            }
            
            productos.append({
                "id": f"tar_{row.get('Codigo', '')}_{_}",
                "document": doc,
                "metadata": metadata
            })
            
        except Exception as e:
            continue
    
    if productos:
        for i in range(0, len(productos), 100):
            batch = productos[i:i+100]
            collection.add(
                ids=[p["id"] for p in batch],
                documents=[p["document"] for p in batch],
                metadatas=[p["metadata"] for p in batch]
            )
        print(f"   ✅ {len(productos)} productos en 'tarjas'")
    
    return "tarjas", len(productos)

# ============================================================================
# HERRAMIENTAS
# ============================================================================

def ingest_herramientas(filepath):
    print(f"\n📦 Ingestando HERRAMIENTAS: {filepath}")
    
    try:
        client.delete_collection(name="herramientas")
    except:
        pass
    
    collection = client.create_collection(
        name="herramientas",
        embedding_function=embedding_func
    )
    
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    
    productos = []
    for _, row in df.iterrows():
        try:
            precio = limpiar_precio(row.get("Precio autorizado", 0))
            if precio == 0:
                continue
            
            doc = f"{row.get('Descripcion', '')} {row.get('Categoría', '')}"
            
            metadata = {
                "codigo": str(row.get("Codigo", "")),
                "descripcion": str(row.get("Descripcion", "")),
                "proveedor": str(row.get("Proveedor", "")),
                "categoria": str(row.get("Categoría", "")),
                "subcategoria": str(row.get("Subcategoria", "")),
                "precio_unitario": precio,
                "unidad": str(row.get("Udm", "pieza")),
                "medida": str(row.get("Medida", "")),
                "color": normalizar_texto(row.get("Color", "")),
                "es_promo": False
            }
            
            productos.append({
                "id": f"her_{row.get('Codigo', '')}_{_}",
                "document": doc,
                "metadata": metadata
            })
            
        except Exception as e:
            continue
    
    if productos:
        for i in range(0, len(productos), 100):
            batch = productos[i:i+100]
            collection.add(
                ids=[p["id"] for p in batch],
                documents=[p["document"] for p in batch],
                metadatas=[p["metadata"] for p in batch]
            )
        print(f"   ✅ {len(productos)} productos en 'herramientas'")
    
    return "herramientas", len(productos)

# ============================================================================
# POLVOS (pegamentos, boquillas, etc)
# ============================================================================

def ingest_polvos(filepath):
    print(f"\n📦 Ingestando POLVOS: {filepath}")
    
    try:
        client.delete_collection(name="polvos")
    except:
        pass
    
    collection = client.create_collection(
        name="polvos",
        embedding_function=embedding_func
    )
    
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    
    productos = []
    for _, row in df.iterrows():
        try:
            precio = limpiar_precio(row.get("Precio venta Final", 0))
            if precio == 0:
                precio = limpiar_precio(row.get("Precio sistema", 0))
            
            if precio == 0:
                continue
            
            doc = f"{row.get('Descripcion', '')} {row.get('Subcategoria', '')}"
            
            metadata = {
                "codigo": str(row.get("Codigo", "")),
                "descripcion": str(row.get("Descripcion", "")),
                "proveedor": str(row.get("Proveedor", "")),
                "categoria": str(row.get("Categoría", "")),
                "subcategoria": str(row.get("Subcategoria", "")),
                "precio_unitario": precio,
                "unidad": str(row.get("Udm", "saco")),
                "presentacion": str(row.get("Presentación", "")),
                "color": normalizar_texto(row.get("Color", "")),
                "es_promo": False
            }
            
            productos.append({
                "id": f"pol_{row.get('Codigo', '')}_{_}",
                "document": doc,
                "metadata": metadata
            })
            
        except Exception as e:
            continue
    
    if productos:
        for i in range(0, len(productos), 100):
            batch = productos[i:i+100]
            collection.add(
                ids=[p["id"] for p in batch],
                documents=[p["document"] for p in batch],
                metadatas=[p["metadata"] for p in batch]
            )
        print(f"   ✅ {len(productos)} productos en 'polvos'")
    
    return "polvos", len(productos)

# ============================================================================
# OTRAS RESTANTES (lo que no clasificó en categorías específicas)
# ============================================================================

def ingest_otras(filepath):
    print(f"\n📦 Ingestando OTRAS RESTANTES: {filepath}")
    
    try:
        client.delete_collection(name="otras")
    except:
        pass
    
    collection = client.create_collection(
        name="otras",
        embedding_function=embedding_func
    )
    
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    
    productos = []
    for _, row in df.iterrows():
        try:
            precio = limpiar_precio(row.get("Precio autorizado", 0))
            if precio == 0:
                precio = limpiar_precio(row.get("Precio sistema", 0))
            
            if precio == 0:
                continue
            
            doc = f"{row.get('Descripcion', '')} {row.get('Categoría', '')}"
            
            metadata = {
                "codigo": str(row.get("Codigo", "")),
                "descripcion": str(row.get("Descripcion", "")),
                "proveedor": str(row.get("Proveedor", "")),
                "categoria": str(row.get("Categoría", "")),
                "subcategoria": str(row.get("Subcategoria", "")),
                "precio_unitario": precio,
                "unidad": str(row.get("Udm", "pieza")),
                "medida": str(row.get("Medida", "")),
                "color": normalizar_texto(row.get("Color", "")),
                "es_promo": False
            }
            
            productos.append({
                "id": f"otr_{row.get('Codigo', '')}_{_}",
                "document": doc,
                "metadata": metadata
            })
            
        except Exception as e:
            continue
    
    if productos:
        for i in range(0, len(productos), 100):
            batch = productos[i:i+100]
            collection.add(
                ids=[p["id"] for p in batch],
                documents=[p["document"] for p in batch],
                metadatas=[p["metadata"] for p in batch]
            )
        print(f"   ✅ {len(productos)} productos en 'otras'")
    
    return "otras", len(productos)

# ============================================================================
# PROMO (marcar productos en promoción en TODAS las colecciones)
# ============================================================================

def marcar_promos(filepath):
    print(f"\n🏷️  Marcando promociones en TODAS las colecciones: {filepath}")
    
    df = pd.read_csv(filepath)
    codigos_promo = set(df["Codigo"].astype(str).tolist())
    
    # Lista de todas las colecciones donde buscar
    colecciones = ["nacionales", "importados", "griferia", "lavabos", "sanitarios", 
                   "muebles", "tinacos", "espejos", "tarjas", "herramientas", "polvos", "otras"]
    
    total_promos = 0
    for coleccion in colecciones:
        try:
            collection = client.get_collection(name=coleccion)
            result = collection.get()
            
            actualizados = 0
            for i, metadata in enumerate(result["metadatas"]):
                if metadata and metadata.get("codigo") in codigos_promo:
                    metadata["es_promo"] = True
                    collection.update(
                        ids=[result["ids"][i]],
                        metadatas=[metadata]
                    )
                    actualizados += 1
                    total_promos += 1
            
            if actualizados > 0:
                print(f"   ✅ {actualizados} promos en '{coleccion}'")
        except Exception as e:
            # La colección puede no existir aún, ignorar
            pass
    
    print(f"   🏷️  TOTAL: {total_promos} productos marcados como promoción")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ingesta VAMA - Versión con categorías separadas")
    parser.add_argument("--tipo", required=True, 
                       choices=["nacionales", "importados", "griferia", "lavabos", "sanitarios", 
                               "muebles", "tinacos", "espejos", "tarjas", "herramientas", 
                               "polvos", "otras", "promos", "todo"],
                       help="Tipo de ingesta")
    parser.add_argument("--file", help="Ruta al archivo CSV")
    
    args = parser.parse_args()
    
    data_dir = "data"
    
    if args.tipo == "todo":
        print("🚀 INGESTA COMPLETA VAMA - CATEGORÍAS SEPARADAS")
        print("=" * 60)
        
        # Nacionales e importados (pisos)
        if os.path.exists(f"{data_dir}/nacionales.csv"):
            ingest_nacionales(f"{data_dir}/nacionales.csv")
        
        if os.path.exists(f"{data_dir}/importados.csv"):
            ingest_importados(f"{data_dir}/importados.csv")
        
        # Nuevas categorías de grifería
        if os.path.exists(f"{data_dir}/griferia_pura.csv"):
            ingest_griferia(f"{data_dir}/griferia_pura.csv")
        
        if os.path.exists(f"{data_dir}/lavabos.csv"):
            ingest_lavabos(f"{data_dir}/lavabos.csv")
        
        if os.path.exists(f"{data_dir}/sanitarios.csv"):
            ingest_sanitarios(f"{data_dir}/sanitarios.csv")
        
        # Muebles (unificados)
        if os.path.exists(f"{data_dir}/muebles.csv"):
            ingest_muebles(f"{data_dir}/muebles.csv")
        
        # Nuevas categorías de otras.csv
        if os.path.exists(f"{data_dir}/tinacos.csv"):
            ingest_tinacos(f"{data_dir}/tinacos.csv")
        
        if os.path.exists(f"{data_dir}/espejos.csv"):
            ingest_espejos(f"{data_dir}/espejos.csv")
        
        if os.path.exists(f"{data_dir}/tarjas.csv"):
            ingest_tarjas(f"{data_dir}/tarjas.csv")
        
        if os.path.exists(f"{data_dir}/herramientas.csv"):
            ingest_herramientas(f"{data_dir}/herramientas.csv")
        
        # Polvos
        if os.path.exists(f"{data_dir}/polvos.csv"):
            ingest_polvos(f"{data_dir}/polvos.csv")
        
        # Otras restantes
        if os.path.exists(f"{data_dir}/otras_restantes.csv"):
            ingest_otras(f"{data_dir}/otras_restantes.csv")
        
        # Marcar promociones en todas las colecciones
        if os.path.exists(f"{data_dir}/promo.csv"):
            marcar_promos(f"{data_dir}/promo.csv")
        
        print("\n" + "=" * 60)
        print("✅ INGESTA COMPLETA CON CATEGORÍAS SEPARADAS")
        
    elif args.tipo == "nacionales":
        ingest_nacionales(args.file)
    elif args.tipo == "importados":
        ingest_importados(args.file)
    elif args.tipo == "griferia":
        ingest_griferia(args.file)
    elif args.tipo == "lavabos":
        ingest_lavabos(args.file)
    elif args.tipo == "sanitarios":
        ingest_sanitarios(args.file)
    elif args.tipo == "muebles":
        ingest_muebles(args.file)
    elif args.tipo == "tinacos":
        ingest_tinacos(args.file)
    elif args.tipo == "espejos":
        ingest_espejos(args.file)
    elif args.tipo == "tarjas":
        ingest_tarjas(args.file)
    elif args.tipo == "herramientas":
        ingest_herramientas(args.file)
    elif args.tipo == "polvos":
        ingest_polvos(args.file)
    elif args.tipo == "otras":
        ingest_otras(args.file)
    elif args.tipo == "promos":
        marcar_promos(args.file)

if __name__ == "__main__":
    main()