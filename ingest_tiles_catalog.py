#!/usr/bin/env python3
"""
VAMA - Ingesta respetando nombres originales del catálogo
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
            
            doc = f"{row['Descripcion']} {row.get('Color', '')} {row.get('Tipologia', '')} {row.get('Acabado', '')}"
            
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
# GRIFERIA
# ============================================================================

def ingest_griferia(filepath):
    print(f"\n📦 Ingestando GRIFERIA: {filepath}")
    
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
                "acabado": normalizar_texto(row.get("Acabado", ""))
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
            # CORRECCIÓN: columna correcta es "Precio venta Final"
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
                "color": normalizar_texto(row.get("Color", ""))
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
# OTRAS (complementos: muebles, espejos, tinacos, etc)
# ============================================================================

def ingest_otras(filepath):
    print(f"\n📦 Ingestando OTRAS: {filepath}")
    
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
                "color": normalizar_texto(row.get("Color", ""))
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
# PROMO (marcar productos en promoción)
# ============================================================================

def marcar_promos(filepath):
    print(f"\n🏷️  Marcando promociones: {filepath}")
    
    df = pd.read_csv(filepath)
    codigos_promo = set(df["Codigo"].astype(str).tolist())
    
    for coleccion in ["nacionales", "importados"]:
        try:
            collection = client.get_collection(name=coleccion)
            result = collection.get()
            
            actualizados = 0
            for i, metadata in enumerate(result["metadatas"]):
                if metadata.get("codigo") in codigos_promo:
                    metadata["es_promo"] = True
                    collection.update(
                        ids=[result["ids"][i]],
                        metadatas=[metadata]
                    )
                    actualizados += 1
            
            print(f"   ✅ {actualizados} promos en '{coleccion}'")
        except Exception as e:
            print(f"   ⚠️  Error en {coleccion}: {e}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ingesta VAMA")
    parser.add_argument("--tipo", required=True, 
                       choices=["nacionales", "importados", "griferia", "polvos", "otras", "promos", "todo"],
                       help="Tipo de ingesta")
    parser.add_argument("--file", help="Ruta al archivo CSV")
    
    args = parser.parse_args()
    
    if args.tipo == "todo":
        print("🚀 INGESTA COMPLETA VAMA")
        print("=" * 50)
        
        if os.path.exists("data/nacionales.csv"):
            ingest_nacionales("data/nacionales.csv")
        
        if os.path.exists("data/importados.csv"):
            ingest_importados("data/importados.csv")
        
        if os.path.exists("data/griferia.csv"):
            ingest_griferia("data/griferia.csv")
        
        if os.path.exists("data/polvos.csv"):
            ingest_polvos("data/polvos.csv")
        
        if os.path.exists("data/otras.csv"):
            ingest_otras("data/otras.csv")
        
        if os.path.exists("data/promo.csv"):
            marcar_promos("data/promo.csv")
        
        print("\n" + "=" * 50)
        print("✅ INGESTA COMPLETA")
        
    elif args.tipo == "nacionales":
        ingest_nacionales(args.file)
    elif args.tipo == "importados":
        ingest_importados(args.file)
    elif args.tipo == "griferia":
        ingest_griferia(args.file)
    elif args.tipo == "polvos":
        ingest_polvos(args.file)
    elif args.tipo == "otras":
        ingest_otras(args.file)
    elif args.tipo == "promos":
        marcar_promos(args.file)

if __name__ == "__main__":
    main()