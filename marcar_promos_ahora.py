#!/usr/bin/env python3
"""
marcar_promos_ahora.py - Marca productos en promoción en TODAS las colecciones
Ejecutar: python3 marcar_promos_ahora.py
"""
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import os

CHROMA_PATH = "chroma_db_v3"
EMBEDDING_MODEL = "BAAI/bge-m3"

print("🔌 Conectando a ChromaDB...")
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL, device="cpu"
)
client = chromadb.PersistentClient(path=CHROMA_PATH)

# Todas las colecciones
colecciones = [
    "nacionales", "importados", "griferia", "lavabos", "sanitarios",
    "muebles", "tinacos", "espejos", "tarjas", "herramientas", "polvos", "otras"
]

# Leer promos
df_promo = pd.read_csv("data/promo.csv")
codigos_promo = set(df_promo["Codigo"].astype(str).tolist())
print(f"📦 {len(codigos_promo)} códigos en promoción")

total = 0
for coleccion in colecciones:
    try:
        coll = client.get_collection(coleccion, embedding_function=embedding_func)
        results = coll.get()
        
        actualizados = 0
        for i, meta in enumerate(results["metadatas"]):
            if meta and meta.get("codigo") in codigos_promo:
                meta["es_promo"] = True
                coll.update(
                    ids=[results["ids"][i]],
                    metadatas=[meta]
                )
                actualizados += 1
                total += 1
        
        if actualizados > 0:
            print(f"✅ {actualizados} promos en {coleccion}")
    except:
        continue

print(f"\n🎉 TOTAL: {total} productos marcados como promoción")