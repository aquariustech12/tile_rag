#!/usr/bin/env python3
"""
Script para agregar sucursales a ChromaDB
Ejecutar: python3 poblar_sucursales.py
"""

import chromadb
from chromadb.utils import embedding_functions
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-m3", device="cpu"
)

# Conectar a ChromaDB
chroma = chromadb.PersistentClient(path="./chroma_db_v3")

# Crear o obtener colección de sucursales
try:
    coleccion = chroma.get_collection("sucursales", embedding_function=embed_fn)
    print("📦 Colección 'sucursales' ya existe, eliminando documentos...")
    coleccion.delete(where={})  # Limpiar para recargar
except:
    coleccion = chroma.create_collection("sucursales", embedding_function=embed_fn)
    print("✅ Colección 'sucursales' creada")

# Datos de sucursales
sucursales = [
    {
        "id": "culiacan_tresrios",
        "nombre": "CULIACÁN - TRES RIOS",
        "direccion": "Blvd. Enrique Sánchez Alonso #1515, Desarrollo Urbano Tres Ríos",
        "telefono": "(667) 752 20 78",
        "horario": "Lunes a Viernes 8am-8pm, Sábados 9am-6pm",
        "ciudad": "Culiacán",
        "zona": "Tres Ríos"
    },
    {
        "id": "culiacan_ramirez",
        "nombre": "CULIACÁN - IGNACIO RAMÍREZ",
        "direccion": "Ignacio Ramírez #981 pte, Jorge Almada",
        "telefono": "(667) 752 02 61",
        "horario": "Lunes a Viernes 8am-8pm, Sábados 9am-6pm",
        "ciudad": "Culiacán",
        "zona": "Ignacio Ramírez"
    },
    {
        "id": "mochis_bienestar",
        "nombre": "LOS MOCHIS - BIENESTAR",
        "direccion": "Bienestar #633, Col. Bienestar",
        "telefono": "(668) 812 58 13",
        "horario": "Lunes a Viernes 8am-8pm, Sábados 9am-6pm",
        "ciudad": "Los Mochis",
        "zona": "Bienestar"
    },
    {
        "id": "mochis_independencia",
        "nombre": "LOS MOCHIS - INDEPENDENCIA",
        "direccion": "Av. Independencia #2049 pte, Jardines del Country",
        "telefono": "(668) 176 96 13",
        "horario": "Lunes a Viernes 8am-8pm, Sábados 9am-6pm",
        "ciudad": "Los Mochis",
        "zona": "Independencia"
    }
]

# Agregar a ChromaDB
for suc in sucursales:
    texto_busqueda = f"{suc['nombre']} {suc['direccion']} {suc['ciudad']} {suc['zona']} teléfono {suc['telefono']}"
    
    metadatos = {
        "tipo": "sucursal",
        "nombre": suc['nombre'],
        "direccion": suc['direccion'],
        "telefono": suc['telefono'],
        "horario": suc['horario'],
        "ciudad": suc['ciudad'],
        "zona": suc['zona']
    }
    
    coleccion.upsert(
        ids=[suc['id']],
        documents=[texto_busqueda],
        metadatas=[metadatos]
    )
    print(f"   ✅ {suc['nombre']}")

print(f"\n🎉 {len(sucursales)} sucursales agregadas a ChromaDB")