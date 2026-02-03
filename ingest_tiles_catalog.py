import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os

# Configuración de rutas
CHROMA_PATH = "chroma_db"
DATA_PATH = "data"
COLLECTION_NAME = "tiles_inventory"

# Modelo de embedding multilingüe (ideal para español técnico)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

def clean_float(value):
    """Limpia valores numéricos de los CSVs"""
    try:
        return float(str(value).replace('$', '').replace(',', '').strip())
    except:
        return 0.0

def ingest_data():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, 
        embedding_function=embedding_func
    )

    # Configuración de los archivos específicos
    files_to_process = [
        {"name": "Copia de ESCALA DE PRECIOS DICIEMBRE.xlsx - Pisos Nacionales.csv", "type": "piso"},
        {"name": "Copia de ESCALA DE PRECIOS DICIEMBRE.xlsx - Pisos Importados.csv", "type": "piso"},
        {"name": "Copia de ESCALA DE PRECIOS DICIEMBRE.xlsx - Polvos.csv", "type": "polvo"}
    ]

    for file_info in files_to_process:
        path = os.path.join(DATA_PATH, file_info["name"])
        if not os.path.exists(path):
            print(f"⚠️ No se encontró: {file_info['name']}")
            continue

        print(f"Procesando {file_info['name']}...")
        df = pd.read_csv(path)

        for i, row in df.iterrows():
            # Extraer campos con manejo de errores por nombres de columnas con espacios
            desc = str(row.get('Descripcion ', row.get('Descripcion', 'N/A')))
            codigo = str(row.get('Codigo', 'S/C'))
            
            if file_info["type"] == "piso":
                precio = clean_float(row.get('Oferta Final M2 ', 0))
                metraje = clean_float(row.get('Metraje', 1.0))
                formato = str(row.get('Formato', 'N/A'))
                color = str(row.get('Color', 'N/A'))
                
                content = f"Piso: {desc}. Formato: {formato}. Color: {color}."
                metadata = {
                    "codigo": codigo,
                    "precio": precio,
                    "metraje_caja": metraje,
                    "categoria": "piso"
                }
            else:
                # Lógica para Polvos (Adhesivos)
                precio = clean_float(row.get('Precio venta Final', 0))
                presentacion = str(row.get('Presentación', 'N/A'))
                content = f"Adhesivo/Polvo: {desc}. Presentación: {presentacion}."
                metadata = {
                    "codigo": codigo,
                    "precio": precio,
                    "metraje_caja": 1.0,
                    "categoria": "polvo"
                }

            collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[f"{file_info['type']}_{codigo}_{i}"]
            )

    print(f"✅ Ingesta finalizada. {collection.count()} productos en la base de datos.")

if __name__ == "__main__":
    ingest_data()