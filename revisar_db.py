import chromadb
from chromadb.utils import embedding_functions

# Configuración idéntica a tu código
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-m3", device="cpu"
)
chroma = chromadb.PersistentClient(path="chroma_db_v3")

# Vamos a ver qué hay en 'nacionales'
try:
    col = chroma.get_collection("nacionales", embedding_function=embed_fn)
    sample = col.peek(limit=3)
    
    print("\n--- REVISIÓN DE DATOS EN CHROMA ---")
    for i in range(len(sample['ids'])):
        print(f"\nID: {sample['ids'][i]}")
        print(f"METADATOS: {sample['metadatas'][i]}")
        print(f"TEXTO QUE LEE EL LLM (Document): {sample['documents'][i]}")
    print("\n----------------------------------")
except Exception as e:
    print(f"Error al leer la colección: {e}")