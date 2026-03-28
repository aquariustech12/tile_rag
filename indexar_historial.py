import json
from chromadb.utils import embedding_functions
import chromadb

CHROMA_PATH = "./chroma_db_v3"

embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-m3", device="cpu"
)
client = chromadb.PersistentClient(path=CHROMA_PATH)

# Obtener o crear la colección
col = client.get_or_create_collection("conversaciones", embedding_function=embed_fn)

with open("conversaciones_completas.jsonl", "r") as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        user_msg = data.get("mensaje_usuario", "")
        bot_resp = data.get("respuesta_bot", "")
        if not user_msg or not bot_resp:
            continue
        doc_id = f"conv_{data.get('timestamp', i)}"
        texto = f"{user_msg}\n{bot_resp}"
        col.upsert(
            documents=[texto],
            metadatas=[{
                "user_msg": user_msg,
                "bot_resp": bot_resp,
                "timestamp": data.get("timestamp", ""),
                "telefono": data.get("telefono", "")
            }],
            ids=[doc_id]
        )
        if i % 100 == 0:
            print(f"Indexados {i} registros...")
print("Indexación completada.")