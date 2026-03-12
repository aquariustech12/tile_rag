import chromadb
from chromadb.utils import embedding_functions
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import sys

# 1. MODELO BGE-M3
print("🔮 Cargando modelo BGE-M3...")
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-m3",
    device="cuda"
)

# 2. CONEXIÓN
CHROMA_PATH = "chroma_db_v2"
COLLECTION_NAME = "tiles_catalog_v2"

try:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_func)
    print(f"✅ Conectado a {COLLECTION_NAME}")
except Exception as e:
    print(f"❌ Error de conexión: {e}")
    sys.exit()

# 3. EXTRAER DATOS (Corregido)
print("📡 Extrayendo galaxia de productos...")
data = collection.get(include=['embeddings', 'metadatas'])

# FIX: Usamos len() directamente sobre la lista de embeddings
if data['embeddings'] is None or len(data['embeddings']) == 0:
    print("❌ Error: No se encontraron embeddings en la colección.")
    sys.exit()

# 4. TSNE
print(f"🧠 Procesando {len(data['embeddings'])} productos...")
embeddings = np.array(data['embeddings'])
perp = min(30, len(embeddings) - 1)
vis_dims = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto').fit_transform(embeddings)

# 5. GRAFICAR
plt.figure(figsize=(14, 10))
categories = [m.get('categoria', 'otros') for m in data['metadatas']]
unique_cats = sorted(list(set(categories)))
colors = plt.cm.get_cmap('tab20', len(unique_cats))

for i, cat in enumerate(unique_cats):
    indices = [j for j, c in enumerate(categories) if c == cat]
    plt.scatter(
        vis_dims[indices, 0], 
        vis_dims[indices, 1], 
        color=colors(i), 
        label=f"{cat.upper()} ({len(indices)})", 
        alpha=0.7, 
        s=60,
        edgecolors='white',
        linewidth=0.5
    )

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Categorías de Inventario")
plt.title("MAPA RELACIONAL VECTORIAL - VAMA (BGE-M3 ENGINE)", fontsize=16, fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.4)
plt.tight_layout()

plt.savefig("galaxia_vama_v2.png", dpi=300)
print(f"🚀 ¡LOGRADO! Revisa 'galaxia_vama_v2.png' en tu carpeta.")