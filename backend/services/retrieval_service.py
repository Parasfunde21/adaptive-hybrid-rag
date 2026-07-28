import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="../../chroma_db")
collection = client.get_collection("documents")

query = "What is Agile Scrum?"

query_embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print("\nTop Results:\n")

for i, doc in enumerate(results["documents"][0], 1):
    print(f"Result {i}")
    print(doc[:300])
    print("-" * 60)