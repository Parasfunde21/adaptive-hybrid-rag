import chromadb
from embedding_service import generate_embeddings
from chunk_service import chunk_text
from pdf_service import extract_text, clean_text

# Create persistent database
client = chromadb.PersistentClient(path="../../chroma_db")

collection = client.get_or_create_collection(
    name="documents"
)

pdf = "../../data/raw/sample.pdf"

text = extract_text(pdf)
cleaned = clean_text(text)
chunks = chunk_text(cleaned)

embeddings = generate_embeddings(chunks)

# Clear old data (useful while developing)
try:
    client.delete_collection("documents")
except:
    pass

collection = client.get_or_create_collection("documents")

for i, chunk in enumerate(chunks):
    collection.add(
        ids=[str(i)],
        documents=[chunk],
        embeddings=[embeddings[i].tolist()],
        metadatas=[{"source": "sample.pdf"}]
    )

print(f"Stored {collection.count()} chunks successfully!")