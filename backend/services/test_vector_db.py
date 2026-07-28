import chromadb

client = chromadb.PersistentClient(path="../../chroma_db")

collection = client.get_collection("documents")

print("Total Chunks:", collection.count())