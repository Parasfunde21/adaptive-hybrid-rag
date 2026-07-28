from sentence_transformers import SentenceTransformer
from chunk_service import chunk_text
from pdf_service import extract_text, clean_text

# Load embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(chunks):
    embeddings = model.encode(chunks)

    return embeddings


if __name__ == "__main__":

    pdf = "../../data/raw/sample.pdf"

    text = extract_text(pdf)

    cleaned = clean_text(text)

    chunks = chunk_text(cleaned)

    embeddings = generate_embeddings(chunks)

    print(f"Total Chunks: {len(chunks)}")
    print(f"Embedding Shape: {embeddings.shape}")

    print("\nFirst Chunk:\n")
    print(chunks[0][:200])

    print("\nFirst Embedding (first 10 values):")
    print(embeddings[0][:10])