from pdf_service import extract_text, clean_text


def chunk_text(text, chunk_size=500):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    return chunks


if __name__ == "__main__":
    pdf = "../../data/raw/sample.pdf"

    text = extract_text(pdf)

    cleaned = clean_text(text)

    chunks = chunk_text(cleaned)

    print(f"Total Chunks: {len(chunks)}")

    print("\nFirst Chunk:\n")

    print(chunks[0])