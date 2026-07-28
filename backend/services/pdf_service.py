import fitz
import re


def extract_text(pdf_path):
    document = fitz.open(pdf_path)

    text = ""

    for page in document:
        text += page.get_text()

    document.close()

    return text


def clean_text(text):
    # Replace multiple spaces with one
    text = re.sub(r"\s+", " ", text)

    # Remove unnecessary spaces
    text = text.strip()

    return text


if __name__ == "__main__":
    pdf = "../../data/raw/sample.pdf"

    extracted_text = extract_text(pdf)

    cleaned_text = clean_text(extracted_text)

    print(cleaned_text[:1000])