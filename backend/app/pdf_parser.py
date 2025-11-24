from io import BytesIO
from pdfminer.high_level import extract_text

def pdf_bytes_to_text(data: bytes) -> str:
    # Simple, reliable extraction
    buf = BytesIO(data)
    text = extract_text(buf)
    return text.strip()
