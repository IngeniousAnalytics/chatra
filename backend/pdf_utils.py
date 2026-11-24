import io
import re


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using pypdf if available, else regex stream fallback."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for i, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            if txt.strip():
                pages.append(f"[Page {i + 1}]\n{txt.strip()}")
        extracted = "\n\n".join(pages).strip()
        if extracted:
            return extracted
    except Exception:
        pass

    text_matches = re.findall(rb"\(([^\(\)]+)\)\s*T[jJ]", pdf_bytes)
    if text_matches:
        return " ".join(m.decode("latin1", errors="ignore") for m in text_matches[:300]).strip()
    return "(Unable to parse PDF text. Ensure pypdf is installed for full PDF support.)"


def extract_uploaded_text(filename: str, file_bytes: bytes, max_chars: int = 30000) -> dict:
    """Extract readable text from an uploaded PDF or plain-text file."""
    if filename.lower().endswith(".pdf"):
        extracted_text = extract_text_from_pdf_bytes(file_bytes)
    else:
        extracted_text = file_bytes.decode("utf-8", errors="ignore")

    total_chars = len(extracted_text)
    truncated = total_chars > max_chars
    return {
        "ok": True,
        "filename": filename,
        "chars": total_chars,
        "truncated": truncated,
        "text": extracted_text[:max_chars],
    }
