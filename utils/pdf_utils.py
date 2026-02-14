"""PDF에서 텍스트 추출 (n8n extractFromFile)."""
from pathlib import Path

from pypdf import PdfReader


def extract_text_from_pdf(file_path: str | Path) -> str:
    """PDF 파일 경로에서 전체 텍스트 추출."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n\n".join(parts)
