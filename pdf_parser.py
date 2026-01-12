import pdfplumber
import os
from configuration import config

class PDFParser:
    def __init__(self):
        self.chunk_size = config.CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP

    def _split_text(self, text):
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += (self.chunk_size - self.chunk_overlap)
        return chunks

    def parse_single(self, file_path: str):
        chunks = []
        try:
            with pdfplumber.open(file_path) as pdf:
                max_pages = getattr(config, 'MAX_PAGES', 1000)
                pages = pdf.pages[:max_pages]
                
                for page_idx, page in enumerate(pages):
                    text = page.extract_text()
                    if not text:
                        continue
                    
                    page_chunks = self._split_text(text)
                    
                    filename = os.path.basename(file_path).replace(".pdf", "")
                    
                    for chunk in page_chunks:
                        chunks.append({
                            "text": chunk,
                            "pdf_sha1": filename,
                            "page_index": page_idx
                        })
        except Exception as e:
            print(f"  Ошибка при обработке {os.path.basename(file_path)}: {e}")
        return chunks