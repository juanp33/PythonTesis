from PyPDF2 import PdfReader
from fpdf import FPDF
import tempfile, os, shutil
from pathlib import Path
import aiofiles, ocrmypdf

# === Extraer texto de PDF ===
def extract_text_from_pdf(path: str, max_pages: int | None = None) -> str:
    """Extrae texto de un archivo PDF."""
    reader = PdfReader(path)
    pages = range(len(reader.pages)) if max_pages is None else range(min(max_pages, len(reader.pages)))
    return "\n".join([reader.pages[i].extract_text() or "" for i in pages])

# === Generar PDF a partir de texto ===
def generar_pdf(path: str, text: str):
    """Genera un archivo PDF con el texto recibido."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text.encode("latin-1", "replace").decode("latin-1"))
    pdf.output(str(path))

# === OCR de archivos subidos ===
async def procesar_evidencias(evidence_files: list, ocr: bool = True, max_pages: int | None = None, max_chars: int = 120_000) -> str:
    """Convierte archivos PDF o imágenes en texto concatenado para análisis."""
    if not evidence_files:
        return ""
    tmpdir = tempfile.mkdtemp(prefix="evid_")
    partes = []
    try:
        for idx, uf in enumerate(evidence_files, start=1):
            raw_path = os.path.join(tmpdir, f"raw_{idx}.pdf")
            async with aiofiles.open(raw_path, "wb") as f:
                await f.write(await uf.read())

            text_path = raw_path
            if ocr:
                ocr_path = os.path.join(tmpdir, f"ocr_{idx}.pdf")
                try:
                    ocrmypdf.ocr(raw_path, ocr_path, force_ocr=True, output_type="pdf", skip_text=False)
                    text_path = ocr_path
                except Exception:
                    text_path = raw_path

            text = extract_text_from_pdf(text_path, max_pages=max_pages).strip()
            header = f"=== PRUEBA {idx}: {uf.filename or 'adjunto.pdf'} ==="
            partes.append(f"{header}\n{text}\n")

        full = "\n".join(partes)
        if len(full) > max_chars:
            full = full[:max_chars] + "\n[...texto truncado por longitud]"
        return full
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
