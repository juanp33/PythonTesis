from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
import tempfile, os, ocrmypdf
from pathlib import Path

router = APIRouter(prefix="/ocr", tags=["OCR"])

@router.post("/")
async def ocr_archivo_con_texto(archivo: UploadFile = File(...)):
    try:
        tmpdir = tempfile.mkdtemp(prefix="ocr_")
        input_path = os.path.join(tmpdir, archivo.filename)
        with open(input_path, "wb") as f:
            f.write(await archivo.read())

        ext = Path(archivo.filename).suffix.lower()
        if ext in [".jpg", ".jpeg", ".png"]:
            img = Image.open(input_path).convert("RGB")
            input_pdf = os.path.join(tmpdir, "input.pdf")
            img.save(input_pdf, "PDF")
            input_path = input_pdf

        output_pdf = os.path.join(tmpdir, "ocr_output.pdf")
        ocrmypdf.ocr(input_path, output_pdf, force_ocr=True, output_type="pdf")

        return FileResponse(output_pdf, media_type="application/pdf", filename="archivo_ocr.pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en OCR: {str(e)}")
