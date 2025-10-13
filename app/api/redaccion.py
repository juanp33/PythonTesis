from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from app.core.config import client, PDF_DIR
from fpdf import FPDF
from pathlib import Path
import tempfile, aiofiles, shutil, uuid

router = APIRouter(prefix="/redaccion", tags=["Redacción"])
EVIDENCES = {}
REDACCION_CHATS = {}

@router.post("/")
async def redaccion_documento(
    document_type: str = Form(...),
    instructions: str = Form(""),
    template_file: UploadFile | None = File(None),
    evidence_files: list[UploadFile] | None = File(None)
):
    tmpdir = tempfile.mkdtemp(prefix="red_")
    try:
        base_text = ""
        if template_file:
            path = Path(tmpdir) / template_file.filename
            async with aiofiles.open(path, "wb") as f:
                await f.write(await template_file.read())
            base_text = path.read_text(encoding="utf-8", errors="ignore")

        evidence_text = ""
        if evidence_files:
            for ev in evidence_files:
                evidence_text += f"\n- {ev.filename}"

        prompt = f"""
Redactá un documento jurídico uruguayo tipo **{document_type}**.
• Instrucciones: {instructions}
• Evidencias: {evidence_text or '[sin evidencias]'}
• Plantilla: {base_text or '[sin plantilla]'}
"""

        completion = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "Sos un abogado uruguayo experto en derecho civil y procesal."},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=8000,
        )
        text = completion.choices[0].message.content.strip()
        evidence_id = str(uuid.uuid4())
        pdf_path = PDF_DIR / f"redaccion_{evidence_id}.pdf"

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, text.encode("latin-1", "replace").decode("latin-1"))
        pdf.output(str(pdf_path))

        EVIDENCES[evidence_id] = evidence_text
        return {"texto": text, "pdf_url": f"http://127.0.0.1:8000/redaccion/pdf/{evidence_id}"}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
