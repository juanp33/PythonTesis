from fastapi import APIRouter, UploadFile, File, Form
from app.core.config import client
from pathlib import Path
import tempfile, aiofiles, shutil

router = APIRouter(prefix="/resumen", tags=["Resumen"])

@router.post("/")
async def resumir_documentos(files: list[UploadFile] = File(...), ocr: bool = Form(True)):
    informes = []
    tmpdir = tempfile.mkdtemp(prefix="resumen_")
    try:
        for uf in files:
            raw_path = Path(tmpdir) / uf.filename
            async with aiofiles.open(raw_path, "wb") as f:
                await f.write(await uf.read())

            if uf.filename.lower().endswith(".pdf"):
                from app.core.utils import extract_text_from_pdf
                text = extract_text_from_pdf(str(raw_path))
            else:
                text = raw_path.read_text(encoding="utf-8", errors="ignore")

            prompt = f"Resumí de forma clara y completa el documento {uf.filename}:\n{text[:80000]}"
            resp = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=6000
            )
            resumen = resp.choices[0].message.content.strip()
            informes.append({"nombre": uf.filename, "resumen": resumen})

        informe_final = "\n\n".join([f"{i['nombre']}:\n{i['resumen']}" for i in informes])
        return {"documentos": informes, "informe_final": informe_final}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
