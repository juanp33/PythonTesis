from fastapi import APIRouter, UploadFile, File, Form
from app.core.config import client
from pathlib import Path
import tempfile, aiofiles, shutil
import os
from typing import List
from app.core.utils import extract_text_from_pdf, generar_pdf
router = APIRouter(prefix="/resumen", tags=["Resumen"])

@router.post("/")

async def resumir_documentos(
    files: List[UploadFile] = File(..., description="Lista de documentos a resumir"),
    ocr: bool = Form(True)
):
    informes = []

    tmpdir = tempfile.mkdtemp(prefix="resumenes_")
    try:
        for idx, uf in enumerate(files, start=1):
            # Guardar archivo temporal
            raw_path = os.path.join(tmpdir, uf.filename)
            async with aiofiles.open(raw_path, "wb") as f:
                data = await uf.read()
                await f.write(data)

            # Extraer texto (PDF o texto plano)
            if uf.filename.lower().endswith(".pdf"):
                text = extract_text_from_pdf(raw_path)
            else:
                try:
                    text = Path(raw_path).read_text(encoding="utf-8", errors="ignore")
                except:
                    text = ""

            # Resumir con GPT
            prompt = f"""
            Leé atentamente el siguiente documento y elaborá un resumen **detallado pero claro y conciso** en un solo párrafo.
No repitas frases textuales, sino que sintetizá el contenido con lenguaje natural y fluido, destacando los puntos más relevantes.

Tu resumen debe:
- Incluir los hechos, argumentos o hallazgos principales en orden lógico.
- Mencionar actores, fechas y acciones concretas si aparecen.
- Explicar brevemente el propósito o contexto del documento.
- Omitir citas literales, repeticiones o fórmulas legales innecesarias.
- Mantener un tono formal, objetivo y neutro.
            Documento: {uf.filename}

            Contenido:
            {text[:100000]}  # limitar por seguridad
            """
            resp = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=16000
            )
            resumen = resp.choices[0].message.content.strip()

            informes.append({
                "nombre": uf.filename,
                "resumen": resumen
            })

        informe_final = "\n\n".join([f"Documento: {inf['nombre']}\nResumen: {inf['resumen']}" for inf in informes])
        return {"documentos": informes, "informe_final": informe_final}

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    