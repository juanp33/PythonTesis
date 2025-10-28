from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import os
import uuid
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List
import fitz  # PyMuPDF
import aiofiles
from openai import OpenAI
from fpdf import FPDF
from fastapi.responses import FileResponse

router = APIRouter(prefix="/redaccion_documento", tags=["Redacción"])
EVIDENCES = {}
REDACCION_CHATS = {}
client = OpenAI(api_key=os.getenv("APIGPT"))
PDF_DIR = Path("pdfs")
PDF_DIR.mkdir(exist_ok=True)


@router.post("/")
async def redaccion_documento(
    document_type: str = Form(...),
    instructions: str = Form(""),
    template_file: Optional[UploadFile] = File(None),
    evidence_files: Optional[List[UploadFile]] = File(None),
):
    tmpdir = tempfile.mkdtemp(prefix="redaccion_")
    try:
        base_text = ""
        evidence_text = ""

        # ⚙️ Plantilla personalizada o base
        if template_file:
            path = os.path.join(tmpdir, template_file.filename)
            async with aiofiles.open(path, "wb") as f:
                await f.write(await template_file.read())
            base_text = Path(path).read_text(encoding="utf-8", errors="ignore").strip()
        else:
            templates_dir = Path(__file__).parent / "templates"
            template_map = {
                "Demanda": "demanda_base.txt",
                "Poder": "poder_base.txt",
                "Escrito": "escrito_proceso_iniciado.txt",
                "Declaración Jurada": "declaracion_jurada_base.txt",
                "Testamento": "testamento_base.txt",
                "Denuncia": "denuncia_base.txt",
            }
            template_name = template_map.get(document_type)
            if template_name:
                template_path = templates_dir / template_name
                base_text = (
                    template_path.read_text(encoding="utf-8", errors="ignore").strip()
                    if template_path.exists()
                    else "[sin plantilla predeterminada]"
                )
            else:
                base_text = "[sin plantilla]"

        # 📎 Procesar evidencias (PDF o TXT)
        if evidence_files:
            for ev in evidence_files:
                try:
                    ev_path = os.path.join(tmpdir, ev.filename)
                    async with aiofiles.open(ev_path, "wb") as f:
                        await f.write(await ev.read())

                    if ev.filename.lower().endswith(".pdf"):
                        pdf_text = ""
                        with fitz.open(ev_path) as doc:
                            for page in doc:
                                pdf_text += page.get_text("text") + "\n"

                        evidence_text += f"\n---\n📄 {ev.filename}:\n{pdf_text.strip()}\n"

                    elif ev.filename.lower().endswith(".txt"):
                        txt_content = Path(ev_path).read_text(encoding="utf-8", errors="ignore")
                        evidence_text += f"\n---\n📄 {ev.filename}:\n{txt_content.strip()}\n"

                    else:
                        evidence_text += f"\n- {ev.filename} (tipo no soportado)\n"

                except Exception as e:
                    evidence_text += f"\n[⚠️ No se pudo leer {ev.filename}: {e}]\n"
        else:
            evidence_text = "[sin evidencias]"

        # 🧠 Prompt principal (ajustado)
        prompt = f"""
Redactá un documento jurídico completo de tipo **{document_type}**, siguiendo lenguaje jurídico uruguayo formal.
Debe incluir literalmente el texto completo de cada evidencia, sin resumirlo, modificarlo ni parafrasearlo.
Copiá cada evidencia entre líneas de separación, como si formara parte integral del documento.
No inventes hechos ni datos no provistos.

• Instrucciones: {instructions or '[sin instrucciones]'}
• Evidencias (texto completo a incorporar literalmente):
{evidence_text or '[sin evidencias]'}
• Plantilla base:
{base_text or '[sin plantilla]'}
"""

        completion = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "Sos un abogado uruguayo experto en derecho civil y procesal."},
                {"role": "user", "content": prompt},
            ],
            
        )

        generated_text = completion.choices[0].message.content.strip()
        evidence_id = str(uuid.uuid4())
        pdf_path = PDF_DIR / f"redaccion_{evidence_id}.pdf"

        # 📝 Crear el PDF (Arial estándar)
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, generated_text.encode("latin-1", "replace").decode("latin-1"))
        pdf.output(str(pdf_path))

        EVIDENCES[evidence_id] = evidence_text
        REDACCION_CHATS[evidence_id] = {
            "texto_actual": generated_text,
            "pdf_path": str(pdf_path),
            "evidencia_text": evidence_text,
        }

        return {
            "generated_text": generated_text,
            "evidence_id": evidence_id,
            "pdf_url": f"http://127.0.0.1:8000/redaccion_documento/redaccion_chat/pdf/{evidence_id}",
        }

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@router.post("/redaccion_chat/start/")
async def redaccion_chat_start(req: dict):
    chat_id = str(uuid.uuid4())
    evidence_id = req.get("evidence_id", chat_id)

    REDACCION_CHATS[chat_id] = {
        "texto_actual": req["documento_inicial"],
        "pdf_path": str(PDF_DIR / f"redaccion_{evidence_id}.pdf"),
        "evidence_id": evidence_id,
        "evidencia_text": EVIDENCES.get(evidence_id, ""),
    }

    return {
        "chat_id": chat_id,
        "respuesta": "✅ Documento generado correctamente. Ya podés empezar a editarlo.",
        "pdf_url": f"http://127.0.0.1:8000/redaccion_documento/redaccion_chat/pdf/{evidence_id}",
    }


@router.post("/redaccion_chat/message/")
async def redaccion_chat_message(req: dict):
    chat_id = req["chat_id"]
    mensaje = req["mensaje_usuario"]

    if chat_id not in REDACCION_CHATS:
        raise HTTPException(404, "Chat no encontrado")

    chat = REDACCION_CHATS[chat_id]
    texto_actual = chat["texto_actual"]
    evidence_id = chat.get("evidence_id", chat_id)

    evidence_section = ""
    if chat.get("evidencia_text") and chat["evidencia_text"].strip():
        evidence_section = f"""
Evidencias relevantes:
---
{chat['evidencia_text']}
---
"""

    prompt = f"""
El siguiente texto es un documento jurídico uruguayo en proceso de redacción.
Aplicá las instrucciones del usuario manteniendo tono formal y estructura.
Devolvé primero el documento actualizado bajo [DOCUMENTO], luego un feedback de los cambios realizados [CHATBOT].

Documento actual:
---
{texto_actual}
---
Instrucción del usuario:
---
{mensaje}
---
Evidencias del caso a usar si son necesarias:
{evidence_section}
"""

    completion = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": "Sos un abogado uruguayo experto en redacción jurídica."},
            {"role": "user", "content": prompt},
        ]
        
    )

    full_response = completion.choices[0].message.content.strip()
    new_text, feedback = "", "✅ Documento actualizado."

    if "[DOCUMENTO]" in full_response:
        parts = full_response.split("[CHATBOT]")
        new_text = parts[0].replace("[DOCUMENTO]", "").strip()
        if len(parts) > 1:
            feedback = parts[1].strip()

        chat["texto_actual"] = new_text

        pdf_path = Path(chat["pdf_path"])
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, new_text.encode("latin-1", "replace").decode("latin-1"))
        pdf.output(str(pdf_path))
    else:
        new_text = full_response
        feedback = "⚠️ El modelo no siguió el formato esperado."

    return {
        "status": "✅ OK",
        "feedback": feedback,
        "respuesta": feedback,
        "pdf_url": f"http://127.0.0.1:8000/redaccion_documento/redaccion_chat/pdf/{evidence_id}",
    }


@router.get("/redaccion_chat/pdf/{evidence_id}")
async def get_pdf(evidence_id: str):
    pdf_path = PDF_DIR / f"redaccion_{evidence_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF no encontrado")

    headers = {"Content-Disposition": f'inline; filename="redaccion_{evidence_id}.pdf"'}
    return FileResponse(pdf_path, media_type="application/pdf", headers=headers)
