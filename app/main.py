import os, tempfile, aiofiles, shutil, uuid
from typing import List, Optional, Dict, Any
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
import ocrmypdf
from pyannote.audio import Pipeline
from openai import OpenAI
from pydantic import BaseModel
from fpdf import FPDF
from pydub import AudioSegment
from PIL import Image
import os
from docx import Document
from fastapi.responses import StreamingResponse
import io
# ---------- App & CORS ----------
app = FastAPI()

origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Config ----------
DEFAULT_TEMPLATES = {
    "Demanda": "app/templates/demanda_base.txt",
    "Poder": "app/templates/poder_base.txt",
    "Escrito": "app/templates/escrito_base.txt",
    "Testamento": "app/templates/testamento_base.txt",
    "Declaración Jurada": "app/templates/declaracion_Jurada_base.txt",
}

# 🚨 API KEY fija (tuya)
client = OpenAI(api_key=os.getenv("APIGPT"))

# HuggingFace diarización (opcional)
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=os.getenv("HF_TOKEN")
)

BASE_DIR = Path("data")
PDF_DIR = BASE_DIR / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)

EVIDENCES: Dict[str, str] = {}
chats: Dict[str, Dict[str, Any]] = {}

# =============== Utilidades ===============
def extract_text_from_pdf(path: str, max_pages: Optional[int] = None) -> str:
    reader = PdfReader(path)
    pages = range(len(reader.pages)) if max_pages is None else range(min(max_pages, len(reader.pages)))
    return "\n".join([reader.pages[i].extract_text() or "" for i in pages])

async def build_evidence_text(
    evidence_files: List[UploadFile],
    ocr: bool = True,
    max_pages_per_pdf: Optional[int] = None,
    max_chars_total: int = 120_000,
) -> str:
    if not evidence_files:
        return ""
    tmpdir = tempfile.mkdtemp(prefix="evidence_")
    parts: List[str] = []
    try:
        for idx, uf in enumerate(evidence_files, start=1):
            raw_path = os.path.join(tmpdir, f"raw_{idx}.pdf")
            async with aiofiles.open(raw_path, "wb") as f:
                data = await uf.read()
                await f.write(data)
            text_path = raw_path
            if ocr:
                ocr_path = os.path.join(tmpdir, f"ocr_{idx}.pdf")
                try:
                    ocrmypdf.ocr(raw_path, ocr_path, force_ocr=True, output_type="pdf", skip_text=False)
                    text_path = ocr_path
                except Exception:
                    text_path = raw_path
            text = extract_text_from_pdf(text_path, max_pages=max_pages_per_pdf).strip()
            header = f"=== PRUEBA {idx}: {uf.filename or 'adjunto.pdf'} ==="
            parts.append(f"{header}\n{text}\n")
        full = "\n".join(parts)
        if len(full) > max_chars_total:
            full = full[:max_chars_total] + "\n[...truncado por longitud]"
        return full
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def generar_pdf(path: str, messages: list[dict]):
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("DejaVu", "", "C:/Windows/Fonts/DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", size=12)

    last_doc = None
    for m in reversed(messages):
        if m["role"] == "assistant" and m["content"].strip():
            last_doc = m["content"].strip()
            break
    pdf.multi_cell(0, 10, last_doc or "[Documento aún sin contenido]")
    pdf.output(path)

# =============== Chat API ===============
class ChatStartRequest(BaseModel):
    documento_inicial: str

@app.post("/chatbot/start/")
async def start_chat(req: ChatStartRequest):
    chat_id = str(uuid.uuid4())
    chats[chat_id] = {
        "messages": [
            {"role": "system", "content": "Sos un asistente legal que redacta en estilo jurídico formal."},
            {"role": "assistant", "content": req.documento_inicial}
        ],
        "pdf_path": str(PDF_DIR / f"{chat_id}.pdf"),
    }
    generar_pdf(chats[chat_id]["pdf_path"], chats[chat_id]["messages"])
    return {"chat_id": chat_id, "respuesta": req.documento_inicial}

class ChatMessageRequest(BaseModel):
    chat_id: str
    mensaje_usuario: str

@app.post("/chatbot/message/")
async def send_message(req: ChatMessageRequest):
    if req.chat_id not in chats:
        raise HTTPException(404, "Chat no encontrado")

    chats[req.chat_id]["messages"].append({"role": "user", "content": req.mensaje_usuario})

    resp = client.chat.completions.create(
        model="gpt-5",   # ✅ ahora GPT-5
        messages=chats[req.chat_id]["messages"],
        max_completion_tokens=4096,
    )
    answer = resp.choices[0].message.content
    chats[req.chat_id]["messages"].append({"role": "assistant", "content": answer})
    generar_pdf(chats[req.chat_id]["pdf_path"], chats[req.chat_id]["messages"])
    return {"respuesta": answer}

@app.post("/chatbot/upload/{chat_id}")
async def upload_file(chat_id: str, evidence_files: List[UploadFile] = File(...)):
    if chat_id not in chats:
        raise HTTPException(404, "Chat no encontrado")

    text = await build_evidence_text(evidence_files)
    user_msg = f"📎 Usuario subió archivo(s):\n{text[:500]}..."
    chats[chat_id]["messages"].append({"role": "user", "content": user_msg})

    resp = client.chat.completions.create(
        model="gpt-5",   # ✅ ahora GPT-5
        messages=chats[chat_id]["messages"],
        max_completion_tokens=4096,
    )
    answer = resp.choices[0].message.content
    chats[chat_id]["messages"].append({"role": "assistant", "content": answer})
    generar_pdf(chats[chat_id]["pdf_path"], chats[chat_id]["messages"])
    return {"respuesta": answer}

@app.post("/chatbot/send/")
async def send_chat_message(
    chat_id: str = Form(...),
    mensaje_usuario: str = Form(""),
    archivo: UploadFile | None = File(None),
):
    if chat_id not in chats:
        raise HTTPException(404, "Chat no encontrado")

    user_msg = mensaje_usuario.strip()

    if archivo:
        tmp_path = f"uploads/{archivo.filename}"
        os.makedirs("uploads", exist_ok=True)
        with open(tmp_path, "wb") as f:
            f.write(await archivo.read())
        user_msg += f"\n📎 Archivo adjunto: {archivo.filename}"

    chats[chat_id]["messages"].append({"role": "user", "content": user_msg})

    resp = client.chat.completions.create(
        model="gpt-5",   # ✅ ahora GPT-5
        messages=chats[chat_id]["messages"],
        max_completion_tokens=4096,
    )
    answer = resp.choices[0].message.content
    chats[chat_id]["messages"].append({"role": "assistant", "content": answer})
    generar_pdf(chats[chat_id]["pdf_path"], chats[chat_id]["messages"])

    return {"respuesta": answer, "user_msg": user_msg}

@app.get("/chatbot/pdf/{chat_id}")
async def get_pdf(chat_id: str):
    if chat_id not in chats:
        raise HTTPException(404, "Chat no encontrado")
    return FileResponse(chats[chat_id]["pdf_path"], media_type="application/pdf")

# =============== Transcripción + Diarización ===============
@app.post("/transcribir_diarizado/")
async def transcribir_diarizado(audio: UploadFile = File(...)):
    # Guardar archivo temporal
    tmp_path = os.path.join(tempfile.gettempdir(), audio.filename)
    with open(tmp_path, "wb") as f:
        f.write(await audio.read())

    # Convertir SIEMPRE a WAV estándar
    converted_path = tmp_path + ".wav"
    try:
        sound = AudioSegment.from_file(tmp_path)  # autodetecta formato
        sound.export(converted_path, format="wav", codec="pcm_s16le")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error convirtiendo audio: {str(e)}")

    # 1. Transcribir con GPT-4o-transcribe
    with open(converted_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=f
        )

    raw_text = transcription.text.strip()

    # 2. Diarización con GPT-5
#     prompt = f"""
# Eres un asistente jurídico que organiza transcripciones de audios judiciales.
# El siguiente texto es una transcripción cruda.
# Asigna nombres/roles apropiados a cada hablante, si reconoces el nombre del participante por ejemplo : Pedro gonzalez unicamente coloca el nombre y no el rol (Juez, Fiscal, Testigo, Abogado, Acusado, etc).
# Formato esperado:

# Rol o nombre: texto
# Rol o nombre: texto
# ...

# Texto transcrito:
# {raw_text}
#     """

#     resp = client.chat.completions.create(
#         model="gpt-5",
#         messages=[{"role": "user", "content": prompt}],
#         max_completion_tokens=4096,
#     )

#     diarized_text = resp.choices[0].message.content.strip()

    return {"conversacion": raw_text}

# =============== WebSocket live chat ===============
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    history = [{"role": "system", "content": "Eres un asistente legal especializado en derecho uruguayo ."}]
    try:
        while True:
            data = await websocket.receive_text()
            history.append({"role": "user", "content": data})
            completion = client.chat.completions.create(
                model="gpt-5",   # ✅ ahora GPT-5
                messages=history,
                temperature=0.3,
            )
            reply = completion.choices[0].message.content
            history.append({"role": "assistant", "content": reply})
            await websocket.send_text(reply)
    except WebSocketDisconnect:
        print("Cliente desconectado")

@app.post("/ocr_archivo_con_texto/")
async def ocr_archivo_con_texto(archivo: UploadFile = File(...)):
    try:
        # Crear carpeta temporal
        tmpdir = tempfile.mkdtemp(prefix="ocr_")
        input_path = os.path.join(tmpdir, archivo.filename)

        # Guardar archivo subido
        with open(input_path, "wb") as f:
            f.write(await archivo.read())

        # Si es imagen -> convertir a PDF temporal
        ext = Path(archivo.filename).suffix.lower()
        if ext in [".jpg", ".jpeg", ".png"]:
            img = Image.open(input_path).convert("RGB")
            input_pdf = os.path.join(tmpdir, "input.pdf")
            img.save(input_pdf, "PDF")
            input_path = input_pdf

        # Salida final con OCR
        output_pdf = os.path.join(tmpdir, "ocr_output.pdf")

        try:
            ocrmypdf.ocr(input_path, output_pdf, force_ocr=True, output_type="pdf")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error en OCR: {str(e)}")

        # Devolver archivo procesado
        return FileResponse(
            output_pdf,
            media_type="application/pdf",
            filename="archivo_ocr.pdf"
        )

    finally:
        # ⚠️ Importante: no borramos tmpdir inmediatamente porque se necesita leer el PDF en FileResponse
        # FastAPI eliminará el archivo cuando termine la request.
        pass
@app.post("/resumir_documentos/")
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
            Resumí brevemente el siguiente documento en un párrafo claro y conciso.
            Documento: {uf.filename}

            Contenido:
            {text[:10000]}  # limitar por seguridad
            """
            resp = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=600
            )
            resumen = resp.choices[0].message.content.strip()

            informes.append({
                "nombre": uf.filename,
                "resumen": resumen
            })

        # Construir informe consolidado
        informe_final = "\n\n".join(
            [f"📄 {i['nombre']}\nResumen: {i['resumen']}" for i in informes]
        )

        return {"documentos": informes, "informe_final": informe_final}

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    
@app.post("/convertir_documento/")
async def convertir_documento(
    file: UploadFile = File(...),
    formato_salida: str = Form(..., regex="^(pdf|txt|docx)$")
):
    tmpdir = tempfile.mkdtemp(prefix="convert_")
    try:
        input_path = Path(tmpdir) / file.filename
        async with aiofiles.open(input_path, "wb") as f:
            data = await file.read()
            await f.write(data)

        output_path = Path(tmpdir) / f"convertido.{formato_salida}"

        # Provisoriamente: copiar el archivo original
        shutil.copy(str(input_path), str(output_path))

        return FileResponse(
            output_path,
            media_type="application/octet-stream",
            filename=f"convertido.{formato_salida}"
        )
    finally:
        pass