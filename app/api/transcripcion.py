from fastapi import APIRouter, UploadFile, File, HTTPException
from openai import OpenAI
from pydub import AudioSegment
import os, tempfile, uuid, asyncio, mimetypes, httpx

router = APIRouter(prefix="/transcribir_diarizado", tags=["Transcripción"])

OAI_KEY = os.getenv("APIGPT") or os.getenv("OPENAI_API_KEY")
OAI_TIMEOUT = float(os.getenv("OAI_TIMEOUT", "120"))
OAI_RETRIES = int(os.getenv("OAI_RETRIES", "3"))
OAI_CONCURRENCY = int(os.getenv("OAI_CONCURRENCY", "3"))

MODEL_MAX_SECONDS = int(os.getenv("MODEL_MAX_SECONDS", "1400"))
SAFE_CHUNK_SECONDS = min(int(os.getenv("MAX_CHUNK_SECONDS", "1200")), MODEL_MAX_SECONDS)
MP3_BITRATE = os.getenv("MP3_BITRATE", "32k")
MAX_OAI_FILE_MB = 25

TRANSCRIBE_PRIMARY = os.getenv("TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
TRANSCRIBE_LANG = os.getenv("TRANSCRIBE_LANG", "es")
OAI_URL_TRANSCRIBE = "https://api.openai.com/v1/audio/transcriptions"

DIARIZE_MODEL = os.getenv("DIARIZE_MODEL", "gpt-5-mini")

oai_client = OpenAI(api_key=OAI_KEY) if OAI_KEY else None


async def to_thread(fn, *args, **kwargs):
    return await asyncio.to_thread(fn, *args, **kwargs)


def _export_mp3(seg: AudioSegment, out_path: str) -> None:
    seg.set_frame_rate(16000).set_channels(1).export(out_path, format="mp3", bitrate=MP3_BITRATE)


def split_mp3_by_duration(src_path: str) -> list[str]:
    audio = AudioSegment.from_file(src_path)
    total_ms = len(audio)
    chunk_ms = SAFE_CHUNK_SECONDS * 1000
    base, _ = os.path.splitext(src_path)
    out = []
    start, idx = 0, 0
    while start < total_ms:
        end = min(start + chunk_ms, total_ms)
        seg = audio[start:end]
        out_path = f"{base}_p{idx:02d}.mp3"
        _export_mp3(seg, out_path)
        out.append(out_path)
        start, idx = end, idx + 1
    return out


def transcribe_multipart_httpx(path: str, model: str, language: str) -> str:
    if not OAI_KEY:
        raise HTTPException(500, "Falta API key")

    mime = mimetypes.guess_type(path)[0] or "audio/mpeg"
    with open(path, "rb") as f, httpx.Client(timeout=OAI_TIMEOUT) as cli:
        resp = cli.post(
            OAI_URL_TRANSCRIBE,
            headers={"Authorization": f"Bearer {OAI_KEY}"},
            data={"model": model, "language": language, "response_format": "json"},
            files={"file": (os.path.basename(path), f, mime)},
        )
        if resp.status_code >= 400:
            resp.raise_for_status()
        return (resp.json().get("text") or "").strip()


async def transcribe_file(path: str) -> str:
    for _ in range(OAI_RETRIES):
        try:
            return await to_thread(transcribe_multipart_httpx, path, TRANSCRIBE_PRIMARY, TRANSCRIBE_LANG)
        except httpx.HTTPStatusError as e:
            if e.response is not None and e.response.status_code == 400:
                raise HTTPException(400, e.response.text)
        await asyncio.sleep(1)
    raise HTTPException(502, "Falla en transcripción.")


def build_prompt(raw_text: str) -> str:
    return f"""Eres un asistente jurídico especializado en organizar transcripciones de audios judiciales.
Objetivo: etiquetar cada intervención con el NOMBRE si aparece explícito; si no, asignar un ROL (Juez, Fiscal, Defensor, Abogado, Testigo, Imputado, Acusado, Secretario, Perito). Si hay varias personas con el mismo rol, numerarlas (Abogado 1, Abogado 2).
No modifiques el texto ni agregues comentarios ni marcas de tiempo.
Separador entre intervenciones: \n
Formato: Rol o nombre: texto

Texto transcrito:
{raw_text}"""


def _chat_diarize_sync(model: str, content: str, timeout: float) -> str:
    resp = oai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_completion_tokens=6000,
        
        timeout=timeout,
    )
    return (resp.choices[0].message.content or "").strip()


async def diarize_window(text: str) -> str:
    for _ in range(OAI_RETRIES):
        try:
            return await to_thread(_chat_diarize_sync, DIARIZE_MODEL, build_prompt(text), OAI_TIMEOUT)
        except Exception as e:
            print(f"Error en diarización: {e}")
            await asyncio.sleep(1)
    raise HTTPException(502, "Falla en diarización.")


async def diarize_text_safely(raw_text: str) -> str:
    # Divide el texto en fragmentos más pequeños (chunks) si es necesario
    chunk_size = 4000  # Puedes ajustar el tamaño si es necesario
    chunks = [raw_text[i:i + chunk_size] for i in range(0, len(raw_text), chunk_size)]
    
    # Realiza la diarización por partes
    diarized_chunks = []
    for chunk in chunks:
        diarized_text = await diarize_window(chunk)
        diarized_chunks.append(diarized_text)
    
    return "\n".join(diarized_chunks)


async def _transcribe_with_sem(parts: list[str], sem: asyncio.Semaphore) -> list[str]:
    async def _one(p: str):
        async with sem:
            return await transcribe_file(p)
    return await asyncio.gather(*[_one(p) for p in parts])


@router.post("/")
async def transcribir_diarizado(audio: UploadFile = File(...)):
    if not OAI_KEY:
        raise HTTPException(500, "Falta API key")

    tmp_dir = os.path.join(tempfile.gettempdir(), f"trx_{uuid.uuid4().hex}")
    os.makedirs(tmp_dir, exist_ok=True)
    raw_path = os.path.join(tmp_dir, audio.filename or "audio.mp3")

    try:
        with open(raw_path, "wb") as f:
            f.write(await audio.read())

        parts = await to_thread(split_mp3_by_duration, raw_path)
        texts = []
        for part in parts:
            text = await transcribe_file(part)
            if text:
                texts.append(text.strip())

        raw_text = "\n".join(t for t in texts if t).strip()
        if not raw_text:
            raise HTTPException(422, "Transcripción vacía")

        # Realizar la diarización aquí
        diarized_text = await diarize_text_safely(raw_text)

        return {"conversacion": diarized_text}

    finally:
        try:
            for fn in os.listdir(tmp_dir):
                try:
                    os.remove(os.path.join(tmp_dir, fn))
                except:
                    pass
            os.rmdir(tmp_dir)
        except:
            pass
