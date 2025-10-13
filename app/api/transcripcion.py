from fastapi import APIRouter, UploadFile, File, HTTPException
from app.core.config import client
from pydub import AudioSegment
import tempfile, os

router = APIRouter(prefix="/transcripcion", tags=["Transcripción"])

@router.post("/")
async def transcribir_audio(audio: UploadFile = File(...)):
    tmp_path = os.path.join(tempfile.gettempdir(), audio.filename)
    with open(tmp_path, "wb") as f:
        f.write(await audio.read())

    wav_path = tmp_path + ".wav"
    try:
        sound = AudioSegment.from_file(tmp_path)
        sound.export(wav_path, format="wav", codec="pcm_s16le")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error convirtiendo audio: {e}")

    with open(wav_path, "rb") as f:
        transcription = client.audio.transcriptions.create(model="gpt-4o-transcribe", file=f)
    return {"texto": transcription.text.strip()}
