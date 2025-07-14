from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydub import AudioSegment
from PIL import Image
import pytesseract
import whisper, tempfile, os
from pyannote.audio import Pipeline
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
import tempfile
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
import ocrmypdf
import os
whisper_model = whisper.load_model("small")

diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=os.getenv("HF_TOKEN")
)
origins = [
    "http://localhost:5173",  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,           
    allow_credentials=True,
    allow_methods=["*"],             
    allow_headers=["*"],              
)


@app.post("/transcribir_diarizado/")
async def transcribir_diarizado(audio: UploadFile = File(...)):
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(await audio.read())
        tmp.close()

        wav_path = tmp.name.replace(".mp3", ".wav")
        AudioSegment.from_file(tmp.name).set_channels(1).set_frame_rate(16000).export(wav_path, format="wav")

        diarization = diarization_pipeline(wav_path)
        transcription = whisper_model.transcribe(wav_path, language="es", word_timestamps=False)

  
        dialogue = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            texto_segmentado = ""
            for seg in transcription["segments"]:
        
                if seg["start"] < turn.end and seg["end"] > turn.start:
                    texto_segmentado += seg["text"].strip() + " "
            texto_segmentado = texto_segmentado.strip()
            if texto_segmentado:
                dialogue.append(f"{speaker}: {texto_segmentado}")

        os.remove(tmp.name)
        os.remove(wav_path)

        return {"conversacion": dialogue}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/ocr_archivo_con_texto/")
async def ocr_archivo_con_texto(archivo: UploadFile = File(...)):
    try:
        extension = archivo.filename.split(".")[-1].lower()

   
        tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}")
        tmp_input.write(await archivo.read())
        tmp_input.close()

       
        if extension in ["jpg", "jpeg", "png"]:
            imagen = Image.open(tmp_input.name).convert("RGB")
            tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            imagen.save(tmp_pdf.name)
            input_path = tmp_pdf.name
            os.remove(tmp_input.name)  
        elif extension == "pdf":
            input_path = tmp_input.name
        else:
            return JSONResponse(status_code=400, content={"error": "Formato no soportado"})

     
        output_path = input_path.replace(".pdf", "_ocr.pdf")
        ocrmypdf.ocr(input_path, output_path, language="spa", deskew=True)

        return FileResponse(output_path, filename="archivo_ocr.pdf")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})