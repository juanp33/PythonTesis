from fastapi import APIRouter, Form, File, UploadFile
from app.core.config import client
import tempfile, httpx, json
from pathlib import Path

router = APIRouter(prefix="/chat", tags=["Chat Jurídico"])
chat_history = {}
SPRINGBOOT_URL = "http://localhost:8080/api/messages"

SYSTEM_PROMPT = (
    "Eres un asistente jurídico especializado en legislación uruguaya. "
    "Debes responder como un abogado experto en derecho uruguayo. "
    "Si se menciona una ley o decreto (ej: 'Ley 12345'), devuelve un JSON con la acción de búsqueda."
)

async def persist_to_spring(chat_id, message, response):
    async with httpx.AsyncClient() as client_http:
        await client_http.post(SPRINGBOOT_URL, data={"chat_id": chat_id, "message": message, "response": response})

@router.post("/stream")
async def chat_stream(chat_id: str = Form(...), message: str = Form(...)):
    chat_history.setdefault(chat_id, [{"role": "system", "content": SYSTEM_PROMPT}])
    chat_history[chat_id].append({"role": "user", "content": message})

    response = client.chat.completions.create(model="gpt-5", messages=chat_history[chat_id])
    content = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(content)
        if parsed.get("action") == "fetch_page":
            url = parsed["url"]
            return {"response": f"Buscando contenido legal en: {url}"}
    except json.JSONDecodeError:
        pass

    chat_history[chat_id].append({"role": "assistant", "content": content})
    await persist_to_spring(chat_id, message, content)
    return {"response": content}
