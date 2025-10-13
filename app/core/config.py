import os
from pathlib import Path
from openai import OpenAI

# === Directorios base ===
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = BASE_DIR / "pdfs"

# Crear carpetas si no existen
DATA_DIR.mkdir(exist_ok=True)
PDF_DIR.mkdir(exist_ok=True)

# === Variables de entorno ===
API_KEY = os.getenv("APIGPT") or "TU_API_KEY_AQUI"  # 🔒 Reemplazar o definir en .env
HF_TOKEN = os.getenv("HF_TOKEN")

# Cliente OpenAI
client = OpenAI(api_key=API_KEY)

# === Configuración de CORS ===
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# === Plantillas base de documentos ===
DEFAULT_TEMPLATES = {
    "Demanda": str(BASE_DIR / "templates/demanda_base.txt"),
    "Poder": str(BASE_DIR / "templates/poder_base.txt"),
    "Escrito": str(BASE_DIR / "templates/escrito_proceso_iniciado.txt"),
    "Testamento": str(BASE_DIR / "templates/testamento_base.txt"),
    "Declaración Jurada": str(BASE_DIR / "templates/declaracion_jurada_base.txt"),
}

# === Configuración general ===
SPRINGBOOT_URL = "http://localhost:8080/api/messages"
