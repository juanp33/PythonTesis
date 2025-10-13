from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import api_router
from app.core.config import origins

# ================================
#  Inicialización de la aplicación
# ================================

app = FastAPI(
    title="Asistente Jurídico IA",
    description="API backend para la generación, resumen, OCR y redacción de documentos legales en Uruguay.",
    version="1.0.0",
)

# ================================
#  Configuración de CORS
# ================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,           # viene de app/core/config.py
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
#  Registro de rutas
# ================================

app.include_router(api_router)

# ================================
#  Ruta raíz (para comprobar el estado)
# ================================

@app.get("/")
async def root():
    """
    Endpoint base para verificar que la API está activa.
    """
    return {
        "status": "✅ OK",
        "message": "Asistente Jurídico IA funcionando correctamente",
        "docs": "Visita /docs para ver la documentación interactiva",
    }
