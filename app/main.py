# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import api_router

# Crear la aplicación principal
app = FastAPI(
    title="Asistente IA Legal",
    description="API del proyecto PythonTesis - sistema inteligente para abogados",
    version="1.0.0"
)

# Configuración de CORS (permitir peticiones desde el frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Podés restringirlo a tu dominio si querés más adelante
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar todas las rutas definidas en app/api/router.py
app.include_router(api_router)

# Endpoint de prueba para ver si el servidor responde
@app.get("/")
def read_root():
    return {"message": "Servidor en funcionamiento ✅"}

