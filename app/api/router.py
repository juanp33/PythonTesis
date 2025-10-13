from fastapi import APIRouter
from app.api import ocr, resumen, redaccion, transcripcion, conversion, chat

api_router = APIRouter()

api_router.include_router(ocr.router)
api_router.include_router(resumen.router)
api_router.include_router(redaccion.router)
api_router.include_router(transcripcion.router)
api_router.include_router(conversion.router)
api_router.include_router(chat.router)
