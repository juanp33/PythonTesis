from fastapi import APIRouter
from app.api import ocr, resumen, redaccion, transcripcion, conversion, chat


api_router = APIRouter()

api_router.include_router(ocr.router)
api_router.include_router(resumen.router)
api_router.include_router(redaccion.router)
api_router.include_router(transcripcion.router)
api_router.include_router(conversion.router)
api_router.include_router(chat.router)
from fastapi import APIRouter
import logging

api_router = APIRouter()

routers = [
    "ocr",
    "resumen",
    "redaccion",
    "transcripcion",
    "conversion",
    "chat"
]

# Logger para monitorear la inclusión de routers
log_api_router = logging.getLogger("app.api_router")
log_api_router.setLevel(logging.INFO)

for router_name in routers:
    try:
        # Intentamos obtener el router usando su nombre
        router = globals()[router_name].router
        api_router.include_router(router)
        log_api_router.info(f"🚀 {router_name} router montado con éxito.")
    except KeyError as e:
        log_api_router.error(f"❌ Error al incluir router {router_name}: {str(e)}")