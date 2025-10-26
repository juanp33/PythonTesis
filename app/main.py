# app/main.py
import logging, sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import api_router

# ---- Logging global a stdout ----
root = logging.getLogger()
if not root.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"))
    root.addHandler(h)
root.setLevel(logging.DEBUG)  # o INFO

log = logging.getLogger("app.main")
log.info("🚀 main.py cargado")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montás TODOS tus routers (incluido transcripción)
app.include_router(api_router)

@app.on_event("startup")
async def startup():
    log.info("✅ Startup OK (routers montados)")

@app.get("/health")
def health():
    log.info("🩺 /health")
    return {"ok": True}