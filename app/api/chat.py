from fastapi import APIRouter, Form, File, UploadFile, HTTPException
from app.core.config import client
from pathlib import Path
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from PIL import Image
import tempfile, httpx, json, re, os, mimetypes, ocrmypdf
import docx2txt
router = APIRouter(prefix="/chat", tags=["Chat Jurídico"])

chat_history: Dict[str, List[Dict[str, str]]] = {}
SPRINGBOOT_URL = "http://localhost:8080/api/messages"
MODEL = os.getenv("OAI_MODEL", "gpt-5")

SYSTEM_PROMPT = """Eres un asistente jurídico especializado en legislación uruguaya.
Debes responder como un abogado experto en derecho uruguayo, citando leyes, decretos, códigos y artículos aplicables.

CUANDO el usuario mencione una norma por número o nombre (ley, decreto o CÓDIGO), NO expliques nada:
devuelve SOLO un JSON con el formato EXACTO {"action":"fetch_page","url":"<URL_IMPO_CANONICA>","keyword":"<norma_detectada>"}
sin texto adicional.

Reglas de URL IMPO:
1) LEYES → https://www.impo.com.uy/bases/leyes/{numero}-{AAAA}
   - Si hay año explícito de 4 dígitos, úsalo.
   - Si hay sufijo de 2 o 3 dígitos tras '/' o '-', interprétalo como 2000+ sufijo (p.ej., 12→2012, 012→2012).
2) DECRETOS → https://www.impo.com.uy/bases/decretos/{numero}-{sufijo_año}
   - Si el mensaje trae 'N/AAAA' o 'N-AAAA', usar '-AAAA'.
   - Si trae 'N/AAA' (p.ej., '375/012'), conservar '-012'.
   - Si trae 'N/YY', interpretar como 2000+YY y usar '-AAAA' (p.ej., 12→2012).
3) CÓDIGOS (mapa fijo):
   - Código Penal (CP) → https://www.impo.com.uy/bases/codigo-penal/9155-1933
   - Código del Proceso Penal (CPP) → https://www.impo.com.uy/bases/codigo-proceso-penal-2017/19293-2014
   - Código General del Proceso (CGP) → https://www.impo.com.uy/bases/codigo-general-proceso/15982-1988
   - Código Civil (CC) → https://www.impo.com.uy/bases/codigo-civil/16603-1994
   - Código de la Niñez y la Adolescencia (CNA) → https://www.impo.com.uy/bases/codigo-ninez-adolescencia/17823-2004
   - Código Tributario (CT) → https://www.impo.com.uy/bases/codigo-tributario/14306-1974
   - Código de Comercio (CCom) → https://www.impo.com.uy/bases/codigo-comercio/817-1865

4) Si no se menciona una norma concreta, responde normalmente en texto plano.

EJEMPLOS:
Entrada: 'ley 18987' →
{"action":"fetch_page","url":"https://www.impo.com.uy/bases/leyes/18987-2012","keyword":"18987"}
Entrada: 'art 18 del Código Penal' →
{"action":"fetch_page","url":"https://www.impo.com.uy/bases/codigo-penal/9155-1933","keyword":"Código Penal art. 18"}
Entrada: 'CPP art. 129' →
{"action":"fetch_page","url":"https://www.impo.com.uy/bases/codigo-proceso-penal-2017/19293-2014","keyword":"CPP art. 129"}
"""

async def persist_to_spring(chat_id: str, message: str, response: str, files: List[Path]) -> None:
    import mimetypes, httpx
    url = f"{SPRINGBOOT_URL}/save"  # SPRINGBOOT_URL = "http://localhost:8080/api/messages"
    parts = [
        ("chatId", (None, chat_id)),
        ("userMessage", (None, message)),
        ("assistantResponse", (None, response)),
    ]
    for path in files:
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        parts.append(("files", (path.name, path.read_bytes(), ctype)))

    async with httpx.AsyncClient(timeout=20) as http_client:
        r = await http_client.post(url, files=parts)
        r.raise_for_status()

async def scrape_page(url: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=10) as hc:
            r = await hc.get(url)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            text = soup.get_text(separator="\n")
            return text[:15000]
    except Exception as e:
        return f"Error al intentar acceder a {url}: {e}"

def parse_json_loose(content: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            return None
    compact = content.strip()
    if compact.startswith("{") and compact.endswith("}"):
        try:
            return json.loads(compact)
        except json.JSONDecodeError:
            return None
    return None

def flatten_history_to_text(history: List[Dict[str, str]]) -> str:
    parts = []
    for m in history:
        role = m.get("role", "")
        if role == "system":
            continue
        parts.append(f"{role.upper()}: {m.get('content','')}")
    return "\n".join(parts) if parts else ""

def _find_year(*texts: str) -> Optional[int]:
    for t in texts:
        if not t:
            continue
        m = re.search(r'\b(19|20)\d{2}\b', t)
        if m:
            return int(m.group(0))
        m2 = re.search(r'[/-]\s*([0-9]{2,3})\b', t)
        if m2:
            frag = m2.group(1)
            if len(frag) == 3 and frag[0] == "0":
                return 2000 + int(frag)
            if len(frag) == 2:
                return 2000 + int(frag)
    return None

def _add_year_if_law(url: str, year: Optional[int], number_hint: Optional[str]) -> str:
    m = re.match(r'^https?://www\.impo\.com\.uy/bases/leyes/([0-9]+)$', url.strip())
    if not m:
        return url
    num = m.group(1)
    y = year
    if y is None and number_hint == "18987":
        y = 2012
    return f"{url}-{y}" if y else url

def _html_to_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = re.sub(r'<br\s*/?>', '\n', s, flags=re.I)
    s = re.sub(r'</p\s*>', '\n', s, flags=re.I)
    return re.sub(r'<[^>]+>', '', s)

def _extract_article_number(keyword: Optional[str]) -> Optional[str]:
    if not keyword:
        return None
    m = re.search(r'art\.?\s*(\d+)', keyword, flags=re.I)
    return m.group(1) if m else None

async def fetch_norma_json(url: str) -> Dict[str, Any]:
    try:
        url_json = url if url.endswith("?json=true") else f"{url}?json=true"
        async with httpx.AsyncClient(timeout=15) as hc:
            r = await hc.get(url_json, headers={"Accept": "application/json"})
            r.raise_for_status()
            return r.json()
    except Exception as e:
        return {"_error": f"No se pudo obtener JSON de IMPO: {e}", "_url": url}

def compose_norma_text(data: Dict[str, Any], article_hint: Optional[str] = None) -> str:
    if not data or "_error" in data:
        return ""
    lines: List[str] = []
    tipo = data.get("tipoNorma") or ""
    nro = str(data.get("nroNorma") or "").strip()
    anio = str(data.get("anioNorma") or "").strip()
    nombre = (data.get("nombreNorma") or "").strip()
    leyenda = (data.get("leyenda") or "").strip()
    fprom = (data.get("fechaPromulgacion") or "").strip()
    fpub = (data.get("fechaPublicacion") or "").strip()
    firmantes = _html_to_text(data.get("firmantes"))
    encabezado = f"{tipo} {nro}-{anio}: {nombre}".strip()
    lines.append(encabezado)
    meta: List[str] = []
    if fprom:
        meta.append(f"Promulgación: {fprom}")
    if fpub:
        meta.append(f"Publicación: {fpub}")
    if leyenda:
        meta.append(leyenda.strip())
    if meta:
        lines.append(" | ".join(meta))
    if firmantes:
        lines.append(f"Firmantes: {firmantes}")
    articulos = data.get("articulos") or []
    target_only = None
    if article_hint:
        for a in articulos:
            nroA = str(a.get("nroArticulo") or "").strip()
            if nroA == str(article_hint):
                target_only = a
                break
    def _push_art(a: Dict[str, Any]) -> None:
        nroA = str(a.get("nroArticulo") or "").strip()
        titulo = (a.get("tituloArticulo") or "").strip()
        texto = _html_to_text(a.get("textoArticulo"))
        notas = _html_to_text(a.get("notasArticulo"))
        header = f"Artículo {nroA}" + (f" – {titulo}" if titulo else "")
        lines.append(header)
        if texto:
            lines.append(texto.strip())
        if notas:
            lines.append(f"Notas: {notas.strip()}")
    if target_only:
        _push_art(target_only)
    else:
        for a in articulos:
            _push_art(a)
            if len("\n".join(lines)) > 15000:
                break
    text = "\n".join([l for l in lines if l is not None]).strip()
    return text[:15000]

def _read_textfile(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            return path.read_text(encoding="latin-1", errors="ignore")
        except Exception:
            return ""

def _extract_pdf_simple(path: Path) -> str:
    try:
        from pypdf import PdfReader
        r = PdfReader(str(path))
        return "\n".join((p.extract_text() or "") for p in r.pages)
    except Exception:
        try:
            from pdfminer.high_level import extract_text
            return extract_text(str(path))
        except Exception:
            return ""

def _to_pdf_for_ocr(src: Path, tmpdir: Path) -> Path:
    if src.suffix.lower() == ".pdf":
        return src
    img = Image.open(str(src)).convert("RGB")
    dst = tmpdir / f"{src.stem}.pdf"
    img.save(str(dst), "PDF")
    return dst

def _ocr_pdf_to_text(src: Path, tmpdir: Path) -> str:
    pdf_in = _to_pdf_for_ocr(src, tmpdir)
    pdf_out = tmpdir / "ocr_output.pdf"
    try:
        ocrmypdf.ocr(str(pdf_in), str(pdf_out), force_ocr=True, output_type="pdf")
        return _extract_pdf_simple(pdf_out)
    except Exception:
        return ""

def extract_text_from_path(path: Path, tmpdir: Path) -> str:
    ext = path.suffix.lower()
    if ext in [".txt", ".csv", ".md", ".log"]:
        text = _read_textfile(path)
    elif ext in [".pdf"]:
        text = _extract_pdf_simple(path)
        if len((text or "").strip()) < 500:
            text = _ocr_pdf_to_text(path, tmpdir) or text
    elif ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]:
        text = _ocr_pdf_to_text(path, tmpdir)
    elif ext in [".html", ".htm"]:
        try:
            soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
            text = soup.get_text(separator="\n")
        except Exception:
            text = ""
    elif ext == ".rtf":
        try:
            s = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            s = path.read_text(encoding="latin-1", errors="ignore")
        s = re.sub(r'\\par[d]?', '\n', s)
        s = re.sub(r'{\\[^}]+}', '', s)
        s = re.sub(r'\\[a-zA-Z]+\d* ?', '', s)
        s = s.replace('{', '').replace('}', '')
        text = s
    elif ext in [".docx"]:
        try:
            
            text = docx2txt.process(str(path)) or ""
        except Exception:
            try:
                import docx
                d = docx.Document(str(path))
                text = "\n".join(p.text for p in d.paragraphs)
            except Exception:
                text = ""
    else:
        mime, _ = mimetypes.guess_type(str(path))
        if mime and mime.startswith("text/"):
            text = _read_textfile(path)
        else:
            text = ""
    text = re.sub(r'\u0000', '', text or "")
    return text[:20000]

@router.post("/stream")
async def chat_stream(
    chat_id: str = Form(...),
    message: str = Form(...),
    files: Optional[List[UploadFile]] = File(default=None)
):
    try:
        tmpdir = Path(tempfile.mkdtemp(prefix="chat_"))
        saved_files: List[Path] = []
        for file in files or []:
            path = tmpdir / file.filename
            path.write_bytes(await file.read())
            saved_files.append(path)

        extracted_blocks: List[str] = []
        for p in saved_files:
            t = extract_text_from_path(p, tmpdir)
            if t:
                extracted_blocks.append(f"--- Archivo: {p.name} ---\n{t}")
        attachments_text = "\n\n".join(extracted_blocks)
        if len(attachments_text) > 40000:
            attachments_text = attachments_text[:40000]

        chat_history.setdefault(chat_id, [{"role": "system", "content": SYSTEM_PROMPT}])
        chat_history[chat_id].append({"role": "user", "content": message})

        transcript = flatten_history_to_text(chat_history[chat_id])
        base_input = transcript if transcript else message
        if attachments_text:
            base_input = f"{base_input}\n\n[Archivos adjuntos]\n{attachments_text}"

        first = client.responses.create(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            input=base_input,
        )
        content = (getattr(first, "output_text", "") or "").strip()

        parsed = parse_json_loose(content)
        if parsed and parsed.get("action") == "fetch_page" and parsed.get("url"):
            url_in = str(parsed["url"])
            keyword = str(parsed.get("keyword", ""))

            year = _find_year(keyword, message)
            num_hint = re.sub(r'\D+', '', keyword or '')
            url = _add_year_if_law(url_in, year, num_hint)

            norma_json = await fetch_norma_json(url)
            art_hint = _extract_article_number(keyword)
            norma_text = compose_norma_text(norma_json, art_hint)

            if not norma_text:
                scraped_text = await scrape_page(url)
                source_text = scraped_text
                source_kind = "HTML"
            else:
                source_text = norma_text
                source_kind = "JSON"

            follow_input = f"Origen: {source_kind} de {url}\nPalabra clave: {keyword}\n\nContenido:\n{source_text}"
            if attachments_text:
                follow_input = f"{follow_input}\n\n[Archivos adjuntos]\n{attachments_text}"

            followup = client.responses.create(
                model=MODEL,
                instructions="Eres un abogado uruguayo: resume y explica el texto legal proporcionado, citando artículos y fechas cuando corresponda.",
                input=follow_input,
            )
            full_response = getattr(followup, "output_text", "") or ""
        else:
            full_response = content

        chat_history[chat_id].append({"role": "assistant", "content": full_response})
        await persist_to_spring(chat_id, message, full_response, saved_files)

        return {"response": full_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en chat_stream: {e}")
