from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from main import analyze, extract_bytes

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MAX_PDF_BYTES = 5 * 1024 * 1024  # 5 MB

app = FastAPI(title="Resume Analyzer")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


async def read_pdf(upload: UploadFile, label: str) -> str:
    if not (upload.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, f"{label} must be a PDF file.")
    data = await upload.read()
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(413, f"{label} is larger than 5 MB.")
    try:
        text = extract_bytes(data)
    except Exception:
        raise HTTPException(400, f"Could not read {label}. Is it a valid PDF?")
    if not text.strip():
        raise HTTPException(
            422, f"No text found in {label}. Scanned PDFs need OCR first."
        )
    return text


@app.post("/api/match")
async def match_endpoint(
    cv: UploadFile = File(...),
    job: Optional[UploadFile] = File(None),
    job_text: str = Form(""),
):
    cv_text = await read_pdf(cv, "CV")

    if job is not None and job.filename:
        job_content = await read_pdf(job, "Job description")
    elif job_text.strip():
        job_content = job_text
    else:
        raise HTTPException(400, "Provide a job description (PDF or pasted text).")

    return analyze(cv_text, job_content)