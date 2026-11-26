from contextlib import asynccontextmanager
from threading import Lock
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from pipeline import Pipeline
from schemas import PrescriptionResult

MAX_BYTES = 10 * 1024 * 1024

@asynccontextmanager
async def lifespan(app):
    app.state.pipeline = None
    app.state.lock = Lock()
    try:
        yield
    finally:
        if app.state.pipeline is not None:
            app.state.pipeline.close()

app = FastAPI(title="Medical-HTR research prototype", version="2.0", lifespan=lifespan)

@app.get("/")
def root():
    return {"status": "online", "pipeline_loaded": app.state.pipeline is not None,
            "clinical_validation": False}

@app.post("/extract", response_model=PrescriptionResult)
def extract_prescription(request: Request, file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(415, "Expected JPEG or PNG")
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image too large")
    try:
        with request.app.state.lock:
            if request.app.state.pipeline is None:
                request.app.state.pipeline = Pipeline()
            return request.app.state.pipeline.run(data)
    except ValueError:
        raise HTTPException(422, "Invalid image or unsupported extraction result")
    except Exception:
        raise HTTPException(503, "Pipeline unavailable; check model, index and configuration")
