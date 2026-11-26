from contextlib import asynccontextmanager
from threading import Lock
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from starlette.concurrency import run_in_threadpool
from config import Settings
from schemas import Report, TextRequest
from pipeline import PrescriptionPipeline, InvalidImage

logger = logging.getLogger(__name__)
MAX_BYTES = 10 * 1024 * 1024
lock = Lock()

@asynccontextmanager
async def lifespan(app):
    app.state.pipeline = None
    try:
        yield
    finally:
        if app.state.pipeline is not None:
            app.state.pipeline.close()

app = FastAPI(title="Medical-HTR", version="2.0.0", lifespan=lifespan,
              description="Experimental transcription and candidate retrieval, not clinical validation.")

@app.get("/")
def root():
    return {"status": "online", "pipeline_loaded": app.state.pipeline is not None}

def process(data, text_only=False):
    # Local Qdrant and model access are serialized in this single-worker prototype.
    with lock:
        if app.state.pipeline is None:
            try:
                app.state.pipeline = PrescriptionPipeline(Settings())
            except Exception as exc:
                logger.error("Pipeline initialization failed (%s)", type(exc).__name__)
                raise HTTPException(503, "Pipeline unavailable; check local configuration and artifacts.") from exc
        if text_only:
            return app.state.pipeline.run_text(data)
        try:
            return app.state.pipeline.run(data)
        except (FileNotFoundError, OSError) as exc:
            raise HTTPException(503, "HTR artifacts unavailable; check model configuration.") from exc

@app.post("/extract", response_model=Report)
async def extract_prescription(file: UploadFile = File(...)):
    try:
        if file.content_type not in {"image/jpeg", "image/png"}:
            raise HTTPException(415, "JPEG or PNG required.")
        data = await file.read(MAX_BYTES + 1)
        if not data or len(data) > MAX_BYTES:
            raise HTTPException(413, "Empty file or size limit exceeded.")
        return await run_in_threadpool(process, data)
    except InvalidImage as exc:
        raise HTTPException(400, str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Extraction failed (%s)", type(exc).__name__)
        raise HTTPException(500, "Extraction failed; no result was fabricated.") from exc
    finally:
        await file.close()

@app.post("/extract-text", response_model=Report)
async def extract_text(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(422, "Non-empty text required.")
    try:
        return await run_in_threadpool(process, request.text, True)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Text extraction failed (%s)", type(exc).__name__)
        raise HTTPException(500, "Extraction failed; no result was fabricated.") from exc
