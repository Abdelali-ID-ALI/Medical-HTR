"""Runtime configuration; never download or initialize models on import."""
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
BASE_MODEL = "microsoft/trocr-base-handwritten"
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
COLLECTION = "drugbank_pubmedbert"

def flag(name: str) -> bool:
    return os.getenv(name, "false").lower() in {"true", "1", "yes"}

@dataclass(frozen=True)
class Settings:
    model_path: str = os.getenv("TROCR_MODEL_PATH", str(ROOT / "models/trocr-medical-finetuned"))
    allow_base_fallback: bool = flag("TROCR_ALLOW_BASE_FALLBACK")
    device: str = os.getenv("TROCR_DEVICE", "cpu")
    gemini_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    allow_external_text: bool = flag("GEMINI_ALLOW_EXTERNAL_TEXT")
    qdrant_path: str = os.getenv("QDRANT_PATH", str(ROOT / "qdrant_medical"))
