"""Runtime configuration; no model loading on import."""
import os
from dotenv import load_dotenv

load_dotenv()
TROCR_MODEL_PATH = os.getenv("TROCR_MODEL_PATH", "./models/trocr-medical-finetuned")
TROCR_BASE_MODEL = "microsoft/trocr-base-handwritten"
ALLOW_BASE_MODEL = os.getenv("ALLOW_BASE_MODEL", "false").lower() == "true"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "NeuML/pubmedbert-base-embeddings")
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_db")
COLLECTION = "drugbank_names_v2"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
ALLOW_EXTERNAL_LLM = os.getenv("ALLOW_EXTERNAL_LLM", "false").lower() == "true"
