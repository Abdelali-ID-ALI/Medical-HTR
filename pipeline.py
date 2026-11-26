"""Orchestrate existing vision components and V2 extraction."""
import tempfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scanner import scan_document
from segmentation import LineSegmentor
from extraction import PrescriptionReader
from correction import structurer_ordonnance_gemini
from retrieval import DrugRetriever
from config import ALLOW_EXTERNAL_LLM
from schemas import LineResult, LinkedMention, PrescriptionResult

class Pipeline:
    def __init__(self):
        self.reader = PrescriptionReader()
        self.retriever = DrugRetriever() if ALLOW_EXTERNAL_LLM else None

    def close(self):
        if self.retriever:
            self.retriever.close()

    def run(self, image_bytes):
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError("Invalid image")
        with tempfile.TemporaryDirectory() as directory:
            raw = str(Path(directory) / "input.png")
            scan = str(Path(directory) / "scan.png")
            if not cv2.imwrite(raw, image):
                raise RuntimeError("Image write failed")
            scanned = scan_document(raw, debug=False)
            if not cv2.imwrite(scan, scanned):
                raise RuntimeError("Scan write failed")
            # Segmentation already performs its own binarization.
            segments = LineSegmentor().segment(scan, prepare_trocr=False)
        if not segments:
            raise ValueError("No text segments detected")
        lines = [LineResult(index=s["index"], bbox=list(s["bbox"]),
                 quality_score=s["quality_score"],
                 text=self.reader.read_image(Image.fromarray(cv2.cvtColor(s["image"], cv2.COLOR_BGR2RGB))))
                 for s in segments]
        mentions = []
        status = "external_llm_disabled"
        if ALLOW_EXTERNAL_LLM:
            extraction = structurer_ordonnance_gemini([line.model_dump() for line in lines])
            for mention in extraction.medicaments:
                candidates = self.retriever.search(mention.nom)
                mentions.append(LinkedMention(mention=mention, candidates=candidates,
                    status="candidates_found" if candidates else "abstained"))
            status = "extracted"
        return PrescriptionResult(model_id=self.reader.model_id, model_mode=self.reader.model_mode,
            lines=lines, raw_text="\n".join(line.text for line in lines),
            medicaments=mentions, extraction_status=status)
