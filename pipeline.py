"""In-process orchestration; no fake output and no clinical validation."""
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from config import Settings
from scanner import scan_document
from preprocessing import PrescriptionPreprocessor
from segmentation import LineSegmentor
from extraction import PrescriptionReader
from correction import GeminiExtractor
from retrieval import DrugRetriever
from schemas import Line, Report

class InvalidImage(ValueError):
    pass

class PrescriptionPipeline:
    def __init__(self, settings: Settings):
        self.reader = PrescriptionReader(settings)
        self.extractor = GeminiExtractor(settings)
        self.retriever = DrugRetriever(settings.qdrant_path)

    def run(self, data: bytes) -> Report:
        try:
            with Image.open(BytesIO(data)) as image:
                if image.format not in {"JPEG", "PNG"} or image.width * image.height > 20_000_000:
                    raise InvalidImage("Unsupported format or image too large.")
                if min(image.size) < 32:
                    raise InvalidImage("Image too small.")
                bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        except InvalidImage:
            raise
        except Exception as exc:
            raise InvalidImage("Invalid image.") from exc
        scanned = scan_document(bgr, debug=False)
        h, w = scanned.shape[:2]
        segmentor = LineSegmentor(kernel_width=max(5, round(w * .05)),
                                  min_line_width=max(5, round(w * .02)),
                                  min_line_height=max(5, round(h * .005)))
        segments = segmentor.segment(scanned, prepare_trocr=False, include_rejected=True,
                                     preprocessor=PrescriptionPreprocessor(block_size=21, c=15))
        lines, warnings = [], []
        if self.reader.model_mode == "base_model":
            warnings.append("Generic TrOCR fallback: historical fine-tuned CER does not apply.")
        if not segments:
            warnings.append("No text regions detected; manual review needed.")
        for segment in segments:
            accepted = segment["accepted"]
            text = self.reader.read_line(segment["image"]) if accepted else None
            if not accepted or not text:
                warnings.append(f"Line {segment['index']} unread or rejected.")
            lines.append(Line(index=segment["index"], bbox=segment["bbox"],
                              quality_score=segment["quality_score"], text=text, accepted=accepted))
        mentions, matches = [], []
        if any(line.text for line in lines):
            mentions, extraction_warnings = self.extractor.extract(lines)
            warnings.extend(extraction_warnings)
            matches = [self.retriever.search(mention) for mention in mentions]
        if not mentions:
            warnings.append("No medication mentions extracted; absence is not confirmed.")
        return Report(model_mode=self.reader.model_mode,
                      raw_text="\n".join(line.text for line in lines if line.text),
                      lines=lines, medicines=matches, warnings=warnings)

    def close(self):
        self.retriever.close()
