from pathlib import Path
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from config import TROCR_MODEL_PATH, TROCR_BASE_MODEL, ALLOW_BASE_MODEL

class PrescriptionReader:
    def __init__(self, model_name=None):
        if model_name is None:
            if Path(TROCR_MODEL_PATH).is_dir():
                model_name = TROCR_MODEL_PATH
            elif ALLOW_BASE_MODEL:
                model_name = TROCR_BASE_MODEL
            else:
                raise FileNotFoundError("Fine-tuned checkpoint unavailable; configure TROCR_MODEL_PATH or explicitly enable ALLOW_BASE_MODEL.")
        self.model_id = model_name
        self.model_mode = "base" if model_name == TROCR_BASE_MODEL else "custom_checkpoint"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device).eval()

    @torch.inference_mode()
    def read_image(self, image):
        pixels = self.processor(images=image.convert("RGB"), return_tensors="pt").pixel_values.to(self.device)
        ids = self.model.generate(pixels)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0]

    def read_line(self, image_path):
        with Image.open(image_path) as image:
            return self.read_image(image)
