"""TrOCR inference with an explicit checkpoint and opt-in base fallback."""
from pathlib import Path
from PIL import Image
import cv2
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from config import BASE_MODEL, Settings

class PrescriptionReader:
    def __init__(self, settings: Settings):
        target = Path(settings.model_path)
        if target.is_dir() and (target / "config.json").is_file():
            model_name, self.model_mode = str(target), "fine_tuned_checkpoint"
        elif settings.allow_base_fallback:
            model_name, self.model_mode = BASE_MODEL, "base_model"
        else:
            raise FileNotFoundError("Fine-tuned checkpoint missing; restore it or explicitly enable base fallback.")
        self.model_name = model_name
        self.device = torch.device(settings.device)
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device).eval()

    @torch.inference_mode()
    def read_line(self, image):
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                image = opened.convert("RGB")
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pixels = self.processor(images=image.convert("RGB"), return_tensors="pt").pixel_values.to(self.device)
        generated = self.model.generate(pixels, max_new_tokens=128)
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
