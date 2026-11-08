from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import time

class PrescriptionReader:
    def __init__(self, model_name="microsoft/trocr-base-handwritten"):
        print(f"Chargement du modèle {model_name}... (Cela peut prendre 1-2 minutes la première fois)")
        # Le Processor prépare l'image (redimensionnement, normalisation)
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        # Le Model est le réseau de neurones (ViT + RoBERTa)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        print("Modèle chargé et prêt !")

    def read_line(self, image_path):
        # Charger l'image avec PIL (Pillow) au lieu d'OpenCV
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Erreur lors du chargement de l'image: {e}")
            return None

        print(f"Analyse de l'image : {image_path}...")
        start_time = time.time()

        # 1. Préparation des pixels pour le modèle Transformer
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values

        # 2. Génération du texte (Inférence)
        generated_ids = self.model.generate(pixel_values)

        # 3. Décodage des IDs générés en texte lisible
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        end_time = time.time()
        print(f"Temps d'inférence : {round(end_time - start_time, 2)} secondes")
        
        return generated_text

if __name__ == "__main__":
    # Instanciation
    reader = PrescriptionReader()
    
    # ATTENTION : Mets ici le chemin vers une image contenant UNE SEULE LIGNE de texte !
    image_ligne = "/home/Abdelali/Pictures/med/Screenshot_20260415_175450.png"
    
    texte_extrait = reader.read_line(image_ligne)
    
    print("\n" + "="*50)
    print("RÉSULTAT DE L'EXTRACTION :")
    print(texte_extrait)
    print("="*50 + "\n")