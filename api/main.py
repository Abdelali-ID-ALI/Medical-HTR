from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import time

# 1. Initialisation de l'application FastAPI
app = FastAPI(
    title="Medical HTR API",
    description="API d'extraction et de validation d'ordonnances médicales manuscrites.",
    version="1.0.0"
)

# 2. Route de vérification de santé (Health Check)
@app.get("/")
async def root():
    return {"status": "online", "message": "Medical HTR API is running."}

# 3. La Route principale : Réception de l'image et Extraction
@app.post("/extract")
async def extract_prescription(file: UploadFile = File(...)):
    # Vérification que le fichier est bien une image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image (JPEG, PNG).")

    try:
        # Lecture de l'image en mémoire (sans la sauvegarder sur le disque !)
        image_bytes = await file.read()
        
        print(f"📥 Reçu : {file.filename} ({len(image_bytes)} bytes)")
        
        # =====================================================================
        # 🚧 ZONE D'INTÉGRATION DU PIPELINE IA 🚧
        # Ici, nous appellerons plus tard : 
        # resultat = run_full_pipeline(image_bytes)
        # =====================================================================
        
        # SIMULATION DU TRAITEMENT (Pour tester l'API)
        time.sleep(1.5) # Simule le temps de calcul d'OpenCV + TrOCR + RAG
        
        mock_result = {
            "patient": {"nom": "Non détecté", "age": "Non détecté"},
            "diagnostic": "Fièvre",
            "medicaments": [
                {
                    "nom_extrait": "Lepirudin",
                    "dosage_extrait": "50mg",
                    "frequence": "1 injection",
                    "validation_rag": {
                        "statut": "VERIFIED",
                        "molecule_officielle": "Lepirudin",
                        "alerte_drugbank": "No reported interactions."
                    }
                }
            ],
            "metadonnees": {
                "confiance_ocr_globale": 0.89,
                "temps_traitement_ms": 1500
            }
        }
        
        return JSONResponse(content=mock_result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne lors du traitement : {str(e)}")