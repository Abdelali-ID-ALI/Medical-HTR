# Medical-HTR 🩺📝

**Medical-HTR** (Handwritten Text Recognition) est un pipeline d'intelligence artificielle de bout en bout conçu pour numériser, extraire et valider automatiquement le contenu des ordonnances médicales manuscrites. 

Le système combine un modèle de vision par ordinateur (TrOCR fine-tuné) pour la lecture de l'écriture manuscrite, et un système RAG (Retrieval-Augmented Generation) couplé à un LLM pour la validation médicale croisée avec la base de données DrugBank.

---

## 🏗️ Architecture du Projet

Le projet est divisé en deux environnements distincts : 
1. **Cloud (Google Colab)** : Phase d'entraînement intensif et préparation des données.
2. **Local (VS Code / FastAPI)** : Pipeline d'inférence, post-correction et déploiement de l'API.

### 📁 Structure du Code Source

```text
📦 Medical-HTR
 ┣ 📂 api/
 ┃ ┗ 📜 main.py                 # Point d'entrée de l'API FastAPI (Route POST /extract)
 ┣ 📂 Data/
 ┃ ┣ 📂 img/                    # Images d'ordonnances manuscrites pour les tests
 ┃ ┗ 📜 drugbank.xml            # Base de données brute des médicaments
 ┣ 📂 qdrant_db/                # Base de données vectorielle locale (générée via RAG.ipynb)
 ┃
 ┣ 🧠 Modèles & Notebooks (Cloud)
 ┃ ┣ 📜 fine.ipynb              # Entraînement et Fine-Tuning du modèle TrOCR (Vision)
 ┃ ┗ 📜 RAG.ipynb               # Vectorisation de DrugBank (PubMedBERT) et création de l'index Qdrant
 ┃
 ┣ ⚙️ Pipeline d'Inférence (Local)
 ┃ ┣ 📜 scanner.py              # Alignement et redressement de l'image de l'ordonnance
 ┃ ┣ 📜 preprocessing.py        # Nettoyage, binarisation et suppression du bruit (OpenCV)
 ┃ ┣ 📜 segmentation.py         # Extraction des lignes d'écriture individuelles
 ┃ ┣ 📜 extraction.py           # Inférence via le modèle TrOCR pour lire le texte
 ┃ ┣ 📜 correction.py           # Post-correction LLM (Gemini) et validation RAG
 ┃ ┗ 📜 ingest_drugbank.py      # Script local pour gérer la base de connaissances médicale
 ┃
 ┗ 📜 requirements.txt          # Liste des dépendances Python du projet
