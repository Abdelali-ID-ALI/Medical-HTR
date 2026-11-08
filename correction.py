import json
from google import genai
from google.genai import types

# 1. Initialisation du client avec la nouvelle syntaxe
# Remplace bien par ta clé API Google !
client = genai.Client(api_key="")

def structurer_ordonnance_gemini(texte_ocr):
    prompt = f"""Tu es un expert médical.
Texte OCR : {texte_ocr}

1. Corrige les fautes d'orthographe (ex: fepsis -> sepsis, natin -> matin).
2. Extrait les données en suivant STRICTEMENT ce schéma JSON.

Schéma attendu :
{{
  "patient": {{"nom": "...", "age": "..."}},
  "diagnostic": "...",
  "medicaments": [{{"nom": "...", "dosage": "...", "frequence": "..."}}]
}}"""

    print("🚀 Envoi à Gemini 2.5 Flash (Nouvelle API)...")
    
    try:
        # 2. Nouvel appel API avec le bon formateur JSON
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1
            ),
        )
        
        # 3. Conversion du texte JSON en dictionnaire Python
        return json.loads(response.text)
        
    except Exception as e:
        print(f"❌ Erreur lors de l'appel à l'API : {e}")
        return None

# --- TEST ---
texte_brut = "Mr. CH SAMUEL 62yr/Male Severe fepsis with MODS Paracetanol 1000ng 1 cpm natin"
resultat = structurer_ordonnance_gemini(texte_brut)

if resultat:
    print("\n✅ SUCCÈS (Gemini) :")
    print(json.dumps(resultat, indent=4, ensure_ascii=False))