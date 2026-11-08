import xml.etree.ElementTree as ET
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document

XML_FILE = "drugbank.xml"
LIMIT = 10  # Toujours 10 pour le test

def parse_drugbank(file_path, limit):
    documents = []
    count = 0
    
    # L'arme secrète : on déclare le namespace officiel de DrugBank
    ns = {'db': 'http://www.drugbank.ca'}
    
    print(f"⏳ Lecture en streaming de {file_path} (Limite : {limit} médicaments)...")
    context = ET.iterparse(file_path, events=('end',))
    
    for event, elem in context:
        # On cible explicitement la balise "drug" avec son namespace
        if elem.tag == '{http://www.drugbank.ca}drug':
            
            # 1. Extraction de l'ID (Méthode XPath avec namespace)
            # 1. Extraction de l'ID (Méthode robuste en boucle)
            drug_id = "Unknown_ID"
            for id_node in elem.findall("./db:drugbank-id", ns):
                if id_node.get('primary') == 'true':
                    drug_id = id_node.text
                    break
            
            # 2. Extraction du Nom
            name_node = elem.find("./db:name", ns)
            name = name_node.text if name_node is not None else "Unknown_Name"
            
            # 3. Extraction des Dosages
            dosages = []
            # On cherche partout (.//) les balises strength à l'intérieur de dosage
            for strength_node in elem.findall(".//db:dosage/db:strength", ns):
                if strength_node.text:
                    dosages.append(strength_node.text)
            
            dosages = list(set(dosages)) # Dédoublonnage
            dosages_str = ", ".join(dosages) if dosages else "Not specified"

            # 4. Extraction des Interactions
            interactions = []
            for inter_node in elem.findall(".//db:drug-interaction", ns)[:3]:
                inter_name = inter_node.find("./db:name", ns)
                inter_desc = inter_node.find("./db:description", ns)
                if inter_name is not None and inter_desc is not None:
                    interactions.append(f"{inter_name.text} ({inter_desc.text})")
            
            interactions_str = " | ".join(interactions) if interactions else "No reported interactions."

            # 5. Création du Document pour Qdrant
            page_content = f"Drug: {name}. Standard dosages: {dosages_str}. Safety and Interactions: {interactions_str}"
            metadata = {
                "id": drug_id,
                "name": name,
                "known_dosages": dosages_str
            }

            documents.append(Document(page_content=page_content, metadata=metadata))
            
            # Nettoyage de la RAM
            elem.clear()
            count += 1
            
            if count >= limit:
                break
                
    return documents

# --- EXÉCUTION ---
docs = parse_drugbank(XML_FILE, LIMIT)

print(f"\n✅ {len(docs)} médicaments extraits. Voici un aperçu du premier :")
print("-" * 50)
print(f"Contenu vectorisé : \n{docs[0].page_content}")
print(f"Métadonnées : {docs[0].metadata}")
print("-" * 50)

print("\n🧠 Chargement du modèle de vectorisation Sentence-Transformers...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

DB_PATH = "./qdrant_db"

print(f"\n💾 Sauvegarde de la base vectorielle dans : {DB_PATH}")
qdrant_db = Qdrant.from_documents(
    docs,
    embeddings,
    path=DB_PATH,
    collection_name="drugbank_v1",
)

print("\n🚀 SUCCESS : Pipeline RAG initialisé avec succès !")