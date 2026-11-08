import pandas as pd
import json
import os

def prepare_trocr_csv(csv_path, json_path, output_path, image_col='filename', label_col='label'):
    print(f"Traitement de {csv_path}...")
    
    # 1. Charger le dictionnaire de mapping original
    with open(json_path, 'r', encoding='utf-8') as f:
        original_mapping = json.load(f)
    
    # --- LA MODIFICATION MAGIQUE EST ICI ---
    # On inverse le dictionnaire : {"Aceta": 0} devient {"0": "Aceta"}
    # On s'assure que les clés sont des chaînes de caractères (strings) pour matcher avec le CSV
    reversed_mapping = {str(value): key for key, value in original_mapping.items()}
    # --------------------------------------
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Fichier introuvable : {csv_path}")
        return

    if image_col not in df.columns or label_col not in df.columns:
        print(f"Alerte : Colonnes introuvables. Colonnes actuelles : {df.columns.tolist()}")
        return

    # 3. Créer la nouvelle colonne en utilisant le dictionnaire inversé
    df['text_label'] = df[label_col].astype(str).map(reversed_mapping)
    
    valeurs_manquantes = df['text_label'].isna().sum()
    if valeurs_manquantes > 0:
        print(f"⚠️ ATTENTION : {valeurs_manquantes} indices du CSV n'ont pas trouvé de correspondance dans le JSON !")

    # 4. Formater pour TrOCR
    df_trocr = df[[image_col, 'text_label']].rename(columns={image_col: 'file_name', 'text_label': 'text'})

    # 5. Sauvegarder
    df_trocr.to_csv(output_path, index=False)
    print(f"✅ Fichier sauvegardé : {output_path}")
    print(df_trocr.head(3))
    print("-" * 40)

# (Le reste du bloc if __name__ == "__main__": reste exactement le même qu'avant)
if __name__ == "__main__":
    # Définition des chemins basés sur ta capture d'écran
    base_dir = "Data/archive(1)/data/processed"
    mapping_file = os.path.join(base_dir, "mapping.json")
    
    # Fichiers d'entrée
    train_input = os.path.join(base_dir, "train_labels.csv")
    val_input = os.path.join(base_dir, "val_labels.csv")
    test_input = os.path.join(base_dir, "test_labels.csv")
    
    # Fichiers de sortie (ceux que tu utiliseras sur Colab)
    train_output = os.path.join(base_dir, "train_trocr.csv")
    val_output = os.path.join(base_dir, "val_trocr.csv")
    test_output = os.path.join(base_dir, "test_trocr.csv")
    
    # --- IMPORTANT ---
    # Remplace 'nom_image' et 'indice_medicament' par les VRAIS noms 
    # des colonnes qui sont à l'intérieur de tes fichiers CSV !
    COL_IMAGE = 'IMAGE' # Exemple : 'filename', 'image', 'id'
    COL_LABEL = 'MEDICINE_NAME'    # Exemple : 'class', 'label_id', 'drug_id'
    
    prepare_trocr_csv(train_input, mapping_file, train_output, image_col=COL_IMAGE, label_col=COL_LABEL)
    prepare_trocr_csv(val_input, mapping_file, val_output, image_col=COL_IMAGE, label_col=COL_LABEL)
    prepare_trocr_csv(test_input, mapping_file, test_output, image_col=COL_IMAGE, label_col=COL_LABEL)