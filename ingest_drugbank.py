"""Build a fresh local index from an authorized DrugBank XML export."""
import argparse
import xml.etree.ElementTree as ET
import uuid
from pathlib import Path
from config import EMBEDDING_MODEL, QDRANT_PATH, COLLECTION

NS = {"db": "http://www.drugbank.ca"}

def parse_drugbank(path, limit=None):
    records = []
    depth = 0
    for event, element in ET.iterparse(path, events=("start", "end")):
        if event == "start":
            depth += 1
            continue
        if depth == 2 and element.tag == "{http://www.drugbank.ca}drug":
            drug_id = next((x.text for x in element.findall("db:drugbank-id", NS)
                            if x.get("primary") == "true" and x.text), None)
            name = element.findtext("db:name", namespaces=NS)
            aliases = [x.text.strip() for x in element.findall("db:synonyms/db:synonym", NS) if x.text and x.text.strip()]
            if drug_id and name:
                records.append({"drugbank_id": drug_id, "name": name.strip(), "aliases": sorted(set(aliases))})
            element.clear()
            if limit is not None and len(records) >= limit:
                break
        depth -= 1
    return records

def build_index(xml_path, db_path=QDRANT_PATH, limit=None):
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient, models
    records = parse_drugbank(xml_path, limit)
    if not records:
        raise ValueError("No DrugBank records found")
    if Path(db_path).exists():
        raise FileExistsError("Use a new index directory; existing data will not be overwritten.")
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    with QdrantClient(path=db_path) as client:
        client.create_collection(COLLECTION, vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(), distance=models.Distance.COSINE))
        for start in range(0, len(records), 64):
            batch = records[start:start+64]
            vectors = encoder.encode([r["name"] for r in batch], normalize_embeddings=True)
            client.upsert(COLLECTION, points=[models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, r["drugbank_id"])),
                vector=v.tolist(), payload={**r, "embedding_model": EMBEDDING_MODEL})
                for r, v in zip(batch, vectors)])
    return len(records)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("xml")
    parser.add_argument("--db-path", default=QDRANT_PATH)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(f"Indexed {build_index(args.xml, args.db_path, args.limit)} records")
