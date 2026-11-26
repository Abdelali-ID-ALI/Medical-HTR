"""Rebuild the local retrieval index from an authorized DrugBank XML export."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from config import COLLECTION, EMBEDDING_MODEL, Settings

NS = {"db": "http://www.drugbank.ca"}

def parse_drugbank(path, limit=None):
    # Only direct children of the root are drug records; nested drug references are not records.
    stack = []
    count = 0
    for event, element in ET.iterparse(path, events=("start", "end")):
        if event == "start":
            stack.append(element)
            continue
        if len(stack) == 2:
            if element.tag == "{http://www.drugbank.ca}drug":
                identifier = next((node.text for node in element.findall("db:drugbank-id", NS)
                                   if node.get("primary") == "true"), None)
                name = (element.findtext("db:name", default="", namespaces=NS) or "").strip()
                if identifier and name:
                    aliases = [name]
                    for xpath in ("db:synonyms/db:synonym", "db:international-brands/db:international-brand/db:name",
                                  "db:products/db:product/db:name"):
                        aliases.extend(n.text.strip() for n in element.findall(xpath, NS) if n.text and n.text.strip())
                    description = element.findtext("db:description", default="", namespaces=NS) or ""
                    yield {"drugbank_id": identifier, "name": name,
                           "aliases": sorted(set(aliases)), "description": description}
                    count += 1
            stack[0].remove(element)
            element.clear()
        stack.pop()
        if limit is not None and count >= limit:
            break

def build_index(xml_path, db_path, limit=None):
    path = Path(db_path)
    # Build in a new directory to prevent mixing incompatible embeddings.
    if path.exists() and any(path.iterdir()):
        raise ValueError("Index directory is not empty. Choose a new QDRANT_PATH.")
    path.mkdir(parents=True, exist_ok=True)
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    client = QdrantClient(path=str(path))
    count = 0
    try:
        dimension = encoder.get_sentence_embedding_dimension()
        client.create_collection(COLLECTION, vectors_config=models.VectorParams(
            size=dimension, distance=models.Distance.COSINE))
        batch = []
        def insert(records):
            nonlocal count
            # Index identity-focused text, not long interaction descriptions.
            vectors = encoder.encode(
                [r["name"] + " | " + " | ".join(r["aliases"]) for r in records],
                normalize_embeddings=True, show_progress_bar=False)
            points = [models.PointStruct(id=count+i, vector=v.tolist(), payload=r)
                      for i, (v, r) in enumerate(zip(vectors, records))]
            client.upsert(COLLECTION, points=points, wait=True)
            count += len(points)
        for record in parse_drugbank(xml_path, limit):
            batch.append(record)
            if len(batch) == 64:
                insert(batch)
                batch = []
        if batch:
            insert(batch)
        if count == 0:
            raise ValueError("No DrugBank records found.")
        (path / "medical_htr_manifest.json").write_text(json.dumps({
            "model": EMBEDDING_MODEL, "collection": COLLECTION, "count": count,
            "limited": limit is not None, "representation": "name_and_aliases_v1",
        }, indent=2), encoding="utf-8")
    finally:
        client.close()
    return count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml_path")
    parser.add_argument("--db-path", default=Settings().qdrant_path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    print(f"Indexed {build_index(args.xml_path, args.db_path, args.limit)} records.")
