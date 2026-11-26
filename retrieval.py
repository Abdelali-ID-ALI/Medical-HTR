"""Candidate retrieval, not clinical verification."""
import unicodedata
from rapidfuzz import fuzz
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from config import COLLECTION, EMBEDDING_MODEL, QDRANT_PATH
from schemas import Candidate

def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())

class DrugRetriever:
    def __init__(self):
        self.client = QdrantClient(path=QDRANT_PATH)
        try:
            self.encoder = SentenceTransformer(EMBEDDING_MODEL)
            self.records = {}
            offset = None
            while True:
                points, offset = self.client.scroll(COLLECTION, limit=256, offset=offset, with_payload=True)
                for point in points:
                    record = point.payload
                    if record.get("embedding_model") != EMBEDDING_MODEL:
                        raise ValueError("Index embedding model mismatch; rebuild index.")
                    self.records[record["drugbank_id"]] = record
                if offset is None:
                    break
            if not self.records:
                raise ValueError("DrugBank index is empty")
        except Exception:
            self.client.close()
            raise

    def close(self):
        self.client.close()

    def search(self, mention, k=5):
        query = normalize(mention)
        if not query:
            return []
        scores = {key: max(fuzz.ratio(query, normalize(name)) for name in [r["name"], *r["aliases"]])
                  for key, r in self.records.items()}
        exact = [key for key, score in scores.items() if score == 100]
        if exact:
            return [Candidate(drugbank_id=key, name=self.records[key]["name"],
                     lexical_score=100, ranking_score=1, exact_match=True) for key in sorted(exact)[:k]]
        lexical = sorted(scores, key=lambda key: (-scores[key], key))[:20]
        vector = self.encoder.encode(mention, normalize_embeddings=True).tolist()
        points = self.client.query_points(COLLECTION, query=vector, limit=20, with_payload=True).points
        vector_scores = {p.payload["drugbank_id"]: float(p.score) for p in points}
        ranks = {}
        for ranking in (lexical, list(vector_scores)):
            for rank, key in enumerate(ranking, 1):
                ranks[key] = ranks.get(key, 0) + 1 / (60 + rank)
        # Conservative illustrative lexical gate; not calibrated.
        eligible = [key for key in ranks if scores[key] >= 75]
        ordered = sorted(eligible, key=lambda key: (-ranks[key], -scores[key], key))[:k]
        return [Candidate(drugbank_id=key, name=self.records[key]["name"],
                lexical_score=scores[key], vector_score=vector_scores.get(key),
                ranking_score=ranks[key], exact_match=False) for key in ordered]
