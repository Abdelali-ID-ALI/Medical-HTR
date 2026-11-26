"""Lexical-first candidate retrieval. No score is a clinical confidence."""
import json
import unicodedata
from pathlib import Path
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from config import COLLECTION, EMBEDDING_MODEL
from schemas import Candidate, Match, Mention

def normalize(text):
    text = unicodedata.normalize("NFKD", text.casefold())
    return " ".join("".join(c for c in text if not unicodedata.combining(c)).split())

class DrugRetriever:
    def __init__(self, db_path):
        path = Path(db_path)
        manifest = json.loads((path / "medical_htr_manifest.json").read_text(encoding="utf-8"))
        if (manifest["model"], manifest["collection"], manifest["representation"]) != (
            EMBEDDING_MODEL, COLLECTION, "name_and_aliases_v1"
        ):
            raise ValueError("Incompatible index; rebuild from XML.")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)
        self.client = QdrantClient(path=str(path))
        self.records, self.aliases = {}, []
        self.alias_ids = {}
        try:
            if self.client.count(COLLECTION, exact=True).count != manifest["count"]:
                raise ValueError("Incomplete index.")
            offset = None
            while True:
                points, offset = self.client.scroll(COLLECTION, offset=offset, limit=256,
                                                   with_payload=True, with_vectors=False)
                for point in points:
                    record = point.payload
                    identifier = record["drugbank_id"]
                    self.records[identifier] = record
                    for alias in record["aliases"]:
                        key = normalize(alias)
                        if key:
                            self.alias_ids.setdefault(key, []).append((identifier, alias))
                if offset is None:
                    break
            self.aliases = sorted(self.alias_ids)
        except Exception:
            self.client.close()
            raise

    def search(self, mention: Mention, top_k=5):
        query = normalize(mention.name)
        if not query:
            return Match(mention=mention, candidates=[], status="no_candidate")
        candidates = {}
        def get(identifier):
            if identifier not in candidates:
                candidates[identifier] = Candidate(drugbank_id=identifier, name=self.records[identifier]["name"])
            return candidates[identifier]

        exact = self.alias_ids.get(query, [])
        if exact:
            for identifier, alias in exact:
                candidate = get(identifier)
                candidate.exact, candidate.lexical_score, candidate.matched_alias = True, 100.0, alias
            # Even a unique string match is only a candidate, never a verified prescription.
            return Match(mention=mention, candidates=list(candidates.values())[:top_k],
                         status="exact_candidate" if len(candidates) == 1 else "ambiguous_exact")

        lexical = {}
        # Heuristic shortlist cutoff, not calibrated. Prefer whole-name edit similarity.
        for alias, score, _ in process.extract(query, self.aliases, scorer=fuzz.ratio,
                                               limit=40, score_cutoff=70):
            for identifier, original in self.alias_ids[alias]:
                if identifier not in lexical or score > lexical[identifier]:
                    lexical[identifier] = score
                    candidate = get(identifier)
                    candidate.lexical_score, candidate.matched_alias = score, original
        lexical_order = sorted(lexical, key=lambda i: (-lexical[i], i))[:20]
        if not lexical_order:
            # A semantically related drug is insufficient evidence of textual identity.
            return Match(mention=mention, candidates=[], status="no_candidate")
        vector = self.encoder.encode(query, normalize_embeddings=True).tolist()
        hits = self.client.query_points(COLLECTION, query=vector, limit=20, with_payload=True).points
        vector_order = []
        for hit in hits:
            identifier = hit.payload["drugbank_id"]
            if identifier in lexical_order and identifier not in vector_order:
                get(identifier).vector_score = float(hit.score)
                vector_order.append(identifier)
        # RRF with lexical priority; vector-only hits are not promoted.
        for rank, identifier in enumerate(lexical_order, 1):
            candidate = get(identifier)
            candidate.rank_score = 2 / (60 + rank)
            if identifier in vector_order:
                candidate.rank_score += 1 / (60 + vector_order.index(identifier) + 1)
        ranked = sorted((get(i) for i in lexical_order), key=lambda c: (-c.rank_score, c.drugbank_id))
        return Match(mention=mention, candidates=ranked[:top_k], status="review_candidates")

    def close(self):
        self.client.close()
