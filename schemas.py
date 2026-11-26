from pydantic import BaseModel, ConfigDict

class MedicineMention(BaseModel):
    model_config = ConfigDict(extra="forbid")
    line_index: int
    nom: str
    dosage: str | None = None
    frequence: str | None = None

class Extraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    medicaments: list[MedicineMention]

class Candidate(BaseModel):
    drugbank_id: str
    name: str
    lexical_score: float
    vector_score: float | None = None
    ranking_score: float
    exact_match: bool

class LinkedMention(BaseModel):
    mention: MedicineMention
    candidates: list[Candidate]
    status: str
    requires_human_review: bool = True

class LineResult(BaseModel):
    index: int
    bbox: list[int]
    quality_score: float
    text: str

class PrescriptionResult(BaseModel):
    model_id: str
    model_mode: str
    lines: list[LineResult]
    raw_text: str
    medicaments: list[LinkedMention]
    extraction_status: str
    requires_human_review: bool = True
    clinical_validation: bool = False
