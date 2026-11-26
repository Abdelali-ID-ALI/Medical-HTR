from pydantic import BaseModel, ConfigDict, Field

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

class Line(StrictModel):
    index: int
    bbox: tuple[int, int, int, int]
    quality_score: float
    text: str | None
    accepted: bool

class Mention(StrictModel):
    line_index: int
    name: str = Field(min_length=1)
    dosage: str | None = None
    frequency: str | None = None

class Extraction(StrictModel):
    medicines: list[Mention]

class Candidate(StrictModel):
    drugbank_id: str
    name: str
    matched_alias: str | None = None
    lexical_score: float | None = None
    vector_score: float | None = None
    exact: bool = False
    rank_score: float | None = None

class Match(StrictModel):
    mention: Mention
    candidates: list[Candidate]
    status: str

class Report(StrictModel):
    model_mode: str
    raw_text: str
    lines: list[Line]
    medicines: list[Match]
    warnings: list[str]
    requires_human_review: bool = True
