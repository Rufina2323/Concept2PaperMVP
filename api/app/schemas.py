from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class ConceptPairOut(BaseModel):
    id: uuid.UUID
    concept_a: str
    concept_b: str
    score: float
    rank: int

    model_config = {"from_attributes": True}


class SessionCreate(BaseModel):
    keyword: str


class SessionOut(BaseModel):
    id: uuid.UUID
    keyword: str
    status: str
    error_msg: Optional[str] = None
    created_at: datetime
    pairs: List[ConceptPairOut] = []

    model_config = {"from_attributes": True}


class DraftCreate(BaseModel):
    concept_a: str
    concept_b: str


class DraftOut(BaseModel):
    id: uuid.UUID
    concept_a: str
    concept_b: str
    content: Optional[str] = None
    status: str
    error_msg: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ConceptsOut(BaseModel):
    concepts: List[str]
