from fastapi import APIRouter

from app.schemas import ConceptsOut
from app.services.ml_service import get_top_concepts

router = APIRouter(prefix="/concepts", tags=["concepts"])


@router.get("", response_model=ConceptsOut)
def list_concepts():
    """Return the top-20 concepts available for exploration."""
    return ConceptsOut(concepts=get_top_concepts())
