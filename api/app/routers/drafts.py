import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Draft
from app.schemas import DraftCreate, DraftOut
from app.tasks import run_draft

router = APIRouter(prefix="/drafts", tags=["drafts"])


@router.post("", response_model=DraftOut, status_code=status.HTTP_202_ACCEPTED)
def create_draft(body: DraftCreate, db: Session = Depends(get_db)):
    draft = Draft(concept_a=body.concept_a, concept_b=body.concept_b, status="pending")
    db.add(draft)
    db.commit()
    db.refresh(draft)
    run_draft.delay(str(draft.id), draft.concept_a, draft.concept_b)
    return draft


@router.get("/{draft_id}", response_model=DraftOut)
def get_draft(draft_id: uuid.UUID, db: Session = Depends(get_db)):
    draft = db.query(Draft).filter(Draft.id == draft_id).first()
    if draft is None:
        raise HTTPException(status_code=404, detail="Draft not found")
    return draft
