import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Session as SessionModel
from app.schemas import SessionCreate, SessionOut
from app.tasks import run_inference

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionOut, status_code=status.HTTP_202_ACCEPTED)
def create_session(body: SessionCreate, db: Session = Depends(get_db)):
    session = SessionModel(keyword=body.keyword, status="pending")
    db.add(session)
    db.commit()
    db.refresh(session)
    run_inference.delay(str(session.id), session.keyword)
    return session


@router.get("/{session_id}", response_model=SessionOut)
def get_session(session_id: uuid.UUID, db: Session = Depends(get_db)):
    session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session
