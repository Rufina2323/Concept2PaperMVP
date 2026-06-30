import logging
import uuid

from celery import shared_task
from sqlalchemy.orm import Session as DBSession

from app.celery_app import celery
from app.database import SessionLocal
from app.models import ConceptPair, Draft
from app.models import Session as SessionModel
from app.services.llm_service import generate_draft
from app.services.ml_service import score_pairs_for_keyword

logger = logging.getLogger(__name__)


def _get_db() -> DBSession:
    return SessionLocal()


@celery.task(bind=True, max_retries=3)
def run_inference(self, session_id: str, keyword: str):
    db = _get_db()
    try:
        session = db.query(SessionModel).filter(SessionModel.id == uuid.UUID(session_id)).first()
        if session is None:
            logger.error("Session %s not found", session_id)
            return

        session.status = "processing"
        db.commit()

        results = score_pairs_for_keyword(keyword)

        for r in results:
            pair = ConceptPair(
                session_id=uuid.UUID(session_id),
                concept_a=r["concept_a"],
                concept_b=r["concept_b"],
                score=r["score"],
                rank=r["rank"],
            )
            db.add(pair)

        session.status = "done"
        db.commit()
        logger.info("Session %s done — %d pairs scored", session_id, len(results))

    except Exception as exc:
        logger.exception("Inference failed for session %s", session_id)
        db.rollback()
        session = db.query(SessionModel).filter(SessionModel.id == uuid.UUID(session_id)).first()
        if session:
            session.status = "error"
            session.error_msg = str(exc)
            db.commit()
        raise self.retry(exc=exc, countdown=5)
    finally:
        db.close()


@celery.task(bind=True, max_retries=3)
def run_draft(self, draft_id: str, concept_a: str, concept_b: str):
    db = _get_db()
    try:
        draft = db.query(Draft).filter(Draft.id == uuid.UUID(draft_id)).first()
        if draft is None:
            logger.error("Draft %s not found", draft_id)
            return

        draft.status = "processing"
        db.commit()

        content = generate_draft(concept_a, concept_b)

        draft.content = content
        draft.status = "done"
        db.commit()
        logger.info("Draft %s done", draft_id)

    except Exception as exc:
        logger.exception("Draft generation failed for %s", draft_id)
        db.rollback()
        draft = db.query(Draft).filter(Draft.id == uuid.UUID(draft_id)).first()
        if draft:
            draft.status = "error"
            draft.error_msg = str(exc)
            db.commit()
        raise self.retry(exc=exc, countdown=5)
    finally:
        db.close()
