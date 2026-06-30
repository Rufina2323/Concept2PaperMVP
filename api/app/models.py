import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.database import Base


class Session(Base):
    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    keyword = Column(String(512), nullable=False, index=True)
    status = Column(String(20), nullable=False, default="pending")
    error_msg = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    pairs = relationship("ConceptPair", back_populates="session", order_by="ConceptPair.rank")


class ConceptPair(Base):
    __tablename__ = "concept_pairs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)
    concept_a = Column(String(512), nullable=False)
    concept_b = Column(String(512), nullable=False)
    score = Column(Float, nullable=False)
    rank = Column(Integer, nullable=False)

    session = relationship("Session", back_populates="pairs")


class Draft(Base):
    __tablename__ = "drafts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    concept_a = Column(String(512), nullable=False)
    concept_b = Column(String(512), nullable=False)
    content = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default="pending")
    error_msg = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
