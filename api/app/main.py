from fastapi import FastAPI

from app.database import Base, engine
from app.routers import concepts, drafts, sessions

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Concept2Paper MVP", version="1.0.0")

app.include_router(concepts.router, prefix="/api/v1")
app.include_router(sessions.router, prefix="/api/v1")
app.include_router(drafts.router, prefix="/api/v1")


@app.get("/health")
def health():
    return {"status": "ok"}
