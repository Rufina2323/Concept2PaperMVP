from celery import Celery

from app.config import settings

celery = Celery(
    "mvp",
    broker=settings.celery_broker,
    backend=settings.celery_backend,
    include=["app.tasks"],
)
celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,  # one task at a time per worker (model is heavy)
)
