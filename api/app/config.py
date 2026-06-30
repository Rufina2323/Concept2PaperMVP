from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg2://mvp:mvp@postgres:5432/mvp"
    redis_url: str = "redis://redis:6379/0"
    celery_broker: str = "redis://redis:6379/0"
    celery_backend: str = "redis://redis:6379/1"

    data_dir: str = "/data"
    graph_filename: str = "graph_2017_2024.pkl"
    embeddings_filename: str = "article_embeddings.pkl"
    model_filename: str = "mlp_all.pt"

    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"

    top_k_concepts: int = 20
    top_k_results: int = 5

    # Candidate sampling before MLP scoring
    n_candidates: int = 3000
    # strategy mix (must sum to 1.0)
    candidate_strategy_weights: dict = {
        "2hop": 0.55,
        "resource_alloc": 0.30,
        "pref_attach": 0.10,
        "random": 0.05,
    }

    # Last 5 years in the test graph for sparse matrix features
    sparse_years: list[int] = [2020, 2021, 2022, 2023, 2024]


settings = Settings()
