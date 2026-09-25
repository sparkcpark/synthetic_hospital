"""Application configuration via environment variables."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # PostgreSQL
    database_url: str = "postgresql+asyncpg://epic_sim:dev_password@localhost:5432/epic_sim"
    database_url_sync: str = "postgresql+psycopg://epic_sim:dev_password@localhost:5432/epic_sim"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # JWT
    jwt_secret: str = "dev-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 720  # 12 hours — long enough for full eval runs

    # App
    debug: bool = False
    api_prefix: str = ""
    fhir_base_url: str = "http://localhost:8000/fhir"

    # Session management
    session_ttl_seconds: int = 7200
    max_api_calls_per_session: int = 50
    redis_session_prefix: str = "epic_sess:"
    redis_required: bool = False

    # Scoring (reward) endpoint: shared secret sent as X-Scorer-Token. Held by the
    # trainer, never by the policy under evaluation. Empty string disables /score.
    scorer_token: str = "dev-scorer-token-change-in-production"
    # Reset-and-step environment: default action budget per episode (the paper used 40).
    env_default_budget: int = 40
    # Start one episode at boot (Harbor tasks): the agent reads it from GET /env/current.
    autostart_gt_id: int | None = None
    autostart_budget: int | None = None

    # SapBERT search
    sapbert_section_index_path: str = "data/ontology/section_sapbert_embeddings.npz"

    # Source data (for migration)
    sqlite_source: Path = Path(__file__).resolve().parent.parent.parent / "benchmark_v1.3.db"   # released benchmark DB at the repo root
    ontology_dir: Path = Path(__file__).resolve().parent.parent.parent / "data" / "ontology"

    model_config = {"env_prefix": "EPIC_SIM_", "env_file": ".env"}


settings = Settings()
