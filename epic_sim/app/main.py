"""FastAPI application factory."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from epic_sim.app.config import settings
from epic_sim.app.models.base import engine

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Redis connection
    app.state.redis = None
    try:
        import redis.asyncio as aioredis

        r = aioredis.from_url(settings.redis_url, decode_responses=True)
        await r.ping()
        app.state.redis = r
        log.info("Redis connected: %s", settings.redis_url)
    except Exception as exc:
        if settings.redis_required:
            raise RuntimeError(f"Redis required but unavailable: {exc}") from exc
        log.warning("Redis unavailable — sessions disabled: %s", exc)

    # Autostart episode (Harbor tasks set EPIC_SIM_AUTOSTART_GT_ID)
    if settings.autostart_gt_id is not None:
        if app.state.redis is None:
            raise RuntimeError("EPIC_SIM_AUTOSTART_GT_ID requires Redis")
        from epic_sim.app.routers.agent import TOOL_DEFINITIONS
        from epic_sim.app.services import env_service

        view = await env_service.ensure_autostart(
            app.state.redis, [t.model_dump() for t in TOOL_DEFINITIONS],
            settings.autostart_gt_id, settings.autostart_budget,
        )
        log.info("Autostart episode %s for gt_id=%s (%s)", view["episode_id"], view["gt_id"], view["task"])

    # SapBERT section index (lazy, non-blocking)
    if settings.sapbert_section_index_path:
        try:
            from epic_sim.app.services.search_service import search_service

            search_service.load_index(settings.sapbert_section_index_path)
        except Exception as exc:
            log.warning("SapBERT index not loaded: %s", exc)

    yield

    if app.state.redis:
        await app.state.redis.aclose()
    await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Epic EHR Simulation",
        description="FHIR R4-compliant EHR simulation for clinical AI evaluation",
        version="0.2.0",
        lifespan=lifespan,
    )

    from epic_sim.app.routers import agent, auth, env, epic, fhir, score, sessions, terminology

    app.include_router(auth.router, prefix="/auth", tags=["auth"])
    app.include_router(fhir.router, prefix="/fhir", tags=["fhir"])
    app.include_router(terminology.router, prefix="/fhir", tags=["terminology"])
    app.include_router(epic.router, prefix="/epic", tags=["epic"])
    app.include_router(agent.router, prefix="/agent", tags=["agent"])
    app.include_router(sessions.router, prefix="/epic/sessions", tags=["sessions"])
    app.include_router(score.router, prefix="/score", tags=["score"])
    app.include_router(env.router, prefix="/env", tags=["env"])

    @app.get("/health")
    async def health():
        redis_ok = getattr(app.state, "redis", None) is not None
        return {"status": "ok", "redis": redis_ok}

    return app


app = create_app()
