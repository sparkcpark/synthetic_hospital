"""Hybrid chart search: PostgreSQL tsvector + SapBERT semantic embeddings.

Combines keyword-based full-text search (tsvector/ts_rank) with SapBERT
semantic similarity, fused via Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from epic_sim.app.auth.rbac import get_allowed_sections
from epic_sim.app.schemas.epic import SectionEntry

log = logging.getLogger(__name__)

# RRF constant (standard value from Cormack et al. 2009)
RRF_K = 60


class HybridSearchService:
    """Hybrid tsvector + SapBERT search over encounter EHR sections."""

    def __init__(self) -> None:
        self._section_embeddings: np.ndarray | None = None  # (N, 768)
        self._section_ids: np.ndarray | None = None  # (N,) int array
        self._id_to_idx: dict[int, int] | None = None  # section_id → index
        self._embedder = None  # Lazy-loaded SapBERTEmbedder

    @property
    def is_loaded(self) -> bool:
        return self._section_embeddings is not None

    def load_index(self, index_path: str | Path) -> None:
        """Load pre-computed section embeddings from npz file."""
        path = Path(index_path)
        if not path.exists():
            log.warning("SapBERT section index not found at %s — semantic search disabled", path)
            return

        log.info("Loading SapBERT section index from %s", path)
        data = np.load(path, allow_pickle=False)
        self._section_ids = data["section_ids"].astype(np.int64)
        self._section_embeddings = data["embeddings"].astype(np.float32)
        self._id_to_idx = {int(sid): i for i, sid in enumerate(self._section_ids)}
        log.info(
            "Loaded %d section embeddings, shape %s",
            len(self._section_ids),
            self._section_embeddings.shape,
        )

    def _ensure_embedder(self):
        """Lazy-load SapBERT model for query embedding."""
        if self._embedder is None:
            from etl.ontology.sapbert_embedder import SapBERTEmbedder
            self._embedder = SapBERTEmbedder()
        return self._embedder

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string → (1, 768) array."""
        embedder = self._ensure_embedder()
        return embedder.embed_batch([query], batch_size=1)

    async def keyword_search(
        self,
        db: AsyncSession,
        patient_id: int,
        query: str,
        allowed_types: set[str] | None,
        limit: int = 50,
    ) -> list[tuple[int, float]]:
        """tsvector keyword search → list of (section_id, ts_rank)."""
        type_filter = ""
        params: dict = {"pid": patient_id, "query": query, "limit": limit}
        if allowed_types is not None:
            type_filter = "AND ees.section_type = ANY(:types)"
            params["types"] = list(allowed_types)

        sql = text(f"""
            SELECT ees.id, ts_rank(ees.search_vector, plainto_tsquery('english', :query)) AS rank
            FROM encounter_ehr_sections ees
            JOIN longitudinal_encounters le ON le.encounter_id = ees.encounter_id
            WHERE le.patient_id = :pid
              AND ees.search_vector @@ plainto_tsquery('english', :query)
              {type_filter}
            ORDER BY rank DESC
            LIMIT :limit
        """)
        result = await db.execute(sql, params)
        return [(row[0], float(row[1])) for row in result.fetchall()]

    def semantic_search(
        self,
        query: str,
        candidate_ids: set[int],
        limit: int = 50,
    ) -> list[tuple[int, float]]:
        """SapBERT semantic search → list of (section_id, cosine_sim)."""
        if not self.is_loaded or not candidate_ids:
            return []

        # Get indices for candidate section IDs
        indices = [self._id_to_idx[sid] for sid in candidate_ids if sid in self._id_to_idx]
        if not indices:
            return []

        idx_array = np.array(indices)
        candidate_emb = self._section_embeddings[idx_array]  # (C, 768)
        candidate_sids = self._section_ids[idx_array]  # (C,)

        # Embed query
        query_emb = self._embed_query(query)  # (1, 768)

        # Cosine similarity (embeddings are L2-normalized)
        scores = (query_emb @ candidate_emb.T).flatten()  # (C,)

        # Top-k
        if limit < len(scores):
            top_k_idx = np.argpartition(scores, -limit)[-limit:]
            top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
        else:
            top_k_idx = np.argsort(scores)[::-1]

        return [(int(candidate_sids[i]), float(scores[i])) for i in top_k_idx]

    async def hybrid_search(
        self,
        db: AsyncSession,
        patient_id: int,
        query: str,
        role: str,
        limit: int = 20,
    ) -> list[SectionEntry]:
        """Hybrid search combining tsvector + SapBERT via RRF.

        1. tsvector keyword search (ts_rank)
        2. SapBERT semantic search (cosine similarity) — if index loaded
        3. Reciprocal Rank Fusion to combine rankings
        4. RBAC section filtering
        5. Fetch full section text for top results
        """
        allowed_types = get_allowed_sections(role)

        # 1. Keyword search
        keyword_hits = await self.keyword_search(db, patient_id, query, allowed_types, limit=50)

        # 2. Semantic search (if index available)
        semantic_hits: list[tuple[int, float]] = []
        if self.is_loaded:
            # Get all section IDs for this patient (with role filtering)
            type_filter = ""
            params: dict = {"pid": patient_id}
            if allowed_types is not None:
                type_filter = "AND ees.section_type = ANY(:types)"
                params["types"] = list(allowed_types)

            sql = text(f"""
                SELECT ees.id FROM encounter_ehr_sections ees
                JOIN longitudinal_encounters le ON le.encounter_id = ees.encounter_id
                WHERE le.patient_id = :pid {type_filter}
            """)
            result = await db.execute(sql, params)
            candidate_ids = {row[0] for row in result.fetchall()}
            semantic_hits = self.semantic_search(query, candidate_ids, limit=50)

        # 3. RRF fusion
        rrf_scores: dict[int, float] = {}
        for rank, (sid, _score) in enumerate(keyword_hits, start=1):
            rrf_scores[sid] = rrf_scores.get(sid, 0.0) + 1.0 / (RRF_K + rank)
        for rank, (sid, _score) in enumerate(semantic_hits, start=1):
            rrf_scores[sid] = rrf_scores.get(sid, 0.0) + 1.0 / (RRF_K + rank)

        if not rrf_scores:
            return []

        # Sort by RRF score descending
        top_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:limit]

        # 4. Fetch full section data
        sql = text("""
            SELECT ees.id, ees.encounter_id, ees.section_type, ees.section_text, ees.section_order
            FROM encounter_ehr_sections ees
            WHERE ees.id = ANY(:ids)
        """)
        result = await db.execute(sql, {"ids": top_ids})
        rows = {row[0]: row for row in result.fetchall()}

        # Return in RRF order
        entries = []
        for sid in top_ids:
            if sid in rows:
                r = rows[sid]
                entries.append(SectionEntry(
                    section_id=r[0],
                    encounter_id=r[1],
                    section_type=r[2],
                    section_text=r[3],
                    section_order=r[4],
                ))
        return entries


# Module-level singleton
search_service = HybridSearchService()


async def build_section_index(
    db_url: str = "postgresql+asyncpg://epic_sim:dev_password@localhost:5432/epic_sim",
    save_path: str = "data/ontology/section_sapbert_embeddings.npz",
    batch_size: int = 256,
    max_length: int = 256,
) -> None:
    """Pre-compute SapBERT embeddings for all encounter_ehr_sections.

    Run once:
        python -c "
        from epic_sim.app.services.search_service import build_section_index
        import asyncio; asyncio.run(build_section_index())
        "
    """
    import torch
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from transformers import AutoModel, AutoTokenizer

    print("Loading SapBERT model...")
    model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    print(f"Connecting to {db_url.split('@')[1] if '@' in db_url else db_url}")
    engine = create_async_engine(db_url)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with session_factory() as db:
        result = await db.execute(
            text("SELECT id, section_text FROM encounter_ehr_sections ORDER BY id")
        )
        rows = result.fetchall()

    print(f"Loaded {len(rows)} sections")
    section_ids = np.array([r[0] for r in rows], dtype=np.int64)
    texts = [r[1].lower() for r in rows]  # SapBERT requires lowercase

    # Embed in batches
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)
            cls_emb = outputs.last_hidden_state[:, 0, :]
            cls_emb = torch.nn.functional.normalize(cls_emb, p=2, dim=1)
            all_embeddings.append(cls_emb.cpu().numpy())

        if (i // batch_size) % 20 == 0:
            print(f"  Embedded {i + len(batch)}/{len(texts)}")

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    print(f"Embeddings shape: {embeddings.shape}")

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, section_ids=section_ids, embeddings=embeddings)
    print(f"Saved to {out} ({out.stat().st_size / 1e6:.1f} MB)")

    await engine.dispose()
