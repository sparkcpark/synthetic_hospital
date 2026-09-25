"""SapBERT-based semantic embedder for SNOMED CT concept matching.

Embeds biomedical terms using SapBERT (fine-tuned on UMLS synonymy pairs)
and matches against pre-computed SNOMED CT concept embeddings via cosine similarity.

Usage:
    # Build index (one-time, ~2-3 hrs)
    embedder = SapBERTEmbedder()
    embedder.build_snomed_index(snomed_dict, save_path)

    # Load and query
    embedder = SapBERTEmbedder()
    embedder.load_index(index_path)
    results = embedder.find_closest(["Positive Babinski sign"], top_k=3, min_score=0.70)
"""

import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

from etl.utils.logging import get_logger

log = get_logger("etl.ontology.sapbert")

DEFAULT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
DEFAULT_INDEX_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "ontology" / "snomed_sapbert_embeddings.npz"
)


class SapBERTEmbedder:
    """SapBERT-based semantic embedder for SNOMED CT concept matching."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        log.info(f"Loading SapBERT model '{model_name}' on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        log.info("SapBERT model loaded")

        # Index state (populated by load_index or build_snomed_index)
        self._index_embeddings: np.ndarray | None = None  # (N, 768)
        self._index_concept_ids: list[str] | None = None

    def embed_batch(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        """Embed a batch of texts -> (N, 768) numpy array.

        Uses [CLS] token output as the sentence embedding (SapBERT convention).
        All inputs are lowercased because SapBERT uses PubMedBERT-uncased vocabulary
        but the HuggingFace tokenizer has do_lower_case=False.
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = [t.lower() for t in texts[i : i + batch_size]]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded)
                # CLS token embedding
                cls_emb = outputs.last_hidden_state[:, 0, :]
                # Normalize to unit vectors for cosine similarity
                cls_emb = torch.nn.functional.normalize(cls_emb, p=2, dim=1)
                all_embeddings.append(cls_emb.cpu().numpy())

            if (i // batch_size) % 50 == 0 and i > 0:
                log.info(f"  Embedded {i + len(batch)}/{len(texts)} texts")

        return np.vstack(all_embeddings).astype(np.float32)

    def build_snomed_index(self, snomed_dict, save_path: Path = DEFAULT_INDEX_PATH,
                           batch_size: int = 256) -> None:
        """Pre-embed all SNOMED CT descriptions and save to disk.

        For each active concept with a preferred term, embeds the preferred term.
        Saves concept_ids array + embeddings matrix as npz.
        """
        # Collect active concepts with preferred terms
        concept_ids = []
        terms = []
        for cid, concept in snomed_dict.concepts.items():
            if concept.is_active and concept.preferred_term:
                concept_ids.append(cid)
                terms.append(concept.preferred_term)

        log.info(f"Building SapBERT index for {len(terms)} SNOMED concepts...")

        embeddings = self.embed_batch(terms, batch_size=batch_size)
        log.info(f"Embeddings shape: {embeddings.shape}")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_path,
            concept_ids=np.array(concept_ids, dtype=object),
            embeddings=embeddings,
        )
        log.info(f"Saved SNOMED SapBERT index to {save_path} "
                 f"({save_path.stat().st_size / 1e9:.2f} GB)")

        self._index_concept_ids = concept_ids
        self._index_embeddings = embeddings

    def load_index(self, index_path: Path = DEFAULT_INDEX_PATH) -> None:
        """Load pre-computed SNOMED embeddings from disk."""
        log.info(f"Loading SapBERT index from {index_path}")
        data = np.load(index_path, allow_pickle=True)
        self._index_concept_ids = list(data["concept_ids"])
        self._index_embeddings = data["embeddings"].astype(np.float32)
        log.info(f"Loaded {len(self._index_concept_ids)} concept embeddings, "
                 f"shape {self._index_embeddings.shape}")

    def find_closest(self, query_texts: list[str], top_k: int = 1,
                     min_score: float = 0.70,
                     batch_size: int = 256) -> list[list[tuple[str, float]]]:
        """Find closest SNOMED concepts for each query text.

        Returns list of [(concept_id, cosine_score), ...] per query.
        Empty list if no match above min_score.
        """
        if self._index_embeddings is None or self._index_concept_ids is None:
            raise RuntimeError("No index loaded. Call load_index() or build_snomed_index() first.")

        # Embed queries
        query_emb = self.embed_batch(query_texts, batch_size=batch_size)

        # Cosine similarity via matrix multiply (embeddings are already L2-normalized)
        # query_emb: (Q, 768), index: (N, 768) -> sim: (Q, N)
        sim = query_emb @ self._index_embeddings.T

        results = []
        for i in range(len(query_texts)):
            row = sim[i]
            # Get top-k indices
            if top_k < len(row):
                top_indices = np.argpartition(row, -top_k)[-top_k:]
                top_indices = top_indices[np.argsort(row[top_indices])[::-1]]
            else:
                top_indices = np.argsort(row)[::-1][:top_k]

            matches = []
            for idx in top_indices:
                score = float(row[idx])
                if score >= min_score:
                    matches.append((self._index_concept_ids[idx], score))
            results.append(matches)

        return results
