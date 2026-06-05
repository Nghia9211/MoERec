"""
semantic_scorer.py 

Dataset-specific query format to reduce distribution shift:
  Goodreads → "Books similar to: ..."
  Yelp      → "Places similar to: ..."
  Amazon    → "Products related to: ..."
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return b_norm @ a_norm


class SemanticScorer:
    def __init__(
        self,
        vector_store,
        embedding_function,
        id2name: Dict[int, str],
        name2id: Dict[str, int],
        dataset: str = 'amazon',
    ):
        self.vector_store       = vector_store
        self.embedding_function = embedding_function
        self.id2name            = id2name
        self.name2id            = name2id
        self.dataset            = dataset.lower()

        self._cache_hits:   int = 0
        self._cache_misses: int = 0

        self._docstore_cache: Dict[str, str] = {}
        if vector_store is not None:
            self._build_docstore_cache()

    # ─────────────────────────────────────────────────────────────────────
    # Cache
    # ─────────────────────────────────────────────────────────────────────

    def _build_docstore_cache(self):
        try:
            docstore = self.vector_store.docstore
            docs     = getattr(docstore, '_dict', {})
            for doc_id, doc in docs.items():
                name = (doc.metadata.get('item_name') or '').strip().lower()
                if name and doc.page_content:
                    self._docstore_cache[name] = doc.page_content
            print(f"[SemanticScorer] Docstore cache built: {len(self._docstore_cache)} entries "
                  f"(dataset={self.dataset})")
        except Exception as e:
            print(f"[SemanticScorer] Docstore cache init error: {e}")

    def _get_candidate_texts(self, candidate_names: List[str]) -> List[str]:
        import re
        texts = []
        for name in candidate_names:
            clean_name = re.sub(r'\s+\[.*?\]$', '', name).strip().lower()
            rich = self._docstore_cache.get(clean_name)
            if rich:
                self._cache_hits += 1
                texts.append(rich)
            else:
                self._cache_misses += 1
                texts.append(name)
        return texts

    def log_cache_stats(self):
        """Log cache hit/miss rate after a run."""
        total = self._cache_hits + self._cache_misses
        if total > 0:
            hit_rate = self._cache_hits / total * 100
            print(
                f"[SemanticScorer] Cache stats — "
                f"hits={self._cache_hits}, misses={self._cache_misses}, "
                f"total={total}, hit_rate={hit_rate:.1f}%"
            )
            if hit_rate < 50:
                print(
                    f"[SemanticScorer] ⚠️  Low hit rate ({hit_rate:.1f}%)! "
                    f"Check item_name in FAISS metadata vs id2name.txt"
                )
        else:
            print("[SemanticScorer] Cache stats: no queries run yet")

    # ─────────────────────────────────────────────────────────────────────
    # Query builder — dataset-specific format
    # ─────────────────────────────────────────────────────────────────────

    def _build_query(self, seq_str: str) -> str:
        """Build dataset-specific query to reduce distribution shift between
        user query and document text in the FAISS index."""
        ds = self.dataset
        if 'goodreads' in ds:
            return f"Books similar to: {seq_str}"
        elif 'yelp' in ds:
            return f"Places and restaurants similar to: {seq_str}"
        else:  # amazon (default)
            return f"Products related to: {seq_str}"

    # ─────────────────────────────────────────────────────────────────────
    # FAISS ANN retrieval
    # ─────────────────────────────────────────────────────────────────────

    def retrieve_top_candidates(self, seq_str: str, k: int = 20) -> Dict[str, float]:
        if self.vector_store is None or self.embedding_function is None:
            return {}

        query = self._build_query(seq_str)
        try:
            results: List[Tuple] = self.vector_store.similarity_search_with_relevance_scores(
                query, k=k
            )
            scores: Dict[str, float] = {}
            for doc, score in results:
                name = (doc.metadata.get('item_name') or '').strip()
                if name:
                    scores[name] = float(score)
            return scores
        except Exception as e:
            print(f"[SemanticScorer] FAISS search error: {e}")
            return {}

    # ─────────────────────────────────────────────────────────────────────
    # Direct embed scoring (for a given candidate pool)
    # ─────────────────────────────────────────────────────────────────────

    def _embed_and_score(self, query: str, candidate_names: List[str]) -> Dict[str, float]:
        if not candidate_names or self.embedding_function is None:
            return {n: 0.0 for n in candidate_names}

        candidate_texts = self._get_candidate_texts(candidate_names)
        try:
            query_vec = np.array(
                self.embedding_function.embed_query(query), dtype=np.float32
            )
            cand_vecs = np.array(
                self.embedding_function.embed_documents(candidate_texts), dtype=np.float32
            )
            sims = _cosine_sim(query_vec, cand_vecs)
            return {name: float(sims[i]) for i, name in enumerate(candidate_names)}
        except Exception as e:
            print(f"[SemanticScorer] Embed error: {e}")
            return {n: 0.0 for n in candidate_names}

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def score(self, seq_str: str, candidate_ids: List[int], data: dict = None) -> Dict[str, float]:
        if not candidate_ids:
            return {}
        candidate_names = [self.id2name.get(cid, f"item_{cid}") for cid in candidate_ids]
        query = self._build_query(seq_str)
        return self._embed_and_score(query, candidate_names)

    @classmethod
    def from_shared(cls, shared: dict) -> "SemanticScorer":
        return cls(
            vector_store       = shared.get('vector_store'),
            embedding_function = shared.get('embedding_function'),
            id2name            = shared['id2name'],
            name2id            = shared['name2id'],
            dataset            = shared.get('dataset', 'amazon'),
        )