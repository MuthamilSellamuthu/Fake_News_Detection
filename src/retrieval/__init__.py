"""Module 6 — evidence retrieval.

Reads Module 5's frozen corpus; never writes to it.

    index_scope        claim-level scope -> permitted evidence ids (leakage gate)
    encoders           sharded, resumable LaBSE encoding
    faiss_index        exact flat inner-product index over normalised vectors
    bm25_baseline      Tamil-safe lexical baseline
    reranker           XLM-R cross-encoder, candidate-set preserving
    retrieval_metrics  Recall@k / MRR / nDCG against real gold evidence
"""

from src.retrieval.index_scope import (  # noqa: F401
    EvidenceScope,
    assert_index_within_scope,
    build_evidence_scope,
    claim_evidence_ids,
    verify_phase_isolation,
)
from src.retrieval.retrieval_metrics import (  # noqa: F401
    evaluate_retrieval,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "EvidenceScope", "build_evidence_scope", "claim_evidence_ids",
    "assert_index_within_scope", "verify_phase_isolation",
    "evaluate_retrieval", "recall_at_k", "reciprocal_rank", "ndcg_at_k",
]
