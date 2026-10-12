"""Module 6 — retrieval metrics against REAL gold evidence.

Deliberately separate from `src/evaluation/retrieval_metrics.py`. That file
operationalises retrieval as a same-class nearest-neighbour proxy and says so
in its docstring: "this project's data doesn't have an explicit claim/evidence
retrieval structure". Module 5 has since built exactly that structure, so the
proxy is no longer the honest measurement. The old file is left untouched
(Module 4 results depend on it); this one is what Module 6 reports.

Gold relevance is binary and comes from Module 5's `claim.evidence` list.

Claims with NO gold evidence -- notably FEVER's NOT ENOUGH INFO claims, 427
of them in the frozen corpus -- are excluded from every recall denominator.
Including them would silently depress Recall@k with a term no retriever can
ever satisfy. They are counted and reported instead of being dropped quietly.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence


def recall_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    """Fraction of this claim's gold evidence found in the top `k`.

    Per-claim recall, not hit-rate: a claim with 3 gold items of which 2 are
    retrieved scores 2/3. Macro-averaging these gives equal weight to every
    claim regardless of how much evidence it happens to cite.
    """
    gold_set = set(gold)
    if not gold_set:
        raise ValueError("recall_at_k called on a claim with no gold evidence; "
                         "such claims must be excluded before averaging")
    hits = sum(1 for e in retrieved[:k] if e in gold_set)
    return hits / len(gold_set)


def reciprocal_rank(retrieved: Sequence[str], gold: Iterable[str]) -> float:
    """1 / rank of the FIRST gold item; 0.0 if none retrieved. Ranks are 1-based."""
    gold_set = set(gold)
    for pos, e in enumerate(retrieved, start=1):
        if e in gold_set:
            return 1.0 / pos
    return 0.0


def dcg(relevances: Sequence[float]) -> float:
    """Discounted cumulative gain, log2(rank+1) discount, 1-based ranks."""
    return sum(rel / math.log2(i + 1) for i, rel in enumerate(relevances, start=1))


def ndcg_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    """nDCG@k with binary relevance.

    The ideal ranking places min(|gold|, k) relevant items first, so a claim
    with more gold items than `k` is not penalised for the impossible.
    """
    gold_set = set(gold)
    if not gold_set:
        raise ValueError("ndcg_at_k called on a claim with no gold evidence")
    gains = [1.0 if e in gold_set else 0.0 for e in retrieved[:k]]
    ideal = [1.0] * min(len(gold_set), k)
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def evaluate_retrieval(
    results: dict[str, Sequence[str]],
    gold: dict[str, Sequence[str]],
    k_values: Sequence[int] = (1, 5, 10, 50),
    ndcg_k: int = 10,
    languages: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Macro-averaged retrieval metrics over claims that have gold evidence.

    `results` maps claim_id -> ranked evidence ids (best first).
    `gold`    maps claim_id -> gold evidence ids.

    Every claim in `results` is accounted for in exactly one of:
    scored, excluded_no_gold, or missing_from_gold.
    """
    scored: list[str] = []
    no_gold: list[str] = []
    missing_gold_entry: list[str] = []

    for cid in results:
        if cid not in gold:
            missing_gold_entry.append(cid)
        elif not gold[cid]:
            no_gold.append(cid)
        else:
            scored.append(cid)

    per_claim: dict[str, dict[str, float]] = {}
    for cid in scored:
        r, g = results[cid], gold[cid]
        m: dict[str, float] = {f"recall@{k}": recall_at_k(r, g, k) for k in k_values}
        m["mrr"] = reciprocal_rank(r, g)
        m[f"ndcg@{ndcg_k}"] = ndcg_at_k(r, g, ndcg_k)
        per_claim[cid] = m

    def macro(metric: str, subset: Sequence[str]) -> float:
        vals = [per_claim[c][metric] for c in subset if c in per_claim]
        return sum(vals) / len(vals) if vals else 0.0

    names = [f"recall@{k}" for k in k_values] + ["mrr", f"ndcg@{ndcg_k}"]
    out: dict[str, Any] = {
        "overall": {m: macro(m, scored) for m in names},
        "accounting": {
            "n_queries": len(results),
            "n_scored": len(scored),
            "n_excluded_no_gold_evidence": len(no_gold),
            "n_missing_from_gold_map": len(missing_gold_entry),
            "balances": len(results) == len(scored) + len(no_gold) + len(missing_gold_entry),
            "note": ("Claims with no gold evidence (e.g. FEVER NOT ENOUGH INFO) are "
                     "excluded from recall denominators and counted here. They are "
                     "not silently dropped."),
        },
        "k_values": list(k_values),
        "ndcg_k": ndcg_k,
    }

    if languages:
        by_lang: dict[str, dict[str, float]] = {}
        counts: dict[str, int] = {}
        for lang in sorted(set(languages.values())):
            subset = [c for c in scored if languages.get(c) == lang]
            counts[lang] = len(subset)
            if subset:
                by_lang[lang] = {m: macro(m, subset) for m in names}
        out["per_language"] = by_lang
        out["per_language_counts"] = counts

    out["per_claim"] = per_claim
    return out
