"""Module 6 — the six configured ablations.

Each ablation isolates ONE design decision so the paper can attribute any
gain to a specific cause rather than to the pipeline as a whole. Every
ablation is a pure specification (a dict) plus a deterministic runner, so
re-running with the same seed reproduces the same result.

    bm25_vs_dense            is the dense retriever earning its place?
    top_k_values             how much does candidate depth matter?
    claim_only_vs_evidence   does evidence help at all, or is the verifier
                             just learning claim-side priors?
    english_only_vs_multi    does multilingual evidence help Tamil?
    retrieval_only_vs_rerank does the expensive cross-encoder pay for itself?
    oracle_gold_evidence     upper bound with perfect retrieval

`claim_only` is the most important of these. If a claim-only verifier scores
close to the full pipeline, the system is not really verifying against
evidence, and reporting the full-pipeline number as "fact verification" would
overstate what was built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


@dataclass
class AblationSpec:
    name: str
    question: str
    variants: list[dict[str, Any]]
    metric: str = "macro_f1"
    stage: str = "verification"          # or "retrieval"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "question": self.question, "variants": self.variants,
                "metric": self.metric, "stage": self.stage, "notes": self.notes}


def build_ablation_specs(cfg: dict[str, Any]) -> list[AblationSpec]:
    ab = cfg.get("ablations", {})
    specs: list[AblationSpec] = []

    if ab.get("bm25_vs_dense", False):
        specs.append(AblationSpec(
            name="bm25_vs_dense",
            question="Does LaBSE dense retrieval beat a BM25 lexical baseline?",
            variants=[{"retriever": "bm25"}, {"retriever": "dense"}],
            metric="recall@5", stage="retrieval",
            notes="If BM25 matches dense, the dense retriever is not justified."))

    if ab.get("top_k_values"):
        specs.append(AblationSpec(
            name="top_k_sweep",
            question="How does candidate depth affect retrieval quality?",
            variants=[{"top_k": int(k)} for k in ab["top_k_values"]],
            metric="recall@k", stage="retrieval",
            notes="Recall is monotone non-decreasing in k; a flat curve means "
                  "extra candidates add nothing and top_k can be cut."))

    if ab.get("claim_only_vs_claim_evidence", False):
        specs.append(AblationSpec(
            name="claim_only_vs_claim_evidence",
            question="Does retrieved evidence improve verification at all?",
            variants=[{"claim_only": True}, {"claim_only": False}],
            notes="The critical control. A small gap means the verifier is "
                  "exploiting claim-side priors rather than verifying."))

    if ab.get("english_only_vs_multilingual_evidence", False):
        specs.append(AblationSpec(
            name="english_only_vs_multilingual_evidence",
            question="Does multilingual evidence help Tamil claims?",
            variants=[{"evidence_languages": ["en"]},
                      {"evidence_languages": ["en", "ta"]}],
            notes="Relevant to the low-resource claim of this project."))

    if ab.get("retrieval_only_vs_retrieval_rerank", False):
        specs.append(AblationSpec(
            name="retrieval_only_vs_retrieval_rerank",
            question="Does cross-encoder reranking justify its compute cost?",
            variants=[{"rerank": False}, {"rerank": True}],
            notes="Reranking dominates Module 6's GPU budget; if the gain is "
                  "small it should be reported as not cost-effective."))

    if ab.get("oracle_gold_evidence", False):
        specs.append(AblationSpec(
            name="oracle_gold_evidence",
            question="What is the ceiling if retrieval were perfect?",
            variants=[{"use_gold_evidence": False}, {"use_gold_evidence": True}],
            notes="Upper bound only. Module 5 config sets "
                  "gold_evidence_in_primary_path: false, so the oracle is never "
                  "the headline result."))

    return specs


def run_ablation(spec: AblationSpec, runner: Callable[[dict[str, Any]], dict[str, Any]],
                 seed: int = 42) -> dict[str, Any]:
    """Execute every variant through `runner` and summarise.

    `runner` maps a variant dict to a metrics dict. Injecting it keeps this
    testable and keeps the ablation framework independent of whether the
    variant is a retrieval or a verification experiment.
    """
    results = []
    for variant in spec.variants:
        payload = dict(variant)
        payload["seed"] = seed
        metrics = runner(payload)
        results.append({"variant": variant, "metrics": metrics})

    values = [(r["variant"], r["metrics"].get(spec.metric)) for r in results]
    scored = [(v, m) for v, m in values if isinstance(m, (int, float))]
    best = max(scored, key=lambda vm: vm[1]) if scored else (None, None)
    worst = min(scored, key=lambda vm: vm[1]) if scored else (None, None)
    delta = (best[1] - worst[1]) if scored and len(scored) > 1 else 0.0

    return {"name": spec.name, "question": spec.question, "metric": spec.metric,
            "stage": spec.stage, "notes": spec.notes, "seed": seed,
            "results": results, "best_variant": best[0], "best_value": best[1],
            "worst_variant": worst[0], "worst_value": worst[1],
            "delta": delta,
            "interpretation": ("A near-zero delta means this design choice does not "
                               "matter for this corpus and should be reported as such, "
                               "not omitted.")}


def run_all_ablations(cfg: dict[str, Any],
                      runners: dict[str, Callable[[dict[str, Any]], dict[str, Any]]],
                      seed: int = 42) -> dict[str, Any]:
    """Run every configured ablation that has a registered runner."""
    specs = build_ablation_specs(cfg)
    out: dict[str, Any] = {"seed": seed, "n_specs": len(specs), "ablations": {}, "skipped": []}
    for spec in specs:
        runner = runners.get(spec.name)
        if runner is None:
            out["skipped"].append({"name": spec.name,
                                   "reason": "no runner registered for this ablation"})
            continue
        out["ablations"][spec.name] = run_ablation(spec, runner, seed=seed)
    return out
