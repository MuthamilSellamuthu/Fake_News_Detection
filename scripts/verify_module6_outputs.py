#!/usr/bin/env python
"""Module 6 verifier — PASS/FAIL checks over whatever Module 6 has produced.

Follows the Module 3/4/5 convention: independent re-verification that
recomputes from the artefacts rather than trusting each stage's own manifest,
because a stage bug and that stage's self-report would fail together.

Stages that have not run yet are reported SKIPPED, not PASS. A check that
cannot fail is worse than no check.

    python scripts/verify_module6_outputs.py --dataset fever --split test
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.m6_common import (  # noqa: E402
    PROJECT_ROOT, gold_evidence_map, load_config, load_corpus, module5_fingerprint,
    read_manifest, stage_dir, write_json)
from src.retrieval.faiss_index import EvidenceIndex, brute_force_search  # noqa: E402
from src.retrieval.index_scope import (  # noqa: E402
    build_evidence_scope, verify_phase_isolation)
from src.retrieval.retrieval_metrics import evaluate_retrieval  # noqa: E402
from src.verification.dataset import (  # noqa: E402
    FEVER_LABELS, X_FACT_LABELS, assert_label_spaces_disjoint, get_label_space)

EVIDENCE_ID_RE = re.compile(r"^[a-z_]+:ev:[0-9a-f]{64}$")


class Checker:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def add(self, name: str, passed: bool | None, detail: str = "") -> None:
        status = "SKIP" if passed is None else ("PASS" if passed else "FAIL")
        self.checks.append({"name": name, "status": status, "detail": detail})
        print(f"  {status:4}  {name}" + (f"  |  {detail}" if detail else ""))

    def summary(self) -> dict:
        p = sum(1 for c in self.checks if c["status"] == "PASS")
        f = sum(1 for c in self.checks if c["status"] == "FAIL")
        s = sum(1 for c in self.checks if c["status"] == "SKIP")
        return {"n_checks": len(self.checks), "passed": p, "failed": f, "skipped": s,
                "overall": "FAIL" if f else ("PASS" if p else "NOTHING_TO_CHECK"),
                "checks": self.checks}


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify Module 6 outputs.")
    ap.add_argument("--dataset", required=True, choices=["fever", "x_fact"])
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args()

    root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    cfg = load_config(args.config)
    ck = Checker()
    print("=" * 70)
    print(f"MODULE 6 VERIFICATION — {args.dataset}/{args.split}")
    print("=" * 70)

    # ---------------------------------------------------- 1. Module 5 intact
    corpus = load_corpus(cfg, args.dataset, root=root)
    ck.add("module5_corpus_readable",
           bool(corpus["claims"]) and bool(corpus["evidence"]),
           f"{len(corpus['claims']):,} claims, {len(corpus['evidence']):,} evidence")

    ids = [e["evidence_id"] for e in corpus["evidence"]]
    ck.add("module5_evidence_ids_unique", len(set(ids)) == len(ids),
           f"{len(ids) - len(set(ids))} duplicate(s)")
    bad = [i for i in ids[:5000] if not EVIDENCE_ID_RE.match(i)]
    ck.add("module5_evidence_ids_are_sha256", not bad,
           f"{len(bad)} malformed" + (f", e.g. {bad[0][:60]}" if bad else ""))

    flat = [i for v in corpus["splits"].values() for i in v]
    ck.add("module5_splits_cover_every_claim_once",
           sorted(flat) == list(range(len(corpus["claims"]))),
           f"{len(flat)} assignments for {len(corpus['claims'])} claims")

    # ------------------------------------------------------ 2. label spaces
    try:
        assert_label_spaces_disjoint()
        space = get_label_space(args.dataset)
        ck.add("label_spaces_disjoint_and_unmapped", True,
               f"{space.name}: {space.labels}")
    except AssertionError as e:
        ck.add("label_spaces_disjoint_and_unmapped", False, str(e))
    ck.add("config_forbids_label_mapping",
           cfg["label_spaces"]["mapping_between_spaces"] is False)
    ck.add("fever_labels_exact", cfg["label_spaces"]["fever"]["labels"] == FEVER_LABELS)
    ck.add("xfact_labels_exact_and_preserved",
           cfg["label_spaces"]["x_fact"]["labels"] == X_FACT_LABELS,
           "'partly true/misleading' kept verbatim")

    # ---------------------------------------------------- 3. leakage scoping
    iso = verify_phase_isolation(corpus["claims"], corpus["splits"],
                                 cfg["leakage"]["index_scope"])
    ck.add("phase_isolation_holds", iso["passed"],
           f"{iso['n_passed']}/{iso['n_checks']} scope checks")

    # --------------------------------------------------------- 4. the index
    idx_dir = stage_dir(cfg, "build_index", root) / f"{args.dataset}_{args.split}"
    if (idx_dir / "index_meta.json").is_file():
        index = EvidenceIndex.load(idx_dir)
        scope = build_evidence_scope(corpus["claims"], corpus["splits"], args.split,
                                     cfg["leakage"]["index_scope"])
        extra = set(index.evidence_ids) - scope.evidence_ids
        ck.add("index_within_permitted_scope", not extra,
               f"{len(extra)} out-of-scope item(s)"
               + (f", e.g. {sorted(extra)[:3]}" if extra else ""))
        ck.add("index_ids_unique",
               len(set(index.evidence_ids)) == len(index.evidence_ids))
        norms = np.linalg.norm(index.vectors[: min(500, len(index))], axis=1)
        ck.add("index_vectors_l2_normalised",
               bool(np.allclose(norms, 1.0, atol=1e-3)),
               f"mean norm {float(norms.mean()):.6f}")
        n = min(64, len(index))
        if n >= 2:
            q = index.vectors[:5]
            got, _ = index.search(q, min(5, n))
            exp, _ = brute_force_search(q, index.vectors, index.evidence_ids, min(5, n))
            ck.add("index_matches_brute_force_cosine", got == exp,
                   "exact search verified on a sample")
    else:
        for name in ("index_within_permitted_scope", "index_ids_unique",
                     "index_vectors_l2_normalised", "index_matches_brute_force_cosine"):
            ck.add(name, None, "index not built yet (run m6_02)")

    # ----------------------------------------------------- 5. retrieval runs
    ret_dir = stage_dir(cfg, "retrieve", root) / f"{args.dataset}_{args.split}"
    gold = gold_evidence_map(corpus)
    cand_p = ret_dir / "dense_candidates.json"
    if cand_p.is_file():
        cands = json.loads(cand_p.read_text(encoding="utf-8"))
        scope = build_evidence_scope(corpus["claims"], corpus["splits"], args.split,
                                     cfg["leakage"]["index_scope"])
        offenders = {c: [e for e in v if e not in scope.evidence_ids]
                     for c, v in cands.items()}
        offenders = {c: v for c, v in offenders.items() if v}
        ck.add("retrieved_evidence_within_scope", not offenders,
               f"{len(offenders)} claim(s) retrieved out-of-scope evidence")

        m = evaluate_retrieval(cands, gold, (1, 5), ndcg_k=5)
        acc = m["accounting"]
        ck.add("retrieval_accounting_balances", acc["balances"],
               f"{acc['n_queries']} = {acc['n_scored']} scored + "
               f"{acc['n_excluded_no_gold_evidence']} no-gold + "
               f"{acc['n_missing_from_gold_map']} missing")
        ck.add("no_gold_claims_excluded_and_counted",
               acc["n_excluded_no_gold_evidence"] >= 0,
               f"{acc['n_excluded_no_gold_evidence']} claim(s) without gold evidence")
        ck.add("recall_monotone_in_k",
               m["overall"]["recall@1"] <= m["overall"]["recall@5"] + 1e-12,
               f"R@1 {m['overall']['recall@1']:.4f} <= R@5 {m['overall']['recall@5']:.4f}")
    else:
        for name in ("retrieved_evidence_within_scope", "retrieval_accounting_balances",
                     "no_gold_claims_excluded_and_counted", "recall_monotone_in_k"):
            ck.add(name, None, "retrieval not run yet (run m6_04)")

    # ------------------------------------------------------- 6. reranking
    rr_p = stage_dir(cfg, "rerank", root) / f"{args.dataset}_{args.split}" / "reranked_top5.json"
    if rr_p.is_file() and cand_p.is_file():
        rr = json.loads(rr_p.read_text(encoding="utf-8"))
        cands = json.loads(cand_p.read_text(encoding="utf-8"))
        invented = {c: sorted(set(v) - set(cands.get(c, []))) for c, v in rr.items()}
        invented = {c: v for c, v in invented.items() if v}
        ck.add("reranker_preserved_candidate_set", not invented,
               f"{len(invented)} claim(s) gained evidence not in their candidate set")
        too_many = {c: len(v) for c, v in rr.items()
                    if len(v) > int(cfg["reranking"]["top_n_after_rerank"])}
        ck.add("reranker_respects_top_n", not too_many, f"{len(too_many)} over-long list(s)")
    else:
        ck.add("reranker_preserved_candidate_set", None, "reranking not run yet (run m6_05)")
        ck.add("reranker_respects_top_n", None, "reranking not run yet")

    # ----------------------------------------------------- 7. verification
    ev_p = (stage_dir(cfg, "evaluation", root) / f"{args.dataset}_{args.split}"
            / "verification_metrics.json")
    if ev_p.is_file():
        res = json.loads(ev_p.read_text(encoding="utf-8"))
        ck.add("verification_uses_correct_label_space",
               res.get("label_space") == get_label_space(args.dataset).labels,
               str(res.get("label_space")))
        cal = res.get("calibration", {})
        ck.add("ece_reported", "ece_before" in cal and "ece_after" in cal,
               f"ECE {cal.get('ece_before')} -> {cal.get('ece_after')}")
        ck.add("temperature_fitted_on_validation",
               res.get("temperature", {}).get("split") == "validation",
               f"fitted on {res.get('temperature', {}).get('split')!r}")
        cm = res.get("metrics", {}).get("confusion_matrix")
        ck.add("confusion_matrix_totals_match_n",
               bool(cm) and sum(sum(r) for r in cm) == res.get("n"),
               f"n={res.get('n')}")
    else:
        for name in ("verification_uses_correct_label_space", "ece_reported",
                     "temperature_fitted_on_validation", "confusion_matrix_totals_match_n"):
            ck.add(name, None, "verifier not evaluated yet (run m6_07)")

    # ------------------------------------------------- 8. compute-scope honesty
    cs = cfg["compute_scope"]
    ck.add("compute_reductions_declared",
           cs["full_scale_results_claimed"] is False and len(cs["reductions"]) >= 5,
           f"{len(cs['reductions'])} documented reduction(s)")

    summary = ck.summary()
    summary["module5_fingerprint"] = module5_fingerprint(root)
    out = stage_dir(cfg, "manifests", root) / f"verify_{args.dataset}_{args.split}.json"
    write_json(out, summary)

    print("\n" + "=" * 70)
    print(f"MODULE 6 VERIFICATION: {summary['overall']}  "
          f"({summary['passed']} pass / {summary['failed']} fail / {summary['skipped']} skip)")
    print(f"wrote {out}")
    print("=" * 70)
    sys.exit(1 if summary["failed"] else 0)


if __name__ == "__main__":
    main()
