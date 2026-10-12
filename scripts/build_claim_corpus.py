#!/usr/bin/env python
"""Module 5 Stage 3 — build the unified claim/evidence corpus. COLAB.

Reads the downloaded X-FACT and FEVER JSONL files and writes the
`claims.jsonl` / `evidence.jsonl` / `claim_splits.json` artefacts the
config already declares. No retrieval, no embeddings, no training.

Official splits are preserved exactly. The group-aware 70/15/15
splitter is NOT applied: both corpora ship a published protocol, and
reshuffling it would make results incomparable with prior work. The
leakage gate still runs against those official splits — a failure there
is a property of the published dataset and is reported as such, never
repaired by silently re-splitting.

Reads:  datasets/module5_external/{x_fact,fever_gold_evidence}/*.jsonl
Writes: datasets/fact_verification/{claims,evidence}.jsonl,
        claim_splits.json, and outputs/module5/corpus_build_report.{json,md}

Never touches datasets/processed/, embeddings/, outputs/module3/,
outputs/module4/ or models/module4/.

Usage:
    python scripts/build_claim_corpus.py
    python scripts/build_claim_corpus.py --only x_fact
    python scripts/build_claim_corpus.py --skip-leakage-check   # large corpora
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fact_verification.corpus_builder import (  # noqa: E402
    build_corpus_from_files,
    splits_from_claims,
    write_corpus,
)
from src.fact_verification.leakage_gate import (  # noqa: E402
    enforce_group_safe_splits,
    group_claims,
    verify_no_leakage,
)
from src.utils.config_loader import load_yaml  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("module5.build_corpus")

FROZEN = ["datasets/processed", "embeddings", "outputs/module3", "outputs/module4", "models/module4"]

# (directory, dataset tag, {filename stem suffix -> official split name})
CORPUS_LAYOUT = {
    "x_fact": ("x_fact", "x_fact", {"train": "train", "dev": "dev", "test": "test"}),
    "fever": ("fever_gold_evidence", "fever", {"train": "train", "validation": "validation", "test": "test"}),
}


def frozen_signature(root: Path) -> dict[str, Any]:
    sig: dict[str, Any] = {}
    for rel in FROZEN:
        p = root / rel
        if not p.exists():
            sig[rel] = None
            continue
        files = [f for f in p.rglob("*") if f.is_file()]
        sig[rel] = {"n_files": len(files), "total_bytes": sum(f.stat().st_size for f in files)}
    return sig


def discover_files(corpus_dir: Path, split_map: dict[str, str]) -> list[tuple[Path, str]]:
    """Match each JSONL to its official split by filename suffix."""
    found: list[tuple[Path, str]] = []
    for fp in sorted(corpus_dir.glob("*.jsonl")):
        stem = fp.stem.lower()
        for suffix, official in split_map.items():
            if stem.endswith(f"_{suffix}"):
                found.append((fp, official))
                break
        else:
            logger.warning("Unrecognised split for %s — skipped (expected suffixes: %s)",
                           fp.name, sorted(split_map))
    return found


def main() -> None:
    p = argparse.ArgumentParser(description="Module 5 Stage 3: build the unified claim corpus.")
    p.add_argument("--config", default=None)
    p.add_argument("--data-dir", default="datasets/module5_external")
    p.add_argument("--only", choices=["x_fact", "fever"], default=None)
    p.add_argument("--skip-leakage-check", action="store_true",
                   help="Skip the cross-split near-duplicate scan (expensive on FEVER).")
    args = p.parse_args()

    root = PROJECT_ROOT
    cfg = load_yaml(Path(args.config) if args.config else root / "configs" / "module5_fact_verification.yaml")
    data_dir = root / args.data_dir
    out_reports = root / cfg["output"]["reports_dir"]
    out_corpus_root = root / cfg["output"]["corpus_dir"]
    out_reports.mkdir(parents=True, exist_ok=True)

    before = frozen_signature(root)
    lk = cfg["leakage"]

    report: dict[str, Any] = {
        "module": "Module 5 Stage 3 — claim/evidence corpus construction",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split_policy": "PRESERVE_OFFICIAL_WITH_GROUP_SAFETY_REPAIR",
        "split_policy_note": ("X-FACT and FEVER both ship official splits; these are preserved "
                              "verbatim. The group-aware 70/15/15 splitter is reserved for corpora "
                              "without a usable official split and was NOT applied. Where a "
                              "corpus's own split places near-duplicate claims on both sides, the "
                              "affected group is moved whole to TRAIN (deterministic, no RNG, no "
                              "deletions) so that one claim_group_id belongs to exactly one "
                              "split. Evaluation splits only ever shrink. metadata.official_split "
                              "still holds each corpus's published split; claim_splits.json is "
                              "the authoritative final assignment. Per-claim moves are listed "
                              "under group_safety_repair."),
        "corpora": {},
        "warnings": [],
    }

    targets = [args.only] if args.only else ["x_fact", "fever"]
    for key in targets:
        subdir, dataset, split_map = CORPUS_LAYOUT[key]
        corpus_dir = data_dir / subdir
        if not corpus_dir.exists():
            report["warnings"].append(f"{dataset}: {corpus_dir} not found — run the download script first")
            continue

        files = discover_files(corpus_dir, split_map)
        if not files:
            report["warnings"].append(f"{dataset}: no recognised split files in {corpus_dir}")
            continue

        logger.info("[%s] building from %d file(s)...", dataset, len(files))
        t0 = time.perf_counter()
        claims, evidence_pool, stats = build_corpus_from_files(files, dataset, cfg)
        build_seconds = round(time.perf_counter() - t0, 2)

        if not stats["row_accounting_balances"]:
            report["warnings"].append(
                f"{dataset}: ROW ACCOUNTING DOES NOT BALANCE — "
                f"{stats['n_input_rows']} in vs {stats['n_claims_built']} built + "
                f"{stats['n_dropped_no_claim_text']} + {stats['n_dropped_below_min_chars']} dropped")

        splits = splits_from_claims(claims)

        logger.info("[%s] grouping %d claims (threshold=%.2f, ngram=%d)...",
                    dataset, len(claims), lk["near_duplicate_threshold"], lk["near_duplicate_ngram"])
        t1 = time.perf_counter()
        grouping = group_claims(claims, near_duplicate=True,
                                ngram=lk["near_duplicate_ngram"],
                                threshold=lk["near_duplicate_threshold"])
        group_seconds = round(time.perf_counter() - t1, 2)

        # Official splits are preserved, but a corpus may itself place duplicate
        # claims on both sides of its own split boundary (FEVER does). Groups are
        # computed independently above, so nothing reconciled the two until here.
        # Whole groups move; no claim is deleted and no content changes.
        splits, repair = enforce_group_safe_splits(claims, splits, grouping.group_ids)
        if repair["n_groups_repaired"]:
            logger.info("[%s] group-safety repair: %d group(s) spanned splits, "
                        "%d claim(s) moved; counts %s -> %s", dataset,
                        repair["n_groups_repaired"], repair["n_claims_moved"],
                        repair["counts_before"], repair["counts_after"])
        else:
            logger.info("[%s] group-safety repair: no group spanned a split (no-op)", dataset)

        leakage = None
        if not args.skip_leakage_check:
            logger.info("[%s] running leakage gate on OFFICIAL splits...", dataset)
            t2 = time.perf_counter()
            leakage = verify_no_leakage(
                claims, splits, grouping.group_ids,
                near_duplicate_threshold=lk["near_duplicate_threshold"],
                ngram=lk["near_duplicate_ngram"])
            leakage["seconds"] = round(time.perf_counter() - t2, 2)
            leakage["interpretation"] = (
                "Measured on the corpus's OWN official splits. A failure here describes the "
                "published dataset, not our splitting. Report it; do not silently re-split.")
            if not leakage["passed"]:
                report["warnings"].append(
                    f"{dataset}: leakage gate FAILED on official splits — see corpus_build_report.json")

        out_dir = out_corpus_root / dataset
        paths = write_corpus(claims, evidence_pool, splits, grouping.group_ids,
                             out_dir, cfg["output"]["claims_filename"],
                             splits_filename=cfg["output"]["splits_filename"])

        report["corpora"][dataset] = {
            "source_files": [f"{fp.name} -> {sp}" for fp, sp in files],
            "build_seconds": build_seconds,
            "grouping_seconds": group_seconds,
            "statistics": stats,
            "splits": {k: len(v) for k, v in splits.items()},
            "grouping": grouping.summary,
            "group_safety_repair": repair,
            "leakage": leakage,
            "output_files": paths,
        }
        logger.info("[%s] %d claims, %d unique evidence items, %d groups -> %s",
                    dataset, len(claims), len(evidence_pool), grouping.n_groups, out_dir)

    after = frozen_signature(root)
    changed = {k: {"before": before.get(k), "after": after.get(k)}
               for k in after if before.get(k) != after.get(k)}
    report["modules_1_4_integrity"] = {"unchanged": not changed, "changed": changed}
    if changed:
        report["warnings"].append(f"FROZEN MODULES CHANGED: {sorted(changed)}")

    (out_reports / "corpus_build_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    _write_md(report, out_reports / "corpus_build_report.md")

    print("\nMODULE 5 STAGE 3 — CLAIM CORPUS BUILD")
    print("-------------------------------------")
    for name, r in report["corpora"].items():
        s = r["statistics"]
        print(f"\n{name}:")
        print(f"  input rows        : {s['n_input_rows']}")
        print(f"  claims built      : {s['n_claims_built']}  "
              f"(dropped: {s['n_dropped_no_claim_text']} no-text, "
              f"{s['n_dropped_below_min_chars']} below-min)")
        print(f"  row accounting    : {'BALANCES' if s['row_accounting_balances'] else 'MISMATCH'}")
        print(f"  official splits   : {r['splits']}")
        print(f"  languages         : {s['by_language']}")
        print(f"  labels            : {s['by_label']}")
        print(f"  evidence items    : {s['n_unique_evidence_items']} unique "
              f"({s['n_evidence_mentions']} mentions)")
        print(f"  claims w/o evidence: {s['n_claims_with_no_evidence']}")
        print(f"  groups            : {r['grouping']['n_groups']} "
              f"(largest {r['grouping']['largest_group_size']}, "
              f"{r['grouping']['near_duplicate_pairs']} near-dup pairs)")
        if r["leakage"]:
            print(f"  leakage gate      : {'PASS' if r['leakage']['passed'] else 'FAIL'} "
                  f"({r['leakage']['n_passed']}/{r['leakage']['n_checks']}, {r['leakage']['seconds']}s)")
    print(f"\nModules 1-4 unchanged: {report['modules_1_4_integrity']['unchanged']}")
    if report["warnings"]:
        print(f"\nWARNINGS ({len(report['warnings'])}):")
        for w in report["warnings"]:
            print(f"  - {w}")
    print(f"\nWrote: {out_reports/'corpus_build_report.json'}")
    print(f"Wrote: {out_reports/'corpus_build_report.md'}")

    failed = (not report["modules_1_4_integrity"]["unchanged"]
              or any(not r["statistics"]["row_accounting_balances"] for r in report["corpora"].values()))
    sys.exit(1 if failed else 0)


def _write_md(report: dict[str, Any], path: Path) -> None:
    L = ["# Module 5 Stage 3 — Claim/Evidence Corpus Build", "",
         f"Generated: {report['generated_at']}", "",
         f"**Split policy: {report['split_policy']}**", "",
         report["split_policy_note"], ""]
    for name, r in report["corpora"].items():
        s = r["statistics"]
        L += [f"## {name}", "",
              "| | |", "|---|---:|",
              f"| Input rows | {s['n_input_rows']} |",
              f"| Claims built | {s['n_claims_built']} |",
              f"| Dropped (no claim text) | {s['n_dropped_no_claim_text']} |",
              f"| Dropped (below min chars) | {s['n_dropped_below_min_chars']} |",
              f"| Row accounting balances | {s['row_accounting_balances']} |",
              f"| Unique evidence items | {s['n_unique_evidence_items']} |",
              f"| Evidence mentions | {s['n_evidence_mentions']} |",
              f"| Claims without evidence | {s['n_claims_with_no_evidence']} |",
              f"| Claim groups | {r['grouping']['n_groups']} |",
              f"| Largest group | {r['grouping']['largest_group_size']} |", "",
              f"Official splits: `{r['splits']}`", "",
              f"Languages: `{s['by_language']}`", "",
              f"Labels: `{s['by_label']}`", "",
              f"Evidence extraction: `{s['evidence_extraction_methods']}`", ""]
        if r["leakage"]:
            L += [f"### Leakage gate on official splits: "
                  f"**{'PASS' if r['leakage']['passed'] else 'FAIL'}**", "",
                  r["leakage"]["interpretation"], ""]
            for c in r["leakage"]["checks"]:
                L.append(f"- {'OK' if c['passed'] else 'FAIL'} `{c['name']}`: {c['detail']}")
            L.append("")
    L += ["## Modules 1-4 integrity", "",
          f"Unchanged: **{report['modules_1_4_integrity']['unchanged']}**", "",
          "## Warnings", ""] + ([f"- {w}" for w in report["warnings"]] or ["- none"])
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
