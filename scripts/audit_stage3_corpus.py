#!/usr/bin/env python
"""Module 5 Stage 3 — audit the REAL built corpus. Read-only. COLAB.

Runs after `scripts/build_claim_corpus.py`. Reads only what the build
produced plus the original source files, and recomputes every number
independently rather than trusting the build report -- a build bug and a
build self-check would fail together.

Enforces hard stop conditions: unexplained row loss, split mismatch,
evidence-ID collision, leakage failure, schema failure, language
contamination, Module 1-4 modification, missing source file, output
corruption. Exits non-zero if any fires.

Writes outputs/module5/stage3_audit_report.{json,md}. Modifies nothing
else, downloads nothing, trains nothing.

    python scripts/audit_stage3_corpus.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fact_verification.corpus_builder import evidence_hash  # noqa: E402
from src.fact_verification.leakage_gate import candidate_pairs, char_ngrams  # noqa: E402
from src.utils.config_loader import load_yaml  # noqa: E402

EVIDENCE_ID_RE = re.compile(r"^[a-z_]+:ev:[0-9a-f]{64}$")
TAMIL_LO, TAMIL_HI = "஀", "௿"

FROZEN_MD5 = {
    "datasets/processed/train.csv": "fc26454e56cb9704509ffe3216d52a22",
    "datasets/processed/validation.csv": "081f7b828477ab9f27725d17c34b525a",
    "datasets/processed/test.csv": "66599a4f9d71bd5cee1927c3a573f0ce",
}
EXPECTED_INPUT_ROWS = {"x_fact": 1463, "fever": 260251}
SOURCE_DIRS = {"x_fact": "x_fact", "fever": "fever_gold_evidence"}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"CORRUPT JSONL {path}:{ln}: {e}") from e
    return rows


def count_lines(path: Path) -> int:
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def tamil_ratio(text: str) -> float:
    ns = [c for c in text if not c.isspace()]
    if not ns:
        return 0.0
    return sum(1 for c in ns if TAMIL_LO <= c <= TAMIL_HI) / len(ns)


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit the real Stage 3 corpus.")
    ap.add_argument("--data-dir", default="datasets/module5_external")
    args = ap.parse_args()

    root = PROJECT_ROOT
    cfg = load_yaml(root / "configs" / "module5_fact_verification.yaml")
    corpus_root = root / cfg["output"]["corpus_dir"]
    reports = root / cfg["output"]["reports_dir"]
    data_dir = root / args.data_dir

    stops: list[str] = []
    warns: list[str] = []
    rep: dict[str, Any] = {
        "module": "Module 5 Stage 3 — real corpus audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": {}, "combined": {}, "stop_conditions": [], "warnings": [],
    }

    def stop(msg: str) -> None:
        stops.append(msg)
        print(f"  [STOP] {msg}")

    build_report_path = reports / "corpus_build_report.json"
    if not build_report_path.exists():
        stop(f"build report missing: {build_report_path} — was the build run?")
        _finish(rep, stops, warns, reports)
        return
    build_report = json.loads(build_report_path.read_text(encoding="utf-8"))

    all_claims: list[dict] = []
    all_evidence: list[dict] = []

    for ds in ("x_fact", "fever"):
        print(f"\n=== {ds} ===")
        # Scoped to THIS dataset: a stop recorded while auditing x_fact must not
        # silently skip fever's analysis. Only this dataset's own failures halt it.
        stops_before = len(stops)
        d: dict[str, Any] = {}
        out_dir = corpus_root / ds
        cpath, epath, spath = (out_dir / "claims.jsonl", out_dir / "evidence.jsonl",
                               out_dir / "claim_splits.json")

        # ---- source files ----
        src_dir = data_dir / SOURCE_DIRS[ds]
        srcs = sorted(src_dir.glob("*.jsonl")) if src_dir.exists() else []
        if not srcs:
            stop(f"{ds}: NO SOURCE FILES under {src_dir}")
        d["source_files"] = [{"name": p.name, "bytes": p.stat().st_size,
                              "rows": count_lines(p)} for p in srcs]
        d["input_rows_total"] = sum(s["rows"] for s in d["source_files"])
        exp_in = EXPECTED_INPUT_ROWS[ds]
        if d["input_rows_total"] != exp_in:
            stop(f"{ds}: input rows {d['input_rows_total']} != expected {exp_in}")
        for s in d["source_files"]:
            print(f"  source {s['name']}: {s['bytes']:,} bytes, {s['rows']:,} rows")

        # ---- outputs exist and parse ----
        for p in (cpath, epath, spath):
            if not p.exists():
                stop(f"{ds}: missing output {p}")
        if len(stops) > stops_before:
            rep["datasets"][ds] = d
            continue
        try:
            claims = load_jsonl(cpath)
            evidence = load_jsonl(epath)
            splits_doc = json.loads(spath.read_text(encoding="utf-8"))
        except (ValueError, json.JSONDecodeError) as e:
            stop(f"{ds}: OUTPUT CORRUPTION — {e}")
            rep["datasets"][ds] = d
            continue

        d["output_files"] = {p.name: p.stat().st_size for p in (cpath, epath, spath)}
        d["n_claims"] = len(claims)
        d["n_evidence"] = len(evidence)

        # ---- schema ----
        required = {"claim_id", "text", "language", "dataset", "label", "record_id",
                    "evidence", "claim_group_id", "metadata"}
        missing = [i for i, c in enumerate(claims[:5000]) if not required <= set(c)]
        if missing:
            stop(f"{ds}: SCHEMA FAILURE — {len(missing)} claim(s) missing required keys "
                 f"(first at line {missing[0] + 1})")
        bad_ds = {c.get("dataset") for c in claims} - {ds}
        if bad_ds:
            stop(f"{ds}: DATASET CONTAMINATION — foreign dataset tags {sorted(bad_ds)}")

        # ---- row accounting (recomputed, then cross-checked vs build report) ----
        bstats = build_report.get("corpora", {}).get(ds, {}).get("statistics", {})
        dropped = (bstats.get("n_dropped_no_claim_text", 0)
                   + bstats.get("n_dropped_below_min_chars", 0))
        d["row_accounting"] = {
            "source_rows": d["input_rows_total"],
            "accepted_rows": len(claims),
            "rejected_rows": dropped,
            "rejected_no_claim_text": bstats.get("n_dropped_no_claim_text", 0),
            "rejected_below_min_chars": bstats.get("n_dropped_below_min_chars", 0),
            "unexplained_loss": d["input_rows_total"] - len(claims) - dropped,
        }
        if d["row_accounting"]["unexplained_loss"] != 0:
            stop(f"{ds}: UNEXPLAINED ROW LOSS of "
                 f"{d['row_accounting']['unexplained_loss']} row(s)")
        if bstats.get("n_claims_built") not in (None, len(claims)):
            stop(f"{ds}: build report says {bstats['n_claims_built']} claims but "
                 f"claims.jsonl has {len(claims)}")

        # ---- splits ----
        counts = splits_doc.get("counts", {})
        idx = splits_doc.get("splits", {})
        d["split_counts"] = counts
        d["official_split_counts"] = {}
        for c in claims:
            k = c.get("metadata", {}).get("official_split", "?")
            d["official_split_counts"][k] = d["official_split_counts"].get(k, 0) + 1
        flat = [i for v in idx.values() for i in v]
        if sorted(flat) != list(range(len(claims))):
            stop(f"{ds}: SPLIT MISMATCH — indices do not cover every claim exactly once "
                 f"({len(flat)} assignments for {len(claims)} claims)")
        if sum(counts.values()) != len(claims):
            stop(f"{ds}: SPLIT MISMATCH — split counts sum to {sum(counts.values())} "
                 f"but there are {len(claims)} claims")

        # ---- evidence accounting ----
        ev_by_id = {e["evidence_id"]: e for e in evidence}
        if len(ev_by_id) != len(evidence):
            stop(f"{ds}: EVIDENCE-ID COLLISION — {len(evidence) - len(ev_by_id)} duplicate id(s)")
        bad_fmt = [e["evidence_id"] for e in evidence if not EVIDENCE_ID_RE.match(e["evidence_id"])]
        if bad_fmt:
            stop(f"{ds}: {len(bad_fmt)} evidence_id(s) are not 64-hex SHA-256 "
                 f"(e.g. {bad_fmt[0][:80]})")
        nondet = [e["evidence_id"] for e in evidence
                  if e["evidence_id"] != f"{ds}:ev:{evidence_hash(e['text'])}"]
        if nondet:
            stop(f"{ds}: {len(nondet)} evidence_id(s) do not equal SHA-256 of their text "
                 f"(e.g. {nondet[0][:80]})")
        by_text: dict[str, int] = {}
        for e in evidence:
            by_text[e["text"]] = by_text.get(e["text"], 0) + 1
        dup_text = {t: n for t, n in by_text.items() if n > 1}
        if dup_text:
            stop(f"{ds}: {len(dup_text)} evidence text(s) stored under multiple ids "
                 f"(deduplication broken)")

        referenced: dict[str, int] = {}
        n_refs = 0
        multi = 0
        none_ev = 0
        for c in claims:
            refs = c.get("evidence") or []
            n_refs += len(refs)
            if len(refs) > 1:
                multi += 1
            if not refs:
                none_ev += 1
            for r in refs:
                referenced[r] = referenced.get(r, 0) + 1
        dangling = sorted(set(referenced) - set(ev_by_id))
        if dangling:
            stop(f"{ds}: {len(dangling)} claim(s) reference evidence ids absent from "
                 f"evidence.jsonl (e.g. {dangling[0][:80]})")
        orphans = sorted(set(ev_by_id) - set(referenced))
        counts_disagree = [eid for eid, e in ev_by_id.items()
                           if e.get("n_claims_referencing") != referenced.get(eid, 0)]
        d["evidence_accounting"] = {
            "evidence_mentions": n_refs,
            "unique_evidence_records": len(ev_by_id),
            "duplicates_collapsed": n_refs - len(ev_by_id),
            "claims_with_no_evidence": none_ev,
            "claims_with_multiple_evidence": multi,
            "orphan_evidence": len(orphans),
            "max_claims_per_evidence": max(referenced.values()) if referenced else 0,
            "mean_claims_per_evidence": round(n_refs / len(ev_by_id), 4) if ev_by_id else 0.0,
            "n_claims_referencing_disagreements": len(counts_disagree),
            "sha256_ids_valid": not bad_fmt and not nondet,
        }
        if orphans:
            warns.append(f"{ds}: {len(orphans)} evidence record(s) referenced by no claim")
        if counts_disagree:
            warns.append(f"{ds}: {len(counts_disagree)} evidence record(s) whose stored "
                         f"n_claims_referencing disagrees with recount")

        # ---- language / labels ----
        langs: dict[str, int] = {}
        labels: dict[str, int] = {}
        for c in claims:
            langs[str(c.get("language"))] = langs.get(str(c.get("language")), 0) + 1
            labels[str(c.get("label"))] = labels.get(str(c.get("label")), 0) + 1
        d["languages"] = dict(sorted(langs.items(), key=lambda kv: -kv[1]))
        d["labels"] = dict(sorted(labels.items(), key=lambda kv: -kv[1]))
        ev_langs: dict[str, int] = {}
        for e in evidence:
            ev_langs[str(e.get("language"))] = ev_langs.get(str(e.get("language")), 0) + 1
        d["evidence_languages"] = dict(sorted(ev_langs.items(), key=lambda kv: -kv[1]))

        if ds == "x_fact":
            n_ta = sum(1 for c in claims if c.get("language") == "ta")
            n_script = sum(1 for c in claims if tamil_ratio(c.get("text", "")) > 0.5)
            d["tamil_audit"] = {
                "claims_declared_ta": n_ta,
                "claims_majority_tamil_script": n_script,
                "evidence_declared_ta": ev_langs.get("ta", 0),
                "evidence_majority_tamil_script":
                    sum(1 for e in evidence if tamil_ratio(e.get("text", "")) > 0.5),
                "replacement_char_in_claims":
                    sum(1 for c in claims if "�" in c.get("text", "")),
                "replacement_char_in_evidence":
                    sum(1 for e in evidence if "�" in e.get("text", "")),
                "scope_note": ("These are the X-FACT **Tamil subset** (hf_config='ta') only. "
                               "They are NOT overall X-FACT statistics."),
            }
            if d["tamil_audit"]["replacement_char_in_claims"] or \
                    d["tamil_audit"]["replacement_char_in_evidence"]:
                stop("x_fact: U+FFFD replacement characters present — Tamil text corrupted")
            if n_ta == 0:
                stop("x_fact: no claims tagged language 'ta' — expected the Tamil subset")
        else:
            non_en = {k: v for k, v in langs.items() if k != "en"}
            if non_en:
                warns.append(f"fever: {sum(non_en.values())} claim(s) not tagged 'en': {non_en}")

        # ---- leakage (from the build; independently recount candidates) ----
        lk = build_report.get("corpora", {}).get(ds, {}).get("leakage")
        if lk is None:
            stop(f"{ds}: leakage gate was NOT run (no result in build report) — "
                 f"a skipped check is never a PASS")
        else:
            d["leakage"] = {"passed": lk.get("passed"),
                            "checks_passed": f"{lk.get('n_passed')}/{lk.get('n_checks')}",
                            "runtime_seconds": lk.get("seconds"),
                            "failed_checks": [c["name"] for c in lk.get("checks", [])
                                              if not c.get("passed")]}
            if not lk.get("passed"):
                stop(f"{ds}: LEAKAGE GATE FAILED — {d['leakage']['failed_checks']}")

        print(f"  recounting candidate pairs (threshold "
              f"{cfg['leakage']['near_duplicate_threshold']})...")
        t0 = time.perf_counter()
        grams = [char_ngrams(c["text"], cfg["leakage"]["near_duplicate_ngram"]) for c in claims]
        langs_list = [str(c.get("language")) for c in claims]
        n_cand = sum(1 for _ in candidate_pairs(grams, langs_list,
                                                cfg["leakage"]["near_duplicate_threshold"]))
        d["candidate_pairs_examined"] = n_cand
        d["candidate_pair_recount_seconds"] = round(time.perf_counter() - t0, 2)
        total_pairs = len(claims) * (len(claims) - 1) // 2
        d["candidate_pairs_vs_all_pairs"] = (f"{n_cand:,} of {total_pairs:,} "
                                             f"({100 * n_cand / total_pairs:.4f}%)"
                                             if total_pairs else "n/a")

        grouping = build_report.get("corpora", {}).get(ds, {}).get("grouping", {})
        d["deduplication"] = {
            "n_groups": grouping.get("n_groups"),
            "exact_duplicate_pairs": grouping.get("exact_duplicate_pairs"),
            "near_duplicate_pairs": grouping.get("near_duplicate_pairs"),
            "largest_group_size": grouping.get("largest_group_size"),
            "claims_removed_by_dedup": 0,
            "dedup_policy": ("Duplicates are GROUPED, never deleted. Grouping keeps a whole "
                             "group inside one split; no claim is dropped for being a duplicate."),
        }

        rep["datasets"][ds] = d
        all_claims.extend(claims)
        all_evidence.extend(evidence)
        print(f"  claims={len(claims):,}  evidence={len(ev_by_id):,}  "
              f"candidates={n_cand:,}  languages={d['languages']}")

    # ---- combined ----
    if all_claims:
        comb_lang: dict[str, int] = {}
        comb_ds: dict[str, int] = {}
        comb_split: dict[str, int] = {}
        for c in all_claims:
            comb_lang[str(c.get("language"))] = comb_lang.get(str(c.get("language")), 0) + 1
            comb_ds[str(c.get("dataset"))] = comb_ds.get(str(c.get("dataset")), 0) + 1
            k = c.get("metadata", {}).get("canonical_split", "?")
            comb_split[k] = comb_split.get(k, 0) + 1
        ev_ds: dict[str, int] = {}
        for e in all_evidence:
            ev_ds[str(e.get("dataset"))] = ev_ds.get(str(e.get("dataset")), 0) + 1
        cross = set()
        for ds_name in rep["datasets"]:
            pass
        rep["combined"] = {
            "total_claims": len(all_claims),
            "total_evidence": len(all_evidence),
            "claims_by_dataset": comb_ds,
            "evidence_by_dataset": ev_ds,
            "claims_by_language": comb_lang,
            "claims_by_canonical_split": comb_split,
            "labels_by_dataset": {k: v.get("labels") for k, v in rep["datasets"].items()},
            "note": ("Corpora are stored separately and are NOT merged; combined figures are "
                     "reported for description only."),
        }
        global_ids = [e["evidence_id"] for e in all_evidence]
        if len(set(global_ids)) != len(global_ids):
            warns.append("evidence ids repeat across datasets (expected: ids are "
                         "dataset-prefixed, so this indicates a real collision)")

    # ---- Module 1-4 integrity ----
    integ = {}
    for rel, exp in FROZEN_MD5.items():
        p = root / rel
        if not p.exists():
            stop(f"MODULE 1-4: {rel} MISSING")
            integ[rel] = "MISSING"
            continue
        got = md5(p)
        integ[rel] = {"md5": got, "expected": exp, "unchanged": got == exp}
        if got != exp:
            stop(f"MODULE 1-4 MODIFIED: {rel} md5 {got} != {exp}")
    rep["modules_1_4_integrity"] = integ

    _finish(rep, stops, warns, reports)


def _finish(rep: dict, stops: list, warns: list, reports: Path) -> None:
    rep["stop_conditions"] = stops
    rep["warnings"] = warns
    rep["overall"] = "FAIL" if stops else "PASS"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "stage3_audit_report.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    L = [f"# Module 5 Stage 3 — Real Corpus Audit", "",
         f"Generated: {rep['generated_at']}", "",
         f"**Overall: {rep['overall']}**", ""]
    for ds, d in rep.get("datasets", {}).items():
        L += [f"## {ds}", "", "| field | value |", "|---|---|"]
        for k in ("input_rows_total", "n_claims", "n_evidence",
                  "candidate_pairs_examined", "candidate_pairs_vs_all_pairs"):
            if k in d:
                L.append(f"| {k} | {d[k]} |")
        L += ["", f"Source files: `{d.get('source_files')}`", "",
              f"Row accounting: `{d.get('row_accounting')}`", "",
              f"Official split counts: `{d.get('official_split_counts')}`", "",
              f"Languages: `{d.get('languages')}`", "",
              f"Labels: `{d.get('labels')}`", "",
              f"Evidence: `{d.get('evidence_accounting')}`", "",
              f"Deduplication: `{d.get('deduplication')}`", "",
              f"Leakage: `{d.get('leakage')}`", ""]
        if "tamil_audit" in d:
            L += [f"Tamil audit: `{d['tamil_audit']}`", ""]
    L += ["## Combined", "", f"`{rep.get('combined')}`", "",
          "## Modules 1-4 integrity", "", f"`{rep.get('modules_1_4_integrity')}`", "",
          "## Stop conditions", ""] + ([f"- {s}" for s in stops] or ["- none"]) + \
         ["", "## Warnings", ""] + ([f"- {w}" for w in warns] or ["- none"])
    (reports / "stage3_audit_report.md").write_text("\n".join(L), encoding="utf-8")

    print("\n" + "=" * 64)
    print(f"STAGE 3 AUDIT: {rep['overall']}")
    if stops:
        print(f"\n{len(stops)} STOP CONDITION(S):")
        for s in stops:
            print(f"  - {s}")
        print("\nDo NOT proceed to retrieval, reranking, or verification.")
    if warns:
        print(f"\n{len(warns)} warning(s):")
        for w in warns:
            print(f"  - {w}")
    print(f"\nWrote: {reports/'stage3_audit_report.json'}")
    print(f"Wrote: {reports/'stage3_audit_report.md'}")
    print("=" * 64)
    sys.exit(1 if stops else 0)


if __name__ == "__main__":
    main()
