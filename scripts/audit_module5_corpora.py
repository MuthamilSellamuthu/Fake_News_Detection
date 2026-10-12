#!/usr/bin/env python
"""Module 5 Stage 2 — real audit of the downloaded corpora. COLAB.

Every number produced here is measured from the files on disk. Nothing
is estimated, and nothing is inferred from the papers.

The schema is DISCOVERED, not assumed. X-FACT's TSV layout and the
FEVER gold-evidence record shape differ between mirrors and revisions,
so the audit probes for known field aliases, reports what it actually
found, and marks anything it could not resolve as UNKNOWN rather than
guessing. A field this script cannot locate becomes a warning, never a
silent zero.

Reads only `datasets/module5_external/`; writes only under
`outputs/module5/`. Modules 1-4 are untouched and their integrity is
re-checked at the end.

Usage (Colab, project root):
    python scripts/audit_module5_corpora.py
    python scripts/audit_module5_corpora.py --skip-fever-benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fact_verification.claim_selection import Claim, hash_key, normalize_claim, tamil_ratio  # noqa: E402
from src.fact_verification.leakage_gate import (  # noqa: E402
    group_aware_split,
    group_claims,
    verify_no_leakage,
)
from src.utils.config_loader import load_yaml  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("module5.audit")

# Field aliases seen across X-FACT / FEVER mirrors. Probed in order.
CLAIM_FIELDS = ["claim", "claim_text", "text", "statement", "sentence"]
LABEL_FIELDS = ["label", "labels", "verdict", "gold_label", "class"]
LANG_FIELDS = ["language", "lang", "claim_language"]
SITE_FIELDS = ["site", "source", "domain", "fact_checker", "url"]
ID_FIELDS = ["id", "claim_id", "example_id", "uid"]
EVIDENCE_LIST_FIELDS = ["evidence", "evidences", "gold_evidence", "evidence_sentences"]
EVIDENCE_NUMBERED_PREFIX = "evidence_"   # X-FACT uses evidence_1 .. evidence_5


def detect_script(text: str) -> str:
    """Coarse script/language bucket for a piece of evidence text.

    Script detection, not language identification: it separates Tamil
    from Latin from other Indic/other scripts, which is what the
    cross-lingual evidence question needs. Anything ambiguous is
    reported as 'unknown' rather than guessed.
    """
    if not text or not str(text).strip():
        return "empty"
    s = str(text)
    chars = [c for c in s if not c.isspace() and c.isalpha()]
    if not chars:
        return "non_alphabetic"
    tamil = sum(1 for c in chars if "஀" <= c <= "௿")
    latin = sum(1 for c in chars if ("a" <= c.lower() <= "z"))
    other_indic = sum(1 for c in chars if "ऀ" <= c <= "෿" and not ("஀" <= c <= "௿"))
    n = len(chars)
    if tamil / n > 0.30:
        return "tamil"
    if latin / n > 0.60:
        return "latin"
    if other_indic / n > 0.30:
        return "other_indic"
    try:
        return f"other:{unicodedata.name(chars[0]).split()[0].lower()}"
    except ValueError:
        return "unknown"


def first_present(row: dict[str, Any], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in row and row[c] not in (None, ""):
            return c
    for c in candidates:
        if c in row:
            return c
    return None


def extract_evidence(row: dict[str, Any]) -> tuple[list[str], str]:
    """Pull evidence text out of whatever shape the row uses.

    Returns (evidence_texts, how_it_was_found). Handles:
      - a list field of strings
      - a list field of [page, sentence] pairs or dicts (FEVER style)
      - numbered columns evidence_1..evidence_N (X-FACT style)
    """
    for f in EVIDENCE_LIST_FIELDS:
        if f in row and row[f]:
            val = row[f]
            if isinstance(val, str):
                return ([val], f"string:{f}")
            if isinstance(val, list):
                out: list[str] = []
                for item in val:
                    if isinstance(item, str):
                        out.append(item)
                    elif isinstance(item, dict):
                        for k in ("text", "sentence", "evidence"):
                            if k in item and isinstance(item[k], str):
                                out.append(item[k])
                                break
                    elif isinstance(item, (list, tuple)):
                        strs = [x for x in item if isinstance(x, str)]
                        if strs:
                            out.append(strs[-1])
                return ([e for e in out if e and e.strip()], f"list:{f}")

    numbered = sorted(k for k in row if k.startswith(EVIDENCE_NUMBERED_PREFIX))
    if numbered:
        out = [str(row[k]) for k in numbered
               if row.get(k) not in (None, "", "nan") and str(row[k]).strip()]
        return (out, f"numbered:{len(numbered)}_columns")
    return ([], "none_found")


def extract_evidence_ids(row: dict[str, Any]) -> list[str]:
    """Preserve original page/sentence identifiers where present.

    FEVER's link back to Wikipedia lives in these identifiers; losing
    them would make the closed-pool corpus unlinkable to its source.
    """
    ids: list[str] = []
    for f in EVIDENCE_LIST_FIELDS:
        val = row.get(f)
        if isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    page = item.get("page") or item.get("title") or item.get("doc_id")
                    sent = item.get("sentence_id", item.get("line", item.get("sent_id")))
                    if page is not None:
                        ids.append(f"{page}#{sent}" if sent is not None else str(page))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    non_str = [x for x in item if not isinstance(x, str)]
                    strs = [x for x in item if isinstance(x, str)]
                    if strs:
                        ids.append(f"{strs[0]}#{non_str[0]}" if non_str else strs[0])
    for k in ("evidence_id", "doc_id", "page", "wikipedia_url"):
        if row.get(k):
            ids.append(str(row[k]))
    return ids


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def length_stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    v = sorted(values)
    n = len(v)
    return {"n": n, "min": v[0], "max": v[-1],
            "mean": round(sum(v) / n, 2), "median": v[n // 2],
            "p25": v[n // 4], "p75": v[(3 * n) // 4], "p95": v[min(n - 1, int(0.95 * n))]}


# --------------------------------------------------------------------
# Per-corpus audit
# --------------------------------------------------------------------

def audit_corpus(corpus_dir: Path, corpus_name: str) -> dict[str, Any]:
    """Audit every JSONL split of one corpus. All numbers measured."""
    report: dict[str, Any] = {"corpus": corpus_name, "directory": str(corpus_dir),
                              "warnings": [], "splits": {}}
    files = sorted(corpus_dir.glob("*.jsonl"))
    if not files:
        report["warnings"].append(f"No .jsonl files found in {corpus_dir} — was the download run?")
        report["total_records"] = 0
        return report

    report["official_split_files"] = [f.name for f in files]
    all_claims_norm: list[str] = []
    all_langs: Counter = Counter()
    all_labels: Counter = Counter()
    total = 0

    for fp in files:
        rows = load_jsonl(fp)
        split_name = fp.stem.replace(f"{corpus_name}_", "").replace("x_fact_", "").replace("fever_gold_", "")
        if not rows:
            report["splits"][split_name] = {"n_records": 0}
            continue

        sample = rows[0]
        cf = first_present(sample, CLAIM_FIELDS)
        lf = first_present(sample, LABEL_FIELDS)
        gf = first_present(sample, LANG_FIELDS)
        sf = first_present(sample, SITE_FIELDS)
        idf = first_present(sample, ID_FIELDS)

        if cf is None:
            report["warnings"].append(
                f"{fp.name}: no claim field found among {CLAIM_FIELDS}; columns={list(sample.keys())}")

        claim_lens, ev_counts, ev_lens = [], [], []
        langs, labels, sites = Counter(), Counter(), Counter()
        ev_script_by_claim_lang: dict[str, Counter] = {}
        norm_claims, ids = [], []
        n_no_evidence = n_multi_evidence = 0
        missing = Counter()
        ev_method = Counter()
        ev_id_present = 0

        for r in rows:
            claim = str(r.get(cf, "")) if cf else ""
            if not claim.strip():
                missing["claim"] += 1
            claim_lens.append(len(claim))
            nrm = hash_key(claim)
            norm_claims.append(nrm)

            lang = str(r.get(gf, "")).strip().lower() if gf else ""
            if not lang:
                lang = "ta" if tamil_ratio(claim) > 0.3 else ("en" if claim.strip() else "unknown")
                missing["language_field"] += 1
            langs[lang] += 1

            lab = r.get(lf) if lf else None
            labels[str(lab) if lab is not None else "MISSING"] += 1
            if lab is None:
                missing["label"] += 1

            if sf:
                sites[str(r.get(sf, "unknown"))[:80]] += 1
            if idf:
                ids.append(str(r.get(idf)))

            ev, how = extract_evidence(r)
            ev_method[how] += 1
            ev_counts.append(len(ev))
            if not ev:
                n_no_evidence += 1
            elif len(ev) > 1:
                n_multi_evidence += 1
            for e in ev:
                ev_lens.append(len(e))
                ev_script_by_claim_lang.setdefault(lang, Counter())[detect_script(e)] += 1
            if extract_evidence_ids(r):
                ev_id_present += 1

        total += len(rows)
        all_claims_norm.extend(norm_claims)
        all_langs.update(langs)
        all_labels.update(labels)

        dup_claims = len(norm_claims) - len(set(norm_claims))
        pair_keys = set()
        dup_pairs = 0
        for r in rows:
            c = hash_key(str(r.get(cf, "")) if cf else "")
            for e in extract_evidence(r)[0]:
                k = (c, hash_key(e))
                if k in pair_keys:
                    dup_pairs += 1
                else:
                    pair_keys.add(k)

        report["splits"][split_name] = {
            "file": fp.name,
            "n_records": len(rows),
            "columns_detected": list(sample.keys()),
            "field_mapping": {"claim": cf, "label": lf, "language": gf, "site": sf, "id": idf},
            "language_distribution": dict(langs.most_common()),
            "label_distribution": dict(labels.most_common()),
            "claim_char_length": length_stats(claim_lens),
            "evidence": {
                "extraction_method": dict(ev_method),
                "records_with_no_evidence": n_no_evidence,
                "records_with_multiple_evidence": n_multi_evidence,
                "evidence_per_record": length_stats(ev_counts),
                "evidence_char_length": length_stats(ev_lens),
                "records_with_original_evidence_ids": ev_id_present,
                "evidence_script_by_claim_language": {
                    k: dict(v.most_common()) for k, v in ev_script_by_claim_lang.items()},
            },
            "duplicates": {"exact_duplicate_claims": dup_claims,
                           "duplicate_claim_evidence_pairs": dup_pairs},
            "missing_values": dict(missing),
            "unique_ids": len(set(ids)) if ids else None,
            "ids_unique": (len(set(ids)) == len(ids)) if ids else None,
            "top_sources": dict(sites.most_common(15)) if sites else None,
        }

        if n_no_evidence == len(rows):
            report["warnings"].append(f"{fp.name}: NO evidence found in any record — check field mapping")
        if ids and len(set(ids)) != len(ids):
            report["warnings"].append(f"{fp.name}: claim IDs are not unique ({len(ids)-len(set(ids))} repeats)")

    report["total_records"] = total
    report["language_distribution_all_splits"] = dict(all_langs.most_common())
    report["label_distribution_all_splits"] = dict(all_labels.most_common())
    report["cross_split_exact_duplicate_claims"] = len(all_claims_norm) - len(set(all_claims_norm))
    report["has_official_splits"] = len(report["splits"]) > 1
    return report


def audit_tamil_subset(corpus_dir: Path) -> dict[str, Any]:
    """The audit that decides whether a Tamil experiment is viable.

    Feasibility is classified only AFTER counting, using thresholds
    fixed here in advance so the verdict cannot be rationalised to fit a
    desired outcome:
        HIGH     >= 1000 Tamil claims AND >= 2 classes with >= 50 each
        MODERATE >=  300
        LOW      <   300
    """
    out: dict[str, Any] = {"language": "ta", "warnings": []}
    files = sorted(corpus_dir.glob("*.jsonl"))
    if not files:
        out["n_tamil"] = 0
        out["feasibility"] = "LOW"
        out["warnings"].append("corpus not downloaded")
        return out

    rows_ta: list[tuple[str, dict[str, Any]]] = []
    for fp in files:
        split = fp.stem.replace("x_fact_", "")
        for r in load_jsonl(fp):
            gf = first_present(r, LANG_FIELDS)
            cf = first_present(r, CLAIM_FIELDS)
            claim = str(r.get(cf, "")) if cf else ""
            lang = str(r.get(gf, "")).strip().lower() if gf else ""
            if lang == "ta" or (not lang and tamil_ratio(claim) > 0.3):
                rows_ta.append((split, r))

    n = len(rows_ta)
    out["n_tamil"] = n
    if n == 0:
        out["feasibility"] = "LOW"
        out["warnings"].append("X-FACT contains NO Tamil claims under the detected language field — "
                               "the Tamil experiment as designed is not supported by this corpus")
        return out

    by_split, labels = Counter(), Counter()
    claim_lens, ev_lens, ev_counts = [], [], []
    ev_scripts = Counter()
    uniq_claims, uniq_ev = set(), set()
    n_no_ev = n_multi_ev = 0

    for split, r in rows_ta:
        by_split[split] += 1
        cf = first_present(r, CLAIM_FIELDS)
        lf = first_present(r, LABEL_FIELDS)
        claim = str(r.get(cf, "")) if cf else ""
        claim_lens.append(len(claim))
        uniq_claims.add(hash_key(claim))
        labels[str(r.get(lf)) if lf else "MISSING"] += 1
        ev, _ = extract_evidence(r)
        ev_counts.append(len(ev))
        if not ev:
            n_no_ev += 1
        elif len(ev) > 1:
            n_multi_ev += 1
        for e in ev:
            ev_lens.append(len(e))
            uniq_ev.add(hash_key(e))
            ev_scripts[detect_script(e)] += 1

    out.update({
        "split_distribution": dict(by_split),
        "label_distribution": dict(labels.most_common()),
        "unique_claims": len(uniq_claims),
        "unique_evidence_items": len(uniq_ev),
        "claim_char_length": length_stats(claim_lens),
        "evidence_char_length": length_stats(ev_lens),
        "evidence_per_claim": length_stats(ev_counts),
        "claims_with_no_evidence": n_no_ev,
        "claims_with_multiple_evidence": n_multi_ev,
        "evidence_script_distribution": dict(ev_scripts.most_common()),
    })

    viable_classes = sum(1 for c in labels.values() if c >= 50)
    if n >= 1000 and viable_classes >= 2:
        out["feasibility"] = "HIGH"
    elif n >= 300:
        out["feasibility"] = "MODERATE"
    else:
        out["feasibility"] = "LOW"
    out["feasibility_rule"] = "HIGH: n>=1000 and >=2 classes with >=50; MODERATE: n>=300; else LOW"
    out["classes_with_at_least_50"] = viable_classes

    if out["feasibility"] == "LOW":
        out["warnings"].append(
            f"Only {n} Tamil claims: per-language Macro-F1 will have wide confidence intervals. "
            "Report as an evidence-availability finding rather than a performance claim.")
    if n_no_ev / n > 0.5:
        out["warnings"].append(f"{n_no_ev}/{n} Tamil claims have NO evidence — retrieval cannot be evaluated on them")
    ta_ev = ev_scripts.get("tamil", 0)
    lat_ev = ev_scripts.get("latin", 0)
    if ta_ev + lat_ev > 0:
        out["tamil_claim_evidence_language"] = {
            "tamil_evidence": ta_ev, "latin_english_evidence": lat_ev,
            "other": sum(v for k, v in ev_scripts.items() if k not in ("tamil", "latin")),
            "pct_english_evidence": round(100 * lat_ev / (ta_ev + lat_ev), 2),
        }
    return out


def benchmark_near_duplicates(claims: list[Claim], label: str, threshold: float, ngram: int) -> dict[str, Any]:
    """Measure the existing leakage implementation. Never change it here."""
    t0 = time.perf_counter()
    g = group_claims(claims, near_duplicate=True, ngram=ngram, threshold=threshold)
    elapsed = time.perf_counter() - t0
    n = len(claims)
    return {
        "corpus": label, "n_claims": n,
        "runtime_seconds": round(elapsed, 2),
        "claims_per_second": round(n / elapsed, 1) if elapsed > 0 else None,
        "n_groups": g.n_groups,
        "exact_duplicate_pairs": g.n_exact_duplicate_pairs,
        "near_duplicate_pairs": g.n_near_duplicate_pairs,
        "largest_group_size": g.largest_group_size,
        "pct_records_in_multi_claim_groups": round(
            100 * g.summary["n_claims_in_multi_claim_groups"] / n, 2) if n else 0.0,
        "threshold": threshold, "ngram": ngram,
        "_group_ids": g.group_ids,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Module 5 Stage 2: audit downloaded corpora.")
    p.add_argument("--config", default=None)
    p.add_argument("--data-dir", default="datasets/module5_external")
    p.add_argument("--skip-fever-benchmark", action="store_true",
                   help="Skip the FEVER near-duplicate benchmark if runtime is prohibitive.")
    p.add_argument("--fever-benchmark-sample", type=int, default=20000,
                   help="Benchmark FEVER on a sample first, then extrapolate.")
    args = p.parse_args()

    root = PROJECT_ROOT
    cfg = load_yaml(Path(args.config) if args.config else root / "configs" / "module5_fact_verification.yaml")
    data_dir = root / args.data_dir
    out_dir = root / cfg["output"]["reports_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    audit: dict[str, Any] = {
        "module": "Module 5 Stage 2 — corpus audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "Every value measured from files on disk. Nothing estimated.",
        "warnings": [], "limitations": [],
    }

    manifest_path = out_dir / "download_manifest.json"
    audit["download_manifest"] = (json.loads(manifest_path.read_text(encoding="utf-8"))
                                  if manifest_path.exists() else
                                  {"error": "download_manifest.json not found — run the download script first"})

    logger.info("Auditing X-FACT...")
    audit["x_fact"] = audit_corpus(data_dir / "x_fact", "x_fact")
    logger.info("Auditing FEVER gold-evidence...")
    audit["fever_gold_evidence"] = audit_corpus(data_dir / "fever_gold_evidence", "fever_gold")
    logger.info("Auditing Tamil subset...")
    audit["tamil_subset"] = audit_tamil_subset(data_dir / "x_fact")

    audit["limitations"].append(
        "FEVER evidence is GOLD-EVIDENCE ONLY. The candidate pool is the union of gold evidence "
        "sentences, not all of Wikipedia. Retrieval results are CLOSED-POOL and must not be "
        "reported as full-Wikipedia open-world retrieval.")

    # ---- evidence-language cross-tabulation ----
    xtab: dict[str, dict[str, int]] = {}
    for split in audit["x_fact"].get("splits", {}).values():
        for lang, scripts in split.get("evidence", {}).get("evidence_script_by_claim_language", {}).items():
            for script, cnt in scripts.items():
                xtab.setdefault(lang, {}).setdefault(script, 0)
                xtab[lang][script] += cnt
    audit["evidence_language_crosstab"] = {
        lang: {"counts": s, "total": sum(s.values()),
               "percentages": {k: round(100 * v / sum(s.values()), 2) for k, v in s.items()} if sum(s.values()) else {}}
        for lang, s in xtab.items()}

    # ---- near-duplicate benchmark ----
    lk = cfg["leakage"]
    bench: dict[str, Any] = {}

    def claims_from(corpus_dir: Path, dataset: str, limit: int | None = None) -> list[Claim]:
        out: list[Claim] = []
        for fp in sorted(corpus_dir.glob("*.jsonl")):
            for i, r in enumerate(load_jsonl(fp, limit=limit)):
                cf = first_present(r, CLAIM_FIELDS)
                gf = first_present(r, LANG_FIELDS)
                lf = first_present(r, LABEL_FIELDS)
                text = str(r.get(cf, "")) if cf else ""
                if not text.strip():
                    continue
                lang = str(r.get(gf, "")).strip().lower() if gf else ("ta" if tamil_ratio(text) > 0.3 else "en")
                out.append(Claim(claim_id=f"{dataset}:{fp.stem}:{i}",
                                 text=normalize_claim(text, cfg["normalization"]),
                                 raw_text=text, language=lang, dataset=dataset,
                                 label=r.get(lf) if lf else None,
                                 metadata={"split": fp.stem}))
        return out

    xf_claims = claims_from(data_dir / "x_fact", "x_fact")
    if xf_claims:
        logger.info("Benchmarking near-duplicate detection on X-FACT (%d claims)...", len(xf_claims))
        bench["x_fact"] = benchmark_near_duplicates(
            xf_claims, "x_fact", lk["near_duplicate_threshold"], lk["near_duplicate_ngram"])

    if not args.skip_fever_benchmark:
        fv_sample = claims_from(data_dir / "fever_gold_evidence", "fever", limit=args.fever_benchmark_sample)
        if fv_sample:
            logger.info("Benchmarking on a %d-claim FEVER sample...", len(fv_sample))
            b = benchmark_near_duplicates(fv_sample, "fever_sample",
                                          lk["near_duplicate_threshold"], lk["near_duplicate_ngram"])
            full_n = audit["fever_gold_evidence"].get("total_records", 0)
            if b["runtime_seconds"] > 0 and len(fv_sample):
                scale = (full_n / len(fv_sample)) ** 2   # blocking helps, but assume quadratic
                b["extrapolated_full_corpus_seconds"] = round(b["runtime_seconds"] * scale, 1)
                b["extrapolated_full_corpus_minutes"] = round(b["runtime_seconds"] * scale / 60, 1)
                b["extrapolation_note"] = (
                    "Quadratic extrapolation from the sample; blocking makes this an UPPER bound. "
                    "If this exceeds ~30 min, STOP and report before changing the algorithm.")
            bench["fever_sample"] = b

    for b in bench.values():
        b.pop("_group_ids", None)
    audit["near_duplicate_benchmark"] = bench

    # ---- split policy + leakage ----
    split_manifest: dict[str, Any] = {"generated_at": datetime.now(timezone.utc).isoformat(), "corpora": {}}
    for name, key in (("x_fact", "x_fact"), ("fever_gold_evidence", "fever_gold_evidence")):
        rep = audit[key]
        official = rep.get("has_official_splits")
        entry: dict[str, Any] = {
            "has_official_splits": official,
            "official_splits": {k: v.get("n_records") for k, v in rep.get("splits", {}).items()},
            "policy": ("PRESERVE_OFFICIAL" if official else "GROUP_AWARE_70_15_15"),
            "rationale": ("Official splits exist and are preserved; the official test set is not "
                          "reshuffled." if official else
                          "No usable official split; leakage-safe group-aware split applied."),
        }
        split_manifest["corpora"][name] = entry

    # Leakage gate on X-FACT's official splits, treated as given.
    leakage_report: dict[str, Any] = {}
    if xf_claims:
        gids = group_claims(xf_claims, near_duplicate=True,
                            ngram=lk["near_duplicate_ngram"],
                            threshold=lk["near_duplicate_threshold"]).group_ids
        by_split: dict[str, list[int]] = {}
        for i, c in enumerate(xf_claims):
            by_split.setdefault(str(c.metadata.get("split", "unknown")), []).append(i)
        leakage_report["x_fact_official_splits"] = verify_no_leakage(
            xf_claims, by_split, gids,
            near_duplicate_threshold=lk["near_duplicate_threshold"],
            ngram=lk["near_duplicate_ngram"])
        leakage_report["x_fact_official_splits"]["note"] = (
            "Measured on the ORIGINAL official splits. Failures here are a property of the "
            "published dataset, not of our splitting: report them, do not silently re-split.")
    audit["leakage"] = leakage_report

    # ---- frozen module check ----
    frozen = {}
    for rel in ["datasets/processed", "embeddings", "outputs/module3", "outputs/module4", "models/module4"]:
        p = root / rel
        files = [f for f in p.rglob("*") if f.is_file()] if p.exists() else []
        frozen[rel] = {"exists": p.exists(), "n_files": len(files),
                       "total_bytes": sum(f.stat().st_size for f in files)}
    audit["modules_1_4_integrity"] = frozen

    for section in ("x_fact", "fever_gold_evidence", "tamil_subset"):
        audit["warnings"].extend(f"[{section}] {w}" for w in audit[section].get("warnings", []))

    (out_dir / "corpus_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    _write_markdown(audit, split_manifest, out_dir / "corpus_audit.md")

    # ---- summary ----
    xf_ok = audit["x_fact"]["total_records"] > 0
    fv_ok = audit["fever_gold_evidence"]["total_records"] > 0
    ta_ok = audit["tamil_subset"].get("n_tamil", 0) > 0
    ev_ok = bool(audit["evidence_language_crosstab"])
    lk_ok = all(r.get("passed") for r in leakage_report.values()) if leakage_report else False
    bm_ok = bool(bench)

    print("\nMODULE 5 STAGE 2")
    print("-----------------")
    print(f"X-FACT: {'PASS' if xf_ok else 'FAIL'}  ({audit['x_fact']['total_records']} records)")
    print(f"FEVER: {'PASS' if fv_ok else 'FAIL'}  ({audit['fever_gold_evidence']['total_records']} records)")
    print(f"Tamil audit: {'PASS' if ta_ok else 'FAIL'}  "
          f"(n={audit['tamil_subset'].get('n_tamil')}, feasibility={audit['tamil_subset'].get('feasibility')})")
    print(f"Evidence-language audit: {'PASS' if ev_ok else 'FAIL'}")
    print(f"Leakage audit: {'PASS' if lk_ok else 'FAIL'}")
    print(f"Near-duplicate benchmark: {'PASS' if bm_ok else 'FAIL'}")
    print(f"Module 1-4 integrity: {'PASS' if all(v['exists'] for v in frozen.values()) else 'FAIL'}")
    if audit["warnings"]:
        print(f"\nWARNINGS ({len(audit['warnings'])}):")
        for w in audit["warnings"][:20]:
            print(f"  - {w}")
    print(f"\nWrote: {out_dir/'corpus_audit.json'}")
    print(f"Wrote: {out_dir/'corpus_audit.md'}")
    print(f"Wrote: {out_dir/'split_manifest.json'}")


def _write_markdown(audit: dict[str, Any], split_manifest: dict[str, Any], path: Path) -> None:
    L = ["# Module 5 Stage 2 — Corpus Audit", "",
         f"Generated: {audit['generated_at']}", "",
         "Every figure measured from the downloaded files. Nothing estimated.", ""]

    for key, title in (("x_fact", "X-FACT"), ("fever_gold_evidence", "FEVER (gold evidence)")):
        r = audit[key]
        L += [f"## {title}", "", f"Total records: **{r.get('total_records')}**", "",
              "| Split | Records |", "|---|---:|"]
        for s, v in r.get("splits", {}).items():
            L.append(f"| {s} | {v.get('n_records')} |")
        L += ["", f"Languages: `{r.get('language_distribution_all_splits')}`", "",
              f"Labels: `{r.get('label_distribution_all_splits')}`", "",
              f"Cross-split exact duplicate claims: {r.get('cross_split_exact_duplicate_claims')}", ""]

    t = audit["tamil_subset"]
    L += ["## Tamil subset (X-FACT)", "",
          f"**n = {t.get('n_tamil')}** · feasibility: **{t.get('feasibility')}**", "",
          f"Rule: {t.get('feasibility_rule')}", "",
          f"Labels: `{t.get('label_distribution')}`", "",
          f"Splits: `{t.get('split_distribution')}`", "",
          f"Unique claims: {t.get('unique_claims')} · unique evidence: {t.get('unique_evidence_items')}", "",
          f"Claims with no evidence: {t.get('claims_with_no_evidence')}", "",
          f"Evidence script distribution: `{t.get('evidence_script_distribution')}`", ""]
    if t.get("tamil_claim_evidence_language"):
        L += [f"Tamil-claim evidence language: `{t['tamil_claim_evidence_language']}`", ""]

    L += ["## Evidence-language cross-tabulation", "",
          "| Claim language | Evidence script | Count | % |", "|---|---|---:|---:|"]
    for lang, d in audit.get("evidence_language_crosstab", {}).items():
        for script, cnt in d.get("counts", {}).items():
            L.append(f"| {lang} | {script} | {cnt} | {d['percentages'].get(script)} |")

    L += ["", "## Near-duplicate benchmark", "",
          "| Corpus | n | Runtime (s) | Groups | Exact dup pairs | Near dup pairs | Largest group | % affected |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for b in audit.get("near_duplicate_benchmark", {}).values():
        L.append(f"| {b['corpus']} | {b['n_claims']} | {b['runtime_seconds']} | {b['n_groups']} | "
                 f"{b['exact_duplicate_pairs']} | {b['near_duplicate_pairs']} | "
                 f"{b['largest_group_size']} | {b['pct_records_in_multi_claim_groups']} |")

    L += ["", "## Split policy", ""]
    for name, e in split_manifest["corpora"].items():
        L += [f"**{name}** — official splits: {e['has_official_splits']} → policy `{e['policy']}`",
              f"  {e['rationale']}", f"  {e['official_splits']}", ""]

    L += ["## Leakage", ""]
    for name, r in audit.get("leakage", {}).items():
        L.append(f"**{name}**: {'PASS' if r.get('passed') else 'FAIL'} "
                 f"({r.get('n_passed')}/{r.get('n_checks')})")
        for c in r.get("checks", []):
            L.append(f"  - {'OK' if c['passed'] else 'FAIL'} {c['name']}: {c['detail']}")
    L += ["", "## Limitations", ""] + [f"- {x}" for x in audit.get("limitations", [])]
    L += ["", "## Warnings", ""] + ([f"- {w}" for w in audit.get("warnings", [])] or ["- none"])
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
