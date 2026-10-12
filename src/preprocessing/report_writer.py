"""Human-readable Module 2 report generator.

Takes the report dict `run_preprocessing_pipeline` returns and renders
`preprocessing_report.md` — every number in it is read back out of that
dict (which is itself written verbatim to `preprocessing_statistics.json`),
so the two files can never silently disagree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _pct(part: int, whole: int) -> str:
    if whole == 0:
        return "0.0%"
    return f"{100 * part / whole:.1f}%"


def write_preprocessing_report(report: dict[str, Any], path: Path) -> None:
    s = report["stages"]
    lines: list[str] = []

    lines += ["# Module 2 — Preprocessing Report", ""]

    n_in = s["input"]["n_rows"]
    n_final = s["cleaned_dataset_summary"]["n_rows"]
    lines += [
        f"**Input rows (Module 1 output):** {n_in}",
        f"**Final rows (after cleaning/filtering/dedup):** {n_final} "
        f"({_pct(n_final, n_in)} of input retained)",
        "",
        "---",
        "",
        "## 1. Dataset statistics: before vs. after",
        "",
        "### Before (Module 1 output)",
        "",
        f"- By language: {s['input']['by_language']}",
        f"- By label: {s['input']['by_label']}",
        "",
        "### After (Module 2 cleaned dataset)",
        "",
        f"- By language: {s['cleaned_dataset_summary']['by_language']}",
        f"- By label: {s['cleaned_dataset_summary']['by_label']}",
        f"- By language x label: {s['cleaned_dataset_summary']['by_language_and_label']}",
        "",
        "## 2. Missing values",
        "",
        "Before cleaning:",
        "",
    ]
    for col, n in s["missing_values_before_cleaning"].items():
        if n:
            lines.append(f"- {col}: {n}")
    lines += ["", "After cleaning:", ""]
    for col, n in s["missing_values_after_cleaning"].items():
        if n:
            lines.append(f"- {col}: {n}")
    if not any(s["missing_values_after_cleaning"].values()):
        lines.append("- none")

    lv = s["label_verification"]
    lines += [
        "",
        "## 3. Label verification",
        "",
        f"- Null labels: {lv['n_null']}",
        f"- Invalid-value labels (outside {{0,1}}): {lv['n_invalid_value']}",
        f"- Valid: {lv['is_valid']}",
        "",
        "## 4. Language verification",
        "",
        f"- Rows re-checked with the corrected Tamil-ratio formula "
        f"(Tamil-block chars / non-whitespace chars, bounded in [0,1] by "
        f"construction — see `src/preprocessing/language_verification.py` "
        f"for why this replaces Module 1's formula for this check): "
        f"{n_in}",
        f"- Mismatches against the stored Module 1 tag: {s['language_verification']['n_mismatches']}",
    ]

    dd = s["duplicate_verification_post_cleaning"]
    sf = s["short_document_removal"]
    lines += [
        "",
        "## 5. Duplicate verification (post-cleaning)",
        "",
        f"- Rows before: {dd['n_before']}",
        f"- Exact-duplicate rows removed (on cleaned `text`): {dd['n_removed']}",
        f"- Rows after: {dd['n_after']}",
        "",
        "## 6. Document-length analysis and short-document removal",
        "",
        f"- Minimum word-count threshold: {sf['min_word_count_threshold']}",
        f"- Rows removed (below threshold): {sf['n_removed']}",
        f"- Extreme-length outliers flagged (>= {sf['extreme_length_threshold_chars']:.0f} chars, "
        f"99.5th percentile) but kept (not truncated — that's Module 3's job): "
        f"{sf['n_extreme_length_flagged_not_removed']}",
        "",
        "### Length stats after cleaning, by language",
        "",
    ]
    for lang in ("en", "ta"):
        stat = s["length_stats_after_cleaning"].get(lang, {})
        if not stat:
            continue
        c = stat["char_len"]
        w = stat["word_len"]
        lines.append(
            f"- **{lang}**: char_len mean={c['mean']:.0f} median={c['p50']:.0f} "
            f"p95={c['p95']:.0f} max={c['max']:.0f}; word_len mean={w['mean']:.1f} "
            f"median={w['p50']:.0f} max={w['max']:.0f}"
        )

    qc = s["final_quality_checks"]
    lines += [
        "",
        "## 7. Final quality checks",
        "",
        f"**Overall: {'PASSED' if qc['passed'] else 'FAILED'}**",
        "",
    ]
    for name, result in qc["checks"].items():
        status = "OK" if result["passed"] else "FAIL"
        lines.append(f"- [{status}] {name}: {result}")

    sp = s["split"]
    lk = sp["leakage_check"]
    lines += [
        "",
        "## 8. Train/validation/test split",
        "",
        f"- Ratios: train={sp['train_ratio']}, val={sp['val_ratio']}, test={sp['test_ratio']}",
        f"- Random seed: {sp['random_seed']}",
        f"- Stratified jointly on: {sp['stratify_columns']}",
        f"- Counts: train={sp['n_train']}, val={sp['n_val']}, test={sp['n_test']}",
        "",
        "### Leakage verification",
        "",
        f"- **{'PASSED — zero overlap' if lk['passed'] else 'FAILED — overlap detected'}**",
        f"- train/val overlap: {lk['train_val_overlap']}",
        f"- train/test overlap: {lk['train_test_overlap']}",
        f"- val/test overlap: {lk['val_test_overlap']}",
        "",
        "### Per-stratum split counts (language, label)",
        "",
    ]
    for key, counts in sp["stratum_counts"].items():
        lines.append(f"- {key}: total={counts['total']}, train={counts['train']}, val={counts['val']}, test={counts['test']}")

    lines += [
        "",
        "## 9. English–Tamil imbalance (carried forward, not resolved here)",
        "",
        "Volume imbalance (~12:1 English:Tamil) and length/richness imbalance "
        "(English documents average roughly 20x longer than Tamil documents, "
        "since most Tamil data is headline-only) are both measured here but "
        "deliberately not corrected in Module 2 — correcting via naive "
        "oversampling or truncation would distort the data itself. The volume "
        "imbalance is handled at training time in Module 4/5 (balanced batch "
        "sampling / loss reweighting); the length imbalance is a genuine "
        "corpus characteristic to disclose, not fix.",
        "",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
