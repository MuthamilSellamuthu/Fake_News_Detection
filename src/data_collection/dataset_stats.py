"""Dataset statistics reporting — Module 1.

Pure reporting: no filtering or transformation happens here. Takes the
final standardized DataFrame and produces a JSON-serializable summary
plus a short human-readable Markdown report, so the numbers behind
"Module 1 collected N rows" are backed by an artifact instead of a
claim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def compute_dataset_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute counts/distributions the validation report and README summary need."""
    stats: dict[str, Any] = {
        "total_rows": int(len(df)),
        "by_source": df["source"].value_counts(dropna=False).to_dict(),
        "by_language": df["language"].value_counts(dropna=False).to_dict(),
        "by_label": {str(k): int(v) for k, v in df["label"].value_counts(dropna=False).to_dict().items()},
        "by_source_and_label": (
            df.groupby("source")["label"]
            .value_counts()
            .unstack(fill_value=0)
            .rename(columns={0: "real_0", 1: "fake_1"})
            .to_dict(orient="index")
        ),
        "text_length_chars": {
            "mean": float(df["text"].astype(str).str.len().mean()) if len(df) else 0.0,
            "median": float(df["text"].astype(str).str.len().median()) if len(df) else 0.0,
            "min": int(df["text"].astype(str).str.len().min()) if len(df) else 0,
            "max": int(df["text"].astype(str).str.len().max()) if len(df) else 0,
        },
    }

    # Normalize numpy/pandas key types (value_counts keys can be numpy scalars)
    stats["by_source"] = {str(k): int(v) for k, v in stats["by_source"].items()}
    stats["by_language"] = {str(k): int(v) for k, v in stats["by_language"].items()}

    return stats


def _label_balance_line(label_counts: dict[str, int]) -> str:
    fake = label_counts.get("1", 0)
    real = label_counts.get("0", 0)
    total = fake + real
    if total == 0:
        return "No labeled rows."
    return f"Fake (1): {fake} ({100 * fake / total:.1f}%) | Real (0): {real} ({100 * real / total:.1f}%)"


def write_statistics_reports(stats: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    lines = [
        "# Module 1 — Dataset Statistics Report",
        "",
        f"**Total standardized rows:** {stats['total_rows']}",
        "",
        "## Rows by source",
        "",
    ]
    for source, count in sorted(stats["by_source"].items(), key=lambda kv: -kv[1]):
        lines.append(f"- {source}: {count}")

    lines += ["", "## Rows by language", ""]
    for lang, count in sorted(stats["by_language"].items(), key=lambda kv: -kv[1]):
        lines.append(f"- {lang}: {count}")

    lines += ["", "## Label balance (overall)", "", _label_balance_line(stats["by_label"]), ""]

    lines += ["## Label balance by source", ""]
    for source, counts in stats["by_source_and_label"].items():
        fake = int(counts.get("fake_1", 0))
        real = int(counts.get("real_0", 0))
        total = fake + real
        pct = f"{100 * fake / total:.1f}%" if total else "n/a"
        lines.append(f"- {source}: fake={fake}, real={real} (fake share: {pct})")

    tl = stats["text_length_chars"]
    lines += [
        "",
        "## Text length (characters)",
        "",
        f"- mean: {tl['mean']:.1f}",
        f"- median: {tl['median']:.1f}",
        f"- min: {tl['min']}",
        f"- max: {tl['max']}",
        "",
    ]

    md_path.parent.mkdir(parents=True, exist_ok=True)
    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
