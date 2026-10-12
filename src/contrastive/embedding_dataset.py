"""Module 4 — loading and integrity-checking the frozen Module 3 embeddings.

Module 3's outputs are a FROZEN, READ-ONLY baseline (verified 18/18 on
2026-08-09). Every function here opens files for reading only; nothing
in Module 4 writes into `embeddings/`.

The integrity gate in `verify_frozen_inputs()` is not ceremony. During
Module 3 a different, larger dataset version (69,552 rows) was silently
substituted and every count-consistency check still passed, because
each check compared the embeddings against whatever CSV happened to be
present rather than against the verified figures. Module 4 therefore
asserts Module 2's counts explicitly and refuses to train if they
disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SplitEmbeddings:
    """One split's frozen Module 3 arrays."""

    split: str
    pooling: str
    embeddings: np.ndarray  # (N, 768) float32, L2-normalised
    labels: np.ndarray      # (N,) int64
    languages: np.ndarray   # (N,) <U2
    record_ids: np.ndarray  # (N,) <U64

    @property
    def n(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    def language_mask(self, language: str) -> np.ndarray:
        return self.languages.astype(str) == language

    def subset(self, mask: np.ndarray) -> "SplitEmbeddings":
        """Row subset (used for zero-shot English-only training and for
        per-language evaluation). Never mutates the frozen arrays.
        """
        return SplitEmbeddings(
            split=self.split,
            pooling=self.pooling,
            embeddings=self.embeddings[mask],
            labels=self.labels[mask],
            languages=self.languages[mask],
            record_ids=self.record_ids[mask],
        )

    def summary(self) -> dict[str, Any]:
        langs, lang_counts = np.unique(self.languages.astype(str), return_counts=True)
        labels, label_counts = np.unique(self.labels, return_counts=True)
        return {
            "split": self.split,
            "pooling": self.pooling,
            "n_rows": self.n,
            "embedding_dim": self.dim,
            "by_language": {str(k): int(v) for k, v in zip(langs, lang_counts)},
            "by_label": {str(k): int(v) for k, v in zip(labels, label_counts)},
        }


def load_split(embeddings_dir: str | Path, split: str, pooling: str) -> SplitEmbeddings:
    """Read one split's frozen arrays. Read-only."""
    d = Path(embeddings_dir) / split
    emb_path = d / f"{pooling}_embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"Frozen Module 3 embeddings not found: {emb_path}. Module 4 requires "
            "Module 3's verified outputs; it does not generate embeddings."
        )
    return SplitEmbeddings(
        split=split,
        pooling=pooling,
        embeddings=np.load(emb_path),
        labels=np.load(d / "labels.npy"),
        languages=np.load(d / "languages.npy"),
        record_ids=np.load(d / "record_ids.npy"),
    )


def load_all_splits(embeddings_dir: str | Path, pooling: str, splits: list[str]) -> dict[str, SplitEmbeddings]:
    return {s: load_split(embeddings_dir, s, pooling) for s in splits}


def verify_frozen_inputs(
    data: dict[str, SplitEmbeddings],
    expected_counts: dict[str, int],
    expected_total: int,
    expected_dim: int,
    expected_tamil_rows: int,
    reference_record_ids: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Hard gate run before any training.

    Returns a report dict with `passed` plus per-check detail. The
    caller is expected to STOP on failure rather than continue with
    degraded inputs -- silently training on the wrong data is precisely
    the failure mode this guards against.

    `reference_record_ids` lets the caller assert that a second pooling
    strategy's record_ids are identical to the first's, catching any
    row-order drift between the two frozen arrays.
    """
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str, data_: Any = None) -> None:
        entry = {"name": name, "passed": bool(passed), "detail": detail}
        if data_ is not None:
            entry["data"] = data_
        checks.append(entry)

    # ---- counts ----
    counts = {s: e.n for s, e in data.items()}
    total = sum(counts.values())
    counts_ok = all(counts.get(s) == n for s, n in expected_counts.items()) and total == expected_total
    add("dataset_counts", counts_ok,
        f"{counts}, total={total} (expected {expected_counts}, total={expected_total})",
        {"observed": counts, "expected": expected_counts, "total": total})

    # ---- dimensions ----
    dims = {s: e.dim for s, e in data.items()}
    add("embedding_dimensions", all(d == expected_dim for d in dims.values()),
        f"{dims} (expected {expected_dim})", dims)

    # ---- NaN / Inf ----
    nan_inf = {s: {"nan": int(np.isnan(e.embeddings).sum()), "inf": int(np.isinf(e.embeddings).sum())}
               for s, e in data.items()}
    add("no_nan_or_inf", all(v["nan"] == 0 and v["inf"] == 0 for v in nan_inf.values()),
        f"{nan_inf}", nan_inf)

    # ---- L2 normalisation (Module 3 wrote unit vectors) ----
    norms = {}
    norm_ok = True
    for s, e in data.items():
        n = np.linalg.norm(e.embeddings, axis=1)
        norms[s] = {"min": round(float(n.min()), 6), "max": round(float(n.max()), 6)}
        if not np.allclose(n, 1.0, atol=1e-3):
            norm_ok = False
    add("l2_normalised", norm_ok, f"{norms}", norms)

    # ---- metadata alignment ----
    align = {s: {"labels": len(e.labels) == e.n,
                 "languages": len(e.languages) == e.n,
                 "record_ids": len(e.record_ids) == e.n}
             for s, e in data.items()}
    add("metadata_lengths_match", all(all(v.values()) for v in align.values()), f"{align}", align)

    # ---- label / language domains ----
    bad_labels = {s: sorted(set(int(x) for x in np.unique(e.labels)) - {0, 1}) for s, e in data.items()}
    bad_langs = {s: sorted(set(np.unique(e.languages).astype(str).tolist()) - {"en", "ta"}) for s, e in data.items()}
    add("label_language_domains",
        all(not v for v in bad_labels.values()) and all(not v for v in bad_langs.values()),
        f"unexpected labels={bad_labels}, unexpected languages={bad_langs}",
        {"labels": bad_labels, "languages": bad_langs})

    # ---- Tamil total ----
    tamil = {s: int(e.language_mask("ta").sum()) for s, e in data.items()}
    tamil_total = sum(tamil.values())
    add("tamil_row_count", tamil_total == expected_tamil_rows,
        f"{tamil}, total={tamil_total} (expected {expected_tamil_rows})", tamil)

    # ---- leakage ----
    id_sets = {s: set(e.record_ids.astype(str).tolist()) for s, e in data.items()}
    overlaps = {}
    names = sorted(id_sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlaps[f"{a}_vs_{b}"] = len(id_sets[a] & id_sets[b])
    add("zero_split_leakage", all(v == 0 for v in overlaps.values()), f"{overlaps}", overlaps)

    # ---- record_id uniqueness ----
    dup = {s: int(len(e.record_ids) - len(set(e.record_ids.astype(str).tolist()))) for s, e in data.items()}
    add("record_ids_unique", all(v == 0 for v in dup.values()), f"{dup}", dup)

    # ---- cross-pooling record_id identity ----
    if reference_record_ids is not None:
        same = {s: bool(np.array_equal(data[s].record_ids.astype(str),
                                       reference_record_ids[s].astype(str)))
                for s in data if s in reference_record_ids}
        add("record_ids_match_reference_pooling", all(same.values()),
            f"{same}", same)

    passed = all(c["passed"] for c in checks)
    return {
        "passed": passed,
        "n_checks": len(checks),
        "n_passed": sum(1 for c in checks if c["passed"]),
        "n_failed": sum(1 for c in checks if not c["passed"]),
        "checks": checks,
        "split_summaries": {s: e.summary() for s, e in data.items()},
    }
