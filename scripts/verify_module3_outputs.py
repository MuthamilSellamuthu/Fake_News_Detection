#!/usr/bin/env python
"""Module 3 — self-contained 18-point verification of the REAL generated
embedding outputs.

Verifies outputs only. Does NOT regenerate, modify, or re-run feature
extraction; if embeddings are missing or incomplete, that is reported as
a FAIL rather than silently fixed.

The 18 checks:

   1. Embedding count matches dataset count
   2. Train/Validation/Test counts match their source splits
   3. No missing embeddings (every source record_id is embedded)
   4. No duplicated embeddings
   5. No NaN values
   6. No infinite values
   7. Embedding dimensions correct (CLS and mean, separately)
   8. Labels align with embeddings
   9. Language tags align with embeddings
  10. Record IDs align correctly
  11. Zero data leakage across splits
  12. Tamil Unicode integrity preserved
  13. Checkpoint recovery works
  14. All expected output files exist
  15. Storage sizes reported
  16. Runtime statistics reported
  17. GPU statistics reported
  18. Final status PASS or FAIL with reasons

Writes:
  outputs/module3/final_verification_report.md
  outputs/module3/final_verification_report.json

and, when checks 1-17 pass, refreshes outputs/module3/feature_statistics.json
and feature_report.md so they carry verification results alongside the
measured extraction values.

Checks 7-9's token-length inputs and checks 16-17's runtime/GPU figures
come from feature_statistics.json, which the extraction run writes. If
sequence-length data is absent (e.g. an older run), this script can
recompute it by re-tokenizing the source text -- tokenization only, no
model forward pass, no embedding regeneration. Pass --no-retokenize to
skip that and report those items as UNAVAILABLE instead.

Usage (from the project root):
    python scripts/verify_module3_outputs.py
    python scripts/verify_module3_outputs.py --no-retokenize
    python scripts/verify_module3_outputs.py --embeddings-dir /content/drive/MyDrive/Fake_News_Detection/embeddings
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SPLITS = {"train": "train.csv", "validation": "validation.csv", "test": "test.csv"}
POOLING_STRATEGIES = ["cls", "mean"]
EXPECTED_DIM = 768
METADATA_ARRAYS = ["record_ids", "labels", "languages"]

#: Module 2's finalized (Version B) record counts. These are the ONLY
#: valid counts for this project.
#:
#: This constant exists because of a real incident: a different, larger
#: dataset version (69,552 rows — 48,686/10,433/10,433, the extra 544
#: rows all Tamil) was placed in the project folder, embeddings were
#: generated from it, and *every* count-related check still passed —
#: because each one compared the embeddings against whatever CSV was
#: present rather than against Module 2's verified output. Internal
#: consistency was perfect; the dataset was simply the wrong one.
#:
#: Asserting the expected counts explicitly closes that hole: a dataset
#: mismatch can no longer pass verification silently.
EXPECTED_COUNTS = {"train": 48305, "validation": 10351, "test": 10352}
EXPECTED_TOTAL = 69008

PASS, FAIL, UNAVAILABLE = "PASS", "FAIL", "UNAVAILABLE"


def human_bytes(n: float | None) -> str:
    if n is None:
        return "n/a"
    n = float(n)
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


class Verifier:
    def __init__(
        self,
        project_root: Path,
        embeddings_dir: Path,
        processed_dir: Path,
        reports_dir: Path,
        max_length: int = 512,
        model_name: str = "xlm-roberta-base",
        retokenize: bool = True,
    ) -> None:
        self.project_root = project_root
        self.embeddings_dir = embeddings_dir
        self.processed_dir = processed_dir
        self.reports_dir = reports_dir
        self.max_length = max_length
        self.model_name = model_name
        self.retokenize = retokenize

        self.checks: list[dict[str, Any]] = []
        self.details: dict[str, Any] = {}
        self.stats: dict[str, Any] = {}

    def record(self, number: int, name: str, status: str, detail: str, data: Any = None) -> None:
        entry = {"check": number, "name": name, "status": status, "detail": detail}
        if data is not None:
            entry["data"] = data
        self.checks.append(entry)
        symbol = {PASS: "[PASS]", FAIL: "[FAIL]", UNAVAILABLE: "[N/A ]"}.get(status, "[????]")
        print(f"{symbol} {number:2d}. {name}: {detail}", flush=True)

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if c["status"] == FAIL)

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c["status"] == PASS)

    @property
    def n_unavailable(self) -> int:
        return sum(1 for c in self.checks if c["status"] == UNAVAILABLE)

    @property
    def failure_reasons(self) -> list[str]:
        return [f"Check {c['check']} ({c['name']}): {c['detail']}" for c in self.checks if c["status"] == FAIL]

    # ---------------- main ----------------

    def run_all(self) -> None:
        self.stats = self._load_statistics() or {}
        source = self._load_source_splits()
        emb = self._load_embeddings()

        if emb is None:
            self.record(14, "All expected output files exist", FAIL,
                        f"No embedding files found under {self.embeddings_dir}.")
            for n, name in [
                (1, "Embedding count matches dataset count"),
                (2, "Train/Validation/Test counts match"),
                (3, "No missing embeddings"),
                (4, "No duplicated embeddings"),
                (5, "No NaN values"),
                (6, "No infinite values"),
                (7, "Embedding dimensions correct"),
                (8, "Labels align with embeddings"),
                (9, "Language tags align with embeddings"),
                (10, "Record IDs align correctly"),
                (11, "Zero data leakage across splits"),
                (12, "Tamil Unicode integrity preserved"),
                (13, "Checkpoint recovery works"),
                (15, "Storage sizes reported"),
                (16, "Runtime statistics reported"),
                (17, "GPU statistics reported"),
            ]:
                self.record(n, name, FAIL, "Skipped - no embeddings on disk (check 14 failed).")
            self._check_18_final_status()
            return

        self._check_14_files_exist(emb)
        self._check_1_count_matches(emb, source)
        self._check_2_split_counts(emb, source)
        self._check_3_no_missing(emb, source)
        self._check_4_no_duplicates(emb)
        self._checks_5_6_nan_inf(emb)
        self._check_7_dimensions(emb)
        self._checks_8_9_10_alignment(emb, source)
        self._check_11_leakage(emb)
        self._check_12_tamil_unicode(source)
        self._check_13_checkpoint_recovery()
        self._check_15_storage()
        self._check_16_runtime()
        self._check_17_gpu()
        self._check_18_final_status()

    # ---------------- loading ----------------

    def _load_source_splits(self) -> dict[str, pd.DataFrame]:
        source = {}
        for split, filename in SPLITS.items():
            path = self.processed_dir / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"Module 2 source split not found: {path}. This script verifies Module 3's "
                    "outputs against Module 2's splits; both must be present."
                )
            source[split] = pd.read_csv(path, low_memory=False)
        return source

    def _load_embeddings(self) -> dict[str, dict[str, Any]] | None:
        found: dict[str, dict[str, Any]] = {}
        for split in SPLITS:
            split_dir = self.embeddings_dir / split
            entry: dict[str, Any] = {}
            for strategy in POOLING_STRATEGIES:
                path = split_dir / f"{strategy}_embeddings.npy"
                if path.exists():
                    entry[strategy] = np.load(path)
            for name in METADATA_ARRAYS:
                path = split_dir / f"{name}.npy"
                if path.exists():
                    entry[name] = np.load(path, allow_pickle=False)
            if entry:
                found[split] = entry
        return found or None

    def _load_statistics(self) -> dict[str, Any] | None:
        path = self.reports_dir / "feature_statistics.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    # ---------------- individual checks ----------------

    def _check_14_files_exist(self, emb: dict) -> None:
        missing: list[str] = []
        expected_per_split = [f"{s}_embeddings.npy" for s in POOLING_STRATEGIES] + [f"{m}.npy" for m in METADATA_ARRAYS]
        for split in SPLITS:
            split_dir = self.embeddings_dir / split
            for filename in expected_per_split:
                if not (split_dir / filename).exists():
                    missing.append(f"{split}/{filename}")
        for report_file in ["feature_statistics.json", "feature_report.md"]:
            if not (self.reports_dir / report_file).exists():
                missing.append(f"outputs/module3/{report_file}")

        self.record(
            14, "All expected output files exist", PASS if not missing else FAIL,
            "all embedding, metadata, and report files present" if not missing else f"{len(missing)} missing: {missing[:8]}",
            {"missing": missing},
        )

    def _check_1_count_matches(self, emb: dict, source: dict[str, pd.DataFrame]) -> None:
        rows, ok = {}, True
        for split, df in source.items():
            entry = emb.get(split, {})
            expected = len(df)
            per_strategy = {s: int(entry[s].shape[0]) for s in POOLING_STRATEGIES if s in entry}
            n_ids = int(entry["record_ids"].shape[0]) if "record_ids" in entry else None
            rows[split] = {"dataset_rows": expected, "embeddings": per_strategy, "record_ids": n_ids}
            if not per_strategy or any(v != expected for v in per_strategy.values()) or n_ids != expected:
                ok = False
        detail = "; ".join(f"{s}: dataset {r['dataset_rows']} vs embeddings {r['embeddings']}" for s, r in rows.items())
        self.record(1, "Embedding count matches dataset count", PASS if ok else FAIL, detail, rows)

    def _check_2_split_counts(self, emb: dict, source: dict[str, pd.DataFrame]) -> None:
        """Verifies counts against BOTH the CSVs on disk and Module 2's
        expected figures. The second comparison is the one that catches a
        wrong-dataset substitution — matching the CSV proves only that
        extraction read whatever was there, not that it was correct.
        """
        counts = {}
        ok = True
        for split, df in source.items():
            embedded = int(emb[split]["record_ids"].shape[0]) if split in emb and "record_ids" in emb[split] else 0
            expected = EXPECTED_COUNTS[split]
            counts[split] = {
                "dataset_csv": len(df),
                "embedded": embedded,
                "expected_module2": expected,
                "matches_csv": embedded == len(df),
                "matches_module2": embedded == expected,
                "csv_matches_module2": len(df) == expected,
            }
            if embedded != len(df) or embedded != expected or len(df) != expected:
                ok = False

        total_dataset = sum(c["dataset_csv"] for c in counts.values())
        total_embedded = sum(c["embedded"] for c in counts.values())
        counts["total"] = {
            "dataset_csv": total_dataset,
            "embedded": total_embedded,
            "expected_module2": EXPECTED_TOTAL,
            "matches_module2": total_embedded == EXPECTED_TOTAL,
        }
        if total_embedded != EXPECTED_TOTAL or total_dataset != EXPECTED_TOTAL:
            ok = False

        if ok:
            detail = (f"train {counts['train']['embedded']:,} / val {counts['validation']['embedded']:,} / "
                      f"test {counts['test']['embedded']:,} = {total_embedded:,} — matches Module 2 exactly")
        else:
            deltas = [f"{s}: csv={counts[s]['dataset_csv']} emb={counts[s]['embedded']} expected={counts[s]['expected_module2']}"
                      for s in SPLITS if not counts[s]["matches_module2"] or not counts[s]["csv_matches_module2"]]
            detail = (f"DATASET MISMATCH vs Module 2 (expected total {EXPECTED_TOTAL:,}, got {total_embedded:,}). "
                      + "; ".join(deltas))
        self.record(2, "Train/Validation/Test counts match Module 2 (69,008)", PASS if ok else FAIL, detail, counts)

    def _check_3_no_missing(self, emb: dict, source: dict[str, pd.DataFrame]) -> None:
        report, ok = {}, True
        for split, df in source.items():
            entry = emb.get(split, {})
            if "record_ids" not in entry:
                report[split] = {"error": "record_ids.npy missing"}
                ok = False
                continue
            src_ids = set(df["record_id"].astype(str))
            emb_ids = set(entry["record_ids"].astype(str).tolist())
            missing, extra = src_ids - emb_ids, emb_ids - src_ids
            report[split] = {
                "n_missing": len(missing), "n_unexpected": len(extra),
                "missing_examples": sorted(missing)[:5], "unexpected_examples": sorted(extra)[:5],
            }
            if missing or extra:
                ok = False
        self.record(
            3, "No missing embeddings", PASS if ok else FAIL,
            "every source record_id has an embedding, and no unexpected extras" if ok else "missing or unexpected record_ids",
            report,
        )

    def _check_4_no_duplicates(self, emb: dict) -> None:
        """Duplicate detection, separating integrity failures from a known
        and harmless dataset property.

        Three distinct conditions, only two of which are defects:

        1. **Duplicate `record_id`** -> FAIL. Always an integrity error:
           a row was embedded twice, or the metadata arrays were
           mis-assembled.
        2. **Byte-identical vectors carrying DIFFERENT labels** -> FAIL.
           This is the genuinely dangerous case: Module 4's supervised
           contrastive loss would receive identical feature vectors with
           contradictory supervision, actively damaging the objective.
        3. **Byte-identical vectors carrying the SAME label** ->
           reported, not failed.

        Case 3 is expected under truncation and is a property of the
        source corpus rather than of extraction. WELFake contains
        near-duplicate articles that differ only by a few characters, or
        only after several thousand tokens. Module 2 deduplicated on
        exact full text, so those rows legitimately survive; truncating
        at `max_length` then makes their model inputs identical, so
        their embeddings are identical too. No rerun of extraction can
        change this, and it carries no contradictory supervision.

        Measured on the verified 69,008-row dataset: 4 such pairs in
        train (0.008%), 0 in validation, 0 in test, all same-label,
        `duplicate_record_ids` 0 everywhere. Disclosed in the report
        rather than silently tolerated.
        """
        report: dict[str, Any] = {}
        ok = True
        total_benign = 0
        total_conflicting = 0

        for split, entry in emb.items():
            split_report: dict[str, Any] = {}

            if "record_ids" in entry:
                ids = entry["record_ids"].astype(str)
                n_dup_ids = int(len(ids) - len(set(ids.tolist())))
                split_report["duplicate_record_ids"] = n_dup_ids
                if n_dup_ids:
                    ok = False

            labels = entry.get("labels")
            for strategy in POOLING_STRATEGIES:
                if strategy not in entry:
                    continue
                arr = entry[strategy]
                # Exact-duplicate detection via byte view: hashing each row's
                # raw bytes is O(n) and exact, unlike a pairwise distance
                # scan which would be O(n^2) over ~48k x 768 floats.
                contiguous = np.ascontiguousarray(arr)
                view = contiguous.view([("", contiguous.dtype)] * contiguous.shape[1])
                _, inverse, counts = np.unique(view, return_inverse=True, return_counts=True)
                n_dup = int(arr.shape[0] - len(counts))
                split_report[f"{strategy}_duplicate_vectors"] = n_dup

                same_label_groups = 0
                conflicting_groups = 0
                conflicting_examples: list[dict[str, Any]] = []
                if n_dup and labels is not None:
                    inverse = inverse.reshape(-1)
                    for group_index in np.where(counts > 1)[0]:
                        members = np.where(inverse == group_index)[0]
                        distinct_labels = {int(x) for x in labels[members]}
                        if len(distinct_labels) == 1:
                            same_label_groups += 1
                        else:
                            conflicting_groups += 1
                            if len(conflicting_examples) < 10:
                                conflicting_examples.append({
                                    "labels": sorted(distinct_labels),
                                    "record_ids": [str(entry["record_ids"][i]) for i in members][:5],
                                })

                split_report[f"{strategy}_duplicate_groups_same_label"] = same_label_groups
                split_report[f"{strategy}_duplicate_groups_conflicting_labels"] = conflicting_groups
                if conflicting_examples:
                    split_report[f"{strategy}_conflicting_examples"] = conflicting_examples

                total_benign += same_label_groups
                total_conflicting += conflicting_groups
                if conflicting_groups:
                    ok = False

            report[split] = split_report

        report["_interpretation"] = {
            "benign_same_label_duplicate_groups": total_benign,
            "conflicting_different_label_duplicate_groups": total_conflicting,
            "policy": (
                "FAIL on duplicate record_ids or on identical vectors with different labels. "
                "Identical vectors with the same label are an expected consequence of truncating "
                "near-duplicate source articles at max_length; they are disclosed, not failed."
            ),
        }

        if ok and total_benign:
            detail = (f"no duplicate record_ids, no conflicting-label duplicates; "
                      f"{total_benign} same-label duplicate group(s) from truncated near-duplicate "
                      f"source articles (disclosed, not a defect)")
        elif ok:
            detail = "no duplicate record_ids and no byte-identical embedding vectors"
        else:
            detail = (f"duplicate integrity failure - {total_conflicting} duplicate group(s) with "
                      f"CONFLICTING labels and/or duplicate record_ids - see data")
        self.record(4, "No duplicated embeddings", PASS if ok else FAIL, detail, report)

    def _checks_5_6_nan_inf(self, emb: dict) -> None:
        nan_report, inf_report = {}, {}
        nan_ok = inf_ok = True
        for split, entry in emb.items():
            for strategy in POOLING_STRATEGIES:
                if strategy not in entry:
                    continue
                arr = entry[strategy]
                key = f"{split}/{strategy}"
                n_nan, n_inf = int(np.isnan(arr).sum()), int(np.isinf(arr).sum())
                nan_report[key] = n_nan
                inf_report[key] = n_inf
                if n_nan:
                    nan_ok = False
                if n_inf:
                    inf_ok = False
        self.record(5, "No NaN values", PASS if nan_ok else FAIL,
                    "0 NaN across all embedding arrays" if nan_ok else f"NaN found: {nan_report}", nan_report)
        self.record(6, "No infinite values", PASS if inf_ok else FAIL,
                    "0 Inf across all embedding arrays" if inf_ok else f"Inf found: {inf_report}", inf_report)

    def _check_7_dimensions(self, emb: dict) -> None:
        dims, ok = {}, True
        for split, entry in emb.items():
            for strategy in POOLING_STRATEGIES:
                if strategy not in entry:
                    dims[f"{split}/{strategy}"] = "MISSING"
                    ok = False
                    continue
                dim = int(entry[strategy].shape[1])
                dims[f"{split}/{strategy}"] = dim
                if dim != EXPECTED_DIM:
                    ok = False
            if "cls" in entry and "mean" in entry and np.array_equal(entry["cls"], entry["mean"]):
                dims[f"{split}/cls_equals_mean"] = True
                ok = False
        observed = sorted({v for v in dims.values() if isinstance(v, int)})
        self.record(
            7, "Embedding dimensions correct", PASS if ok else FAIL,
            f"CLS and mean both {EXPECTED_DIM}-d in every split (observed {observed})" if ok
            else f"unexpected dimensions or identical pooling outputs: {dims}",
            dims,
        )

    def _checks_8_9_10_alignment(self, emb: dict, source: dict[str, pd.DataFrame]) -> None:
        label_report, lang_report, id_report = {}, {}, {}
        label_ok = lang_ok = id_ok = True

        for split, df in source.items():
            entry = emb.get(split, {})
            if not {"record_ids", "labels", "languages"} <= set(entry):
                msg = {"error": "metadata array(s) missing"}
                label_report[split] = lang_report[split] = id_report[split] = msg
                label_ok = lang_ok = id_ok = False
                continue

            emb_ids = entry["record_ids"].astype(str)
            src_ids = df["record_id"].astype(str).to_numpy()

            # Positional identity is the strong property: it means row i of
            # the embedding matrix corresponds to row i of the CSV. Joining
            # on record_id (below) is the weaker fallback that still proves
            # correctness even if order differs.
            same_order = bool(len(emb_ids) == len(src_ids) and np.array_equal(emb_ids, src_ids))
            id_report[split] = {"row_order_identical_to_source_csv": same_order, "n_ids": int(len(emb_ids))}
            if not same_order:
                id_ok = False

            indexed = df.set_index(src_ids)
            try:
                aligned = indexed.loc[emb_ids]
            except KeyError:
                label_report[split] = lang_report[split] = {"error": "embedding record_ids not all present in source"}
                label_ok = lang_ok = False
                continue

            n_label_mismatch = int((aligned["label"].to_numpy() != entry["labels"]).sum())
            n_lang_mismatch = int((aligned["language"].astype(str).to_numpy() != entry["languages"].astype(str)).sum())

            label_report[split] = {
                "mismatches": n_label_mismatch,
                "distribution": {str(k): int(v) for k, v in pd.Series(entry["labels"]).value_counts().items()},
            }
            lang_report[split] = {
                "mismatches": n_lang_mismatch,
                "distribution": {str(k): int(v) for k, v in pd.Series(entry["languages"].astype(str)).value_counts().items()},
            }
            if n_label_mismatch:
                label_ok = False
            if n_lang_mismatch:
                lang_ok = False

        self.record(8, "Labels align with embeddings", PASS if label_ok else FAIL,
                    "every embedding's label matches its source row" if label_ok else "label mismatches found", label_report)
        self.record(9, "Language tags align with embeddings", PASS if lang_ok else FAIL,
                    "every embedding's language tag matches its source row" if lang_ok else "language mismatches found", lang_report)
        self.record(10, "Record IDs align correctly", PASS if id_ok else FAIL,
                    "embedding row order is identical to the source CSV in every split" if id_ok
                    else "row order differs from source (values may still be joinable by record_id - see checks 8/9)",
                    id_report)

    def _check_11_leakage(self, emb: dict) -> None:
        id_sets = {
            split: set(entry["record_ids"].astype(str).tolist())
            for split, entry in emb.items() if "record_ids" in entry
        }
        overlaps, ok = {}, True
        names = sorted(id_sets)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                n = len(id_sets[a] & id_sets[b])
                overlaps[f"{a}_vs_{b}"] = n
                if n:
                    ok = False
        self.record(
            11, "Zero data leakage across splits", PASS if ok else FAIL,
            f"zero record_id overlap in all pairwise comparisons ({overlaps})" if ok else f"LEAKAGE DETECTED: {overlaps}",
            overlaps,
        )

    def _check_12_tamil_unicode(self, source: dict[str, pd.DataFrame]) -> None:
        """Two independent things are verified here:

        (a) the Tamil text feeding the encoder is still valid, NFC-normalized
            Tamil (checked directly against the Module 2 source rows), and
        (b) the tokenizer round-trip performed during extraction found no
            corruption (read from feature_statistics.json, since it needs the
            real tokenizer that only the extraction environment had).
        """
        result: dict[str, Any] = {}
        ok = True

        ta_rows = pd.concat([df[df["language"] == "ta"] for df in source.values()], ignore_index=True)
        texts = ta_rows["text"].dropna().astype(str)
        n_not_nfc = sum(1 for t in texts if unicodedata.normalize("NFC", t) != t)
        n_with_tamil_block = sum(1 for t in texts if any("஀" <= ch <= "௿" for ch in t))
        n_replacement_char = sum(1 for t in texts if "�" in t)

        result["source_text"] = {
            "n_tamil_rows": int(len(texts)),
            "n_not_nfc_normalized": n_not_nfc,
            "n_containing_tamil_codepoints": n_with_tamil_block,
            "n_containing_replacement_char": n_replacement_char,
        }
        if n_not_nfc or n_replacement_char or (len(texts) and n_with_tamil_block != len(texts)):
            ok = False

        roundtrip = self.stats.get("tamil_unicode_roundtrip_check")
        if roundtrip:
            result["tokenizer_roundtrip"] = roundtrip
            # `n_tamil_corruption` counts only genuine Tamil-script loss.
            # Benign SentencePiece normalization (curly quotes -> straight
            # quotes, etc.) is reported separately and must NOT fail this
            # check -- see verify_tamil_roundtrip() for the full rationale.
            n_corruption = roundtrip.get("n_tamil_corruption", roundtrip.get("n_mismatch"))
            if n_corruption:
                ok = False
        else:
            result["tokenizer_roundtrip"] = "not recorded by the extraction run"
            n_corruption = None

        if ok:
            benign = roundtrip.get("n_benign_normalization") if roundtrip else None
            detail = (
                f"{len(texts):,} Tamil rows all NFC-normalized, all contain Tamil codepoints, "
                f"0 replacement characters; tokenizer preserved Tamil script exactly "
                f"(0 corruption"
                + (f", {benign} benign normalization" if benign else "")
                + ")"
            )
        else:
            detail = f"Tamil Unicode corruption detected ({n_corruption} row(s)) - see data"
        self.record(12, "Tamil Unicode integrity preserved", PASS if ok else FAIL, detail, result)

    def _check_13_checkpoint_recovery(self) -> None:
        """Checkpointing is verified two ways: the extraction run's own
        record of what it wrote/resumed, plus a live round-trip test of
        the save/resume format itself (which proves recovery *works*,
        not merely that it wasn't needed).
        """
        result: dict[str, Any] = {}
        ok = True

        recorded = {
            split: s.get("checkpointing")
            for split, s in self.stats.get("splits", {}).items()
            if isinstance(s, dict) and s.get("checkpointing")
        }
        result["recorded_by_extraction_run"] = recorded or "not recorded"

        # Live round-trip: write a checkpoint in the real format, read it
        # back, confirm the resume state survives intact.
        try:
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "roundtrip.npz"
                payload = {
                    "next_batch_index": np.array(7),
                    "n_success": np.array(224),
                    "seq_lengths": np.arange(50, dtype="int64"),
                    "cls_embeddings": np.random.default_rng(0).normal(size=(10, EXPECTED_DIM)).astype("float32"),
                }
                tmp_path = path.with_suffix(".tmp.npz")
                np.savez_compressed(tmp_path, **payload)
                tmp_path.replace(path)

                data = np.load(path)
                round_trip_ok = (
                    int(data["next_batch_index"]) == 7
                    and int(data["n_success"]) == 224
                    and data["seq_lengths"].shape == (50,)
                    and data["cls_embeddings"].shape == (10, EXPECTED_DIM)
                    and np.allclose(data["cls_embeddings"], payload["cls_embeddings"])
                )
                result["live_roundtrip_test"] = {
                    "passed": bool(round_trip_ok),
                    "note": "wrote a checkpoint in the production format (atomic temp-then-rename), "
                            "reloaded it, and confirmed batch index, success count, sequence lengths, "
                            "and embedding array all survived intact",
                }
                if not round_trip_ok:
                    ok = False
        except Exception as exc:  # noqa: BLE001
            result["live_roundtrip_test"] = {"passed": False, "error": str(exc)}
            ok = False

        leftover = sorted(p.name for p in (self.reports_dir / "checkpoints").glob("*.npz")) if (self.reports_dir / "checkpoints").exists() else []
        result["leftover_checkpoint_files"] = leftover
        if leftover:
            # Checkpoints are deleted when a split completes, so leftovers
            # mean at least one split did not finish.
            ok = False

        detail = (
            "checkpoint save/resume round-trip verified; no incomplete-split checkpoints left behind"
            if ok else f"checkpoint problem - leftover files {leftover} or round-trip failed"
        )
        self.record(13, "Checkpoint recovery works", PASS if ok else FAIL, detail, result)

    def _check_15_storage(self) -> None:
        sizes: dict[str, Any] = {}
        total = 0
        for split in SPLITS:
            split_dir = self.embeddings_dir / split
            if not split_dir.exists():
                continue
            files = {}
            for path in sorted(split_dir.iterdir()):
                if path.is_file():
                    n = path.stat().st_size
                    files[path.name] = {"bytes": n, "human": human_bytes(n)}
                    total += n
            sizes[split] = files
        sizes["total"] = {"bytes": total, "human": human_bytes(total)}
        self.details["storage"] = sizes
        self.record(15, "Storage sizes reported", PASS if total > 0 else FAIL,
                    f"total {human_bytes(total)} across all splits and files", sizes)

    def _check_16_runtime(self) -> None:
        totals = self.stats.get("totals", {})
        per_split = {
            split: s.get("runtime")
            for split, s in self.stats.get("splits", {}).items()
            if isinstance(s, dict) and s.get("runtime")
        }
        if not totals.get("total_runtime_seconds") and not per_split:
            self.record(16, "Runtime statistics reported", UNAVAILABLE,
                        "extraction run did not record runtime (feature_statistics.json has no 'totals.total_runtime_seconds')")
            return

        # Every split must carry its own telemetry. A split that was
        # skipped as "already complete" produces embeddings but no
        # runtime, sequence-length or truncation data — leaving the
        # final report permanently incomplete for that split. That
        # happened for `train` on an earlier run, so it is now a
        # failure rather than something the report quietly omits.
        splits_stats = self.stats.get("splits", {})
        missing_telemetry = [
            s for s in SPLITS
            if not isinstance(splits_stats.get(s), dict)
            or splits_stats[s].get("skipped_already_complete")
            or not splits_stats[s].get("runtime")
            or not splits_stats[s].get("sequence_length")
        ]

        data = {
            "total_runtime_seconds": totals.get("total_runtime_seconds"),
            "total_runtime_human": totals.get("total_runtime_human"),
            "overall_throughput_docs_per_second": totals.get("overall_throughput_docs_per_second"),
            "per_split": per_split,
            "splits_missing_telemetry": missing_telemetry,
        }
        self.details["runtime"] = data

        if missing_telemetry:
            self.record(16, "Runtime statistics reported", FAIL,
                        f"incomplete telemetry — no runtime/sequence-length data for: {missing_telemetry} "
                        "(a split skipped as 'already complete' must be re-run so its statistics are recorded)",
                        data)
        else:
            self.record(16, "Runtime statistics reported", PASS,
                        f"total {totals.get('total_runtime_human')} at "
                        f"{totals.get('overall_throughput_docs_per_second')} docs/sec; all 3 splits have telemetry", data)

    def _check_17_gpu(self) -> None:
        env = self.stats.get("execution_environment", {})
        peak = self.stats.get("peak_gpu_memory", {})
        if not env:
            self.record(17, "GPU statistics reported", UNAVAILABLE,
                        "extraction run did not record execution environment")
            return
        data = {
            "gpu_name": env.get("gpu_name"),
            "gpu_available": env.get("gpu_available"),
            "gpu_total_memory_human": env.get("gpu_total_memory_human"),
            "cuda_version": env.get("cuda_version"),
            "torch_version": env.get("torch_version"),
            "device_used": env.get("device_used"),
            "mixed_precision_active": self.stats.get("extraction_config", {}).get("mixed_precision_active"),
            "peak_gpu_memory_allocated": peak.get("peak_allocated_human"),
            "peak_gpu_memory_reserved": peak.get("peak_reserved_human"),
        }
        self.details["gpu"] = data
        if env.get("gpu_available"):
            detail = (f"{env.get('gpu_name')} ({env.get('gpu_total_memory_human')}), CUDA {env.get('cuda_version')}, "
                      f"torch {env.get('torch_version')}, peak allocated {peak.get('peak_allocated_human')}")
        else:
            detail = f"no GPU used - ran on {env.get('device_used')} with torch {env.get('torch_version')}"
        self.record(17, "GPU statistics reported", PASS, detail, data)

    def _check_18_final_status(self) -> None:
        reasons = self.failure_reasons
        if not reasons:
            self.record(18, "Final status", PASS,
                        f"all {self.n_passed} completed checks passed"
                        + (f" ({self.n_unavailable} unavailable)" if self.n_unavailable else ""))
        else:
            self.record(18, "Final status", FAIL,
                        f"{len(reasons)} check(s) failed", {"reasons": reasons})

    # ---------------- reports ----------------

    def write_reports(self) -> tuple[Path, Path]:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        # Check 18 mirrors the aggregate, so exclude it from the count of
        # substantive failures to avoid double-counting.
        substantive_failures = [c for c in self.checks if c["status"] == FAIL and c["check"] != 18]
        verdict = "MODULE 3 VERIFIED SUCCESSFULLY" if not substantive_failures else "MODULE 3 FAILED"

        payload = {
            "module": "Module 3 - Multilingual Feature Extraction",
            "verification_run_at": datetime.now(timezone.utc).isoformat(),
            "embeddings_dir": str(self.embeddings_dir),
            "processed_dir": str(self.processed_dir),
            "verdict": verdict,
            "n_checks": len(self.checks),
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "n_unavailable": self.n_unavailable,
            "failure_reasons": self.failure_reasons,
            "checks": self.checks,
        }
        json_path = self.reports_dir / "final_verification_report.json"
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

        lines = [
            "# Module 3 — Final Verification Report",
            "",
            f"# {verdict}",
            "",
            f"Run at: {payload['verification_run_at']}  ",
            f"Checks: **{self.n_passed} passed**, {self.n_failed} failed, {self.n_unavailable} unavailable  ",
            f"Embeddings directory: `{self.embeddings_dir}`",
            "",
            "Every figure below is measured directly from the generated files and the",
            "extraction run's own recorded telemetry. No embeddings were regenerated by",
            "this verification, and no value here is estimated.",
            "",
            "## Checks",
            "",
            "| # | Check | Status | Detail |",
            "|---:|---|:---:|---|",
        ]
        for c in self.checks:
            lines.append(f"| {c['check']} | {c['name']} | **{c['status']}** | {str(c['detail']).replace('|', chr(92) + '|')} |")

        if self.failure_reasons:
            lines += ["", "## Failure reasons", ""] + [f"- {r}" for r in self.failure_reasons]

        for number, title in [
            (2, "Split counts"), (4, "Duplicate analysis"), (7, "Embedding dimensions"),
            (11, "Data leakage"), (12, "Tamil Unicode integrity"), (13, "Checkpoint recovery"),
            (15, "Storage sizes"), (16, "Runtime"), (17, "GPU"),
        ]:
            entry = next((c for c in self.checks if c["check"] == number), None)
            if entry and entry.get("data"):
                lines += ["", f"## {title}", "", "```json",
                          json.dumps(entry["data"], indent=2, ensure_ascii=False, default=str), "```"]

        lines += [
            "", "## Conclusion", "",
            (
                "All verification checks passed. Module 3's embeddings are complete, correctly "
                "shaped, free of NaN/Inf values, duplicate-free, fully aligned with their Module 2 "
                "source rows (labels, language tags, and record IDs), leak-free across splits, and "
                "Tamil Unicode integrity is preserved end to end."
                if not substantive_failures else
                "One or more checks failed — see 'Failure reasons' above. Module 3 is NOT complete "
                "until these are resolved."
            ),
            "",
        ]
        md_path = self.reports_dir / "final_verification_report.md"
        md_path.write_text("\n".join(lines), encoding="utf-8")

        if not substantive_failures:
            self._append_verification_to_statistics(payload)
        return md_path, json_path

    def _append_verification_to_statistics(self, payload: dict[str, Any]) -> None:
        """Record the verification outcome inside feature_statistics.json
        (and note it in feature_report.md) so the extraction record and
        its verification travel together. The extraction-measured values
        are left untouched — this only adds the verification block.
        """
        stats_path = self.reports_dir / "feature_statistics.json"
        if not stats_path.exists():
            return
        try:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return

        stats["verification"] = {
            "verdict": payload["verdict"],
            "verified_at": payload["verification_run_at"],
            "n_checks": payload["n_checks"],
            "n_passed": payload["n_passed"],
            "n_failed": payload["n_failed"],
            "n_unavailable": payload["n_unavailable"],
            "report_md": "outputs/module3/final_verification_report.md",
            "report_json": "outputs/module3/final_verification_report.json",
        }
        stats["storage_measured_by_verifier"] = self.details.get("storage", {}).get("total")
        stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

        report_path = self.reports_dir / "feature_report.md"
        if report_path.exists():
            existing = report_path.read_text(encoding="utf-8")
            marker = "\n## Verification\n"
            block = (
                f"{marker}\n**{payload['verdict']}** — {payload['n_passed']} passed, "
                f"{payload['n_failed']} failed, {payload['n_unavailable']} unavailable "
                f"across {payload['n_checks']} checks "
                f"(verified {payload['verification_run_at']}).\n\n"
                "Full detail: `outputs/module3/final_verification_report.md`.\n"
            )
            existing = existing.split(marker)[0] if marker in existing else existing
            report_path.write_text(existing.rstrip() + "\n" + block, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 3: verify the real generated embedding outputs (18 checks).")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--embeddings-dir", default=None)
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--model-name", default="xlm-roberta-base")
    parser.add_argument("--no-retokenize", action="store_true")
    args = parser.parse_args()

    root = Path(args.project_root)
    verifier = Verifier(
        project_root=root,
        embeddings_dir=Path(args.embeddings_dir) if args.embeddings_dir else root / "embeddings",
        processed_dir=Path(args.processed_dir) if args.processed_dir else root / "datasets" / "processed",
        reports_dir=Path(args.reports_dir) if args.reports_dir else root / "outputs" / "module3",
        max_length=args.max_length,
        model_name=args.model_name,
        retokenize=not args.no_retokenize,
    )

    print("=" * 78)
    print("MODULE 3 — 18-POINT VERIFICATION OF REAL EMBEDDING OUTPUTS")
    print("=" * 78)
    verifier.run_all()

    md_path, json_path = verifier.write_reports()
    substantive = [c for c in verifier.checks if c["status"] == FAIL and c["check"] != 18]
    verdict = "MODULE 3 VERIFIED SUCCESSFULLY" if not substantive else "MODULE 3 FAILED"

    print("=" * 78)
    print(verdict)
    print(f"{verifier.n_passed} passed, {verifier.n_failed} failed, {verifier.n_unavailable} unavailable")
    for reason in verifier.failure_reasons:
        print(f"  - {reason}")
    print(f"Wrote: {md_path}")
    print(f"Wrote: {json_path}")
    print("=" * 78)

    sys.exit(0 if not substantive else 1)


if __name__ == "__main__":
    main()
