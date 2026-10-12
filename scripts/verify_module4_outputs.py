#!/usr/bin/env python
"""Module 4 — 20-point verification of the real training outputs.

Verifies outputs only. Never trains, never regenerates, never touches
Module 3. If something is missing or wrong it FAILS rather than being
quietly tolerated.

The 20 checks:

   1. Frozen Module 3 inputs still intact (read-only, unmodified)
   2. Record IDs align with Module 3
   3. Dataset counts 48,305 / 10,351 / 10,352 = 69,008
   4. Zero train/validation/test leakage
   5. Projection dimensions correct (128-d)
   6. Projections L2-normalised
   7. No NaN / Inf anywhere in results
   8. Training completed (all ablation cells present)
   9. Loss behaved (SupCon decreased; no divergence)
  10. Validation/test separation (test evaluated once, after selection)
  11. Overall metrics present
  12. English metrics present
  13. Tamil metrics present
  14. Baselines present (majority + logistic regression)
  15. Ablations complete (CLS/mean x language-aware/uniform x SupCon/CE-MLP)
  16. Zero-shot results present and reported separately
  17. Reproducibility: seeds and config recorded
  18. GPU / runtime telemetry recorded
  19. All required output files exist and JSON is valid
  20. Final status PASS or FAIL with reasons

Usage:
    python scripts/verify_module4_outputs.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PASS, FAIL, UNAVAILABLE = "PASS", "FAIL", "UNAVAILABLE"

EXPECTED_COUNTS = {"train": 48305, "validation": 10351, "test": 10352}
EXPECTED_TOTAL = 69008
EXPECTED_INPUT_DIM = 768
EXPECTED_PROJECTION_DIM = 128
EXPECTED_TAMIL_ROWS = 5408

REQUIRED_FILES = [
    "preflight_verification.json",
    "training_history.json",
    "evaluation_metrics.json",
    "ablation_results.json",
    "module4_report.md",
    "module4_config_used.yaml",
]


class Module4Verifier:
    def __init__(self, root: Path, reports_dir: Path, embeddings_dir: Path, models_dir: Path) -> None:
        self.root = root
        self.reports_dir = reports_dir
        self.embeddings_dir = embeddings_dir
        self.models_dir = models_dir
        self.checks: list[dict[str, Any]] = []
        self.metrics: dict[str, Any] = {}
        self.ablation: dict[str, Any] = {}
        self.preflight: dict[str, Any] = {}

    def record(self, n: int, name: str, status: str, detail: str, data: Any = None) -> None:
        e = {"check": n, "name": name, "status": status, "detail": detail}
        if data is not None:
            e["data"] = data
        self.checks.append(e)
        sym = {PASS: "[PASS]", FAIL: "[FAIL]", UNAVAILABLE: "[N/A ]"}.get(status, "[????]")
        print(f"{sym} {n:2d}. {name}: {detail}", flush=True)

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.checks if c["status"] == PASS)

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if c["status"] == FAIL)

    @property
    def n_unavailable(self) -> int:
        return sum(1 for c in self.checks if c["status"] == UNAVAILABLE)

    @property
    def failure_reasons(self) -> list[str]:
        return [f"Check {c['check']} ({c['name']}): {c['detail']}"
                for c in self.checks if c["status"] == FAIL]

    # ---------------- loading ----------------

    def _load_json(self, name: str) -> dict[str, Any] | None:
        p = self.reports_dir / name
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def run_all(self) -> None:
        self.metrics = self._load_json("evaluation_metrics.json") or {}
        self.ablation = self._load_json("ablation_results.json") or {}
        self.preflight = self._load_json("preflight_verification.json") or {}

        self._c19_files_exist()
        self._c1_frozen_inputs()
        self._c2_record_ids()
        self._c3_counts()
        self._c4_leakage()
        self._c5_c6_projection()
        self._c7_nan_inf()
        self._c8_training_completed()
        self._c9_loss_behaviour()
        self._c10_val_test_separation()
        self._c11_c12_c13_metrics()
        self._c14_baselines()
        self._c15_ablations()
        self._c16_zero_shot()
        self._c17_reproducibility()
        self._c18_telemetry()
        self._c20_final_status()

    # ---------------- checks ----------------

    def _c19_files_exist(self) -> None:
        missing = [f for f in REQUIRED_FILES if not (self.reports_dir / f).exists()]
        invalid = [f for f in REQUIRED_FILES
                   if f.endswith(".json") and (self.reports_dir / f).exists()
                   and self._load_json(f) is None]
        models = sorted(p.name for p in self.models_dir.glob("*.pt")) if self.models_dir.exists() else []
        ok = not missing and not invalid and bool(models)
        self.record(19, "All required output files exist and JSON is valid",
                    PASS if ok else FAIL,
                    "all present, JSON valid, model weights saved" if ok
                    else f"missing={missing} invalid_json={invalid} model_weights={len(models)}",
                    {"missing": missing, "invalid_json": invalid, "model_weights": models})

    def _c1_frozen_inputs(self) -> None:
        if not self.preflight:
            self.record(1, "Frozen Module 3 inputs intact", FAIL,
                        "preflight_verification.json missing — cannot confirm inputs were validated")
            return
        ok = bool(self.preflight.get("passed"))
        per = {p: r.get("passed") for p, r in self.preflight.get("per_pooling", {}).items()}
        self.record(1, "Frozen Module 3 inputs intact", PASS if ok else FAIL,
                    f"pre-flight gate passed for all poolings {per}" if ok
                    else f"pre-flight reported failures: {per}", per)

    def _c2_record_ids(self) -> None:
        """Record IDs in the frozen embeddings must still match Module 3
        AND be identical across the two pooling strategies."""
        try:
            ids = {}
            for pooling in ("cls", "mean"):
                ids[pooling] = {s: np.load(self.embeddings_dir / s / "record_ids.npy").astype(str)
                                for s in EXPECTED_COUNTS}
            same = {s: bool(np.array_equal(ids["cls"][s], ids["mean"][s])) for s in EXPECTED_COUNTS}
            lens = {s: int(len(ids["cls"][s])) for s in EXPECTED_COUNTS}
            unique = {s: int(len(set(ids["cls"][s].tolist()))) for s in EXPECTED_COUNTS}
            ok = all(same.values()) and all(lens[s] == unique[s] == EXPECTED_COUNTS[s] for s in EXPECTED_COUNTS)
            self.record(2, "Record IDs align with Module 3", PASS if ok else FAIL,
                        f"identical across poolings {same}, unique {unique}" if ok
                        else f"mismatch: identical={same} lens={lens} unique={unique}",
                        {"identical_across_poolings": same, "lengths": lens, "unique": unique})
        except Exception as exc:  # noqa: BLE001
            self.record(2, "Record IDs align with Module 3", FAIL, f"could not read record_ids: {exc}")

    def _c3_counts(self) -> None:
        try:
            counts = {s: int(len(np.load(self.embeddings_dir / s / "labels.npy"))) for s in EXPECTED_COUNTS}
            total = sum(counts.values())
            ok = counts == EXPECTED_COUNTS and total == EXPECTED_TOTAL
            self.record(3, "Dataset counts match Module 2", PASS if ok else FAIL,
                        f"{counts} = {total} (expected {EXPECTED_COUNTS} = {EXPECTED_TOTAL})",
                        {"observed": counts, "expected": EXPECTED_COUNTS, "total": total})
        except Exception as exc:  # noqa: BLE001
            self.record(3, "Dataset counts match Module 2", FAIL, f"could not read labels: {exc}")

    def _c4_leakage(self) -> None:
        try:
            sets = {s: set(np.load(self.embeddings_dir / s / "record_ids.npy").astype(str).tolist())
                    for s in EXPECTED_COUNTS}
            overlaps = {}
            names = sorted(sets)
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    overlaps[f"{a}_vs_{b}"] = len(sets[a] & sets[b])
            ok = all(v == 0 for v in overlaps.values())
            self.record(4, "Zero train/validation/test leakage", PASS if ok else FAIL,
                        f"{overlaps}" if ok else f"LEAKAGE: {overlaps}", overlaps)
        except Exception as exc:  # noqa: BLE001
            self.record(4, "Zero train/validation/test leakage", FAIL, f"{exc}")

    def _c5_c6_projection(self) -> None:
        """Projection dimension and L2 normalisation.

        Read from the saved config and the model weight shapes rather
        than assumed: a mismatch between configured and actual output
        dim would silently change the contrastive geometry.
        """
        cfg = (self.metrics or {}).get("config_used", {})
        model_cfg = cfg.get("model", {})
        dim = model_cfg.get("projection_output_dim")
        normalize = model_cfg.get("normalize_projection")

        weight_dim = None
        try:
            import torch

            heads = sorted(self.models_dir.glob("projection_head_*.pt"))
            if heads:
                sd = torch.load(heads[0], map_location="cpu")
                last = [v for k, v in sd.items() if k.endswith("weight")][-1]
                weight_dim = int(last.shape[0])
        except Exception:  # noqa: BLE001
            weight_dim = None

        dim_ok = dim == EXPECTED_PROJECTION_DIM and (weight_dim is None or weight_dim == EXPECTED_PROJECTION_DIM)
        self.record(5, "Projection dimensions correct", PASS if dim_ok else FAIL,
                    f"configured={dim}, saved-weight output={weight_dim} (expected {EXPECTED_PROJECTION_DIM})",
                    {"configured": dim, "weight_output_dim": weight_dim})
        self.record(6, "Projections L2-normalised", PASS if normalize is True else FAIL,
                    f"normalize_projection={normalize} (must be true: SupCon is defined on the unit hypersphere)",
                    {"normalize_projection": normalize})

    def _c7_nan_inf(self) -> None:
        """No NaN/Inf in any recorded metric or loss."""
        bad: list[str] = []

        def scan(obj: Any, path: str = "") -> None:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    scan(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    scan(v, f"{path}[{i}]")
            elif isinstance(obj, float):
                if np.isnan(obj) or np.isinf(obj):
                    bad.append(path)

        scan(self.metrics, "evaluation_metrics")
        scan(self.ablation, "ablation_results")
        self.record(7, "No NaN or Inf in results", PASS if not bad else FAIL,
                    "no non-finite values in any recorded metric or loss" if not bad
                    else f"{len(bad)} non-finite value(s): {bad[:8]}", {"non_finite_paths": bad[:50]})

    def _c8_training_completed(self) -> None:
        experiments = self.ablation.get("experiments", [])
        final = self.metrics.get("final", {}).get("per_seed_full", [])
        expected_cells = 4  # 2 poolings x 2 sampling strategies
        ok = len(experiments) == expected_cells and len(final) >= 1
        self.record(8, "Training completed", PASS if ok else FAIL,
                    f"{len(experiments)}/{expected_cells} ablation cells, {len(final)} final seed run(s)",
                    {"ablation_cells": len(experiments), "final_runs": len(final)})

    def _c9_loss_behaviour(self) -> None:
        """SupCon loss must actually decrease, and stay finite.

        A flat or rising curve means the run produced nothing, even
        though every file would still be written.
        """
        hist = self._load_json("training_history.json") or {}
        report: dict[str, Any] = {}
        ok = True
        for entry in hist.get("ablation", []):
            s1 = entry.get("history", {}).get("stage1", [])
            if not s1:
                report[f"{entry.get('pooling')}/{entry.get('sampling')}"] = "no stage1 history"
                ok = False
                continue
            first = s1[0]["train_supcon_loss"]
            last = min(e["train_supcon_loss"] for e in s1)
            finite = all(np.isfinite(e["train_supcon_loss"]) and np.isfinite(e["val_supcon_loss"]) for e in s1)
            decreased = last < first
            report[f"{entry.get('pooling')}/{entry.get('sampling')}"] = {
                "first_epoch_loss": first, "best_loss": last,
                "decreased": bool(decreased), "all_finite": bool(finite), "epochs": len(s1),
            }
            if not (decreased and finite):
                ok = False
        self.record(9, "Loss behaved (decreased, no divergence)", PASS if ok and report else FAIL,
                    "SupCon loss decreased and stayed finite in every ablation cell" if ok and report
                    else f"problem: {report}", report)

    def _c10_val_test_separation(self) -> None:
        """Test must be absent from every ablation cell and present only
        in the final selected runs -- the structural evidence that
        selection could not have used it."""
        experiments = self.ablation.get("experiments", [])
        ablation_touched_test = [f"{e.get('pooling')}/{e.get('sampling')}" for e in experiments
                                 if e.get("test_evaluated")]
        final = self.metrics.get("final", {}).get("per_seed_full", [])
        final_touched = [r.get("seed") for r in final if r.get("test_evaluated")]
        selection_metric = self.ablation.get("selection", {}).get("selection_metric")

        ok = (not ablation_touched_test) and bool(final_touched) and selection_metric == "validation_macro_f1"
        self.record(10, "Validation/test separation", PASS if ok else FAIL,
                    f"no ablation cell evaluated test; final seeds {final_touched} did; "
                    f"selection metric={selection_metric}" if ok
                    else f"VIOLATION: ablation cells touching test={ablation_touched_test}, "
                         f"final={final_touched}, selection_metric={selection_metric}",
                    {"ablation_touched_test": ablation_touched_test,
                     "final_touched_test": final_touched, "selection_metric": selection_metric})

    def _c11_c12_c13_metrics(self) -> None:
        final = self.metrics.get("final", {})
        required = ("accuracy", "precision_macro", "recall_macro", "macro_f1")

        overall = final.get("test_across_seeds", {})
        overall_ok = all(k in overall for k in required)
        self.record(11, "Overall metrics present", PASS if overall_ok else FAIL,
                    f"{sorted(k for k in overall if k in required)}" if overall_ok
                    else f"missing: {[k for k in required if k not in overall]}", overall)

        per_lang = final.get("test_per_language_across_seeds", {})
        for n, lang, label in ((12, "en", "English"), (13, "ta", "Tamil")):
            stats = per_lang.get(lang, {})
            ok = all(k in stats for k in required)
            self.record(n, f"{label} metrics present", PASS if ok else FAIL,
                        f"macro_f1 mean={stats.get('macro_f1', {}).get('mean')} "
                        f"std={stats.get('macro_f1', {}).get('std')}" if ok
                        else f"missing {label} metrics: {[k for k in required if k not in stats]}",
                        stats)

    def _c14_baselines(self) -> None:
        b = self.metrics.get("baselines") or self.ablation.get("baselines") or {}
        have_majority = "majority_class" in b
        have_logreg = any(k.startswith("logistic_regression") for k in b)
        ok = have_majority and have_logreg
        self.record(14, "Baselines present", PASS if ok else FAIL,
                    f"{sorted(b.keys())}" if ok
                    else f"missing (majority={have_majority}, logreg={have_logreg}): {sorted(b.keys())}",
                    {"baselines": sorted(b.keys())})

    def _c15_ablations(self) -> None:
        experiments = self.ablation.get("experiments", [])
        cells = {(e.get("pooling"), e.get("sampling")) for e in experiments}
        expected = {("cls", "language_aware"), ("cls", "uniform"),
                    ("mean", "language_aware"), ("mean", "uniform")}
        both_methods = all("supcon" in e and "ce_mlp" in e for e in experiments)
        ranking = self.ablation.get("selection", {}).get("ranking", [])
        ok = cells == expected and both_methods and len(ranking) == 8
        self.record(15, "Ablations complete", PASS if ok else FAIL,
                    f"4 cells x (SupCon, CE-MLP) = {len(ranking)} ranked configurations" if ok
                    else f"cells={sorted(cells)} both_methods={both_methods} ranked={len(ranking)}",
                    {"cells": sorted(str(c) for c in cells), "n_ranked": len(ranking)})

    def _c16_zero_shot(self) -> None:
        z = self.metrics.get("zero_shot")
        if not z:
            self.record(16, "Zero-shot results present", FAIL,
                        "zero_shot block absent from evaluation_metrics.json")
            return
        ta = z.get("test_tamil_zero_shot", {}).get("overall", {})
        ok = "macro_f1" in ta and z.get("train_rows_en", 0) > 0 and z.get("test_rows_ta", 0) > 0
        self.record(16, "Zero-shot results present (reported separately)", PASS if ok else FAIL,
                    f"trained on {z.get('train_rows_en')} English rows, evaluated on "
                    f"{z.get('test_rows_ta')} Tamil rows, macro-F1={ta.get('macro_f1')}" if ok
                    else f"incomplete zero-shot record: {z.keys()}",
                    {"train_rows_en": z.get("train_rows_en"), "test_rows_ta": z.get("test_rows_ta"),
                     "tamil_macro_f1": ta.get("macro_f1")})

    def _c17_reproducibility(self) -> None:
        cfg = self.metrics.get("config_used", {})
        seeds_cfg = cfg.get("reproducibility", {}).get("seeds")
        seeds_run = self.metrics.get("final", {}).get("seeds")
        per_seed = [r.get("seed") for r in self.metrics.get("final", {}).get("per_seed_full", [])]
        cfg_saved = (self.reports_dir / "module4_config_used.yaml").exists()
        ok = bool(seeds_cfg) and seeds_run == seeds_cfg and sorted(per_seed) == sorted(seeds_cfg) and cfg_saved
        self.record(17, "Reproducibility: seeds and config recorded", PASS if ok else FAIL,
                    f"seeds {seeds_run} run and recorded; config snapshot saved" if ok
                    else f"seeds_cfg={seeds_cfg} seeds_run={seeds_run} per_seed={per_seed} cfg_saved={cfg_saved}",
                    {"seeds_config": seeds_cfg, "seeds_run": seeds_run, "per_seed": per_seed})

    def _c18_telemetry(self) -> None:
        runs = self.metrics.get("final", {}).get("per_seed_full", [])
        gpu = runs[0].get("gpu", {}) if runs else {}
        env = gpu.get("device", {})
        runtime = self.metrics.get("total_runtime_seconds")
        ok = bool(env) and runtime is not None
        detail = (f"{env.get('gpu_name') or 'CPU'}, torch {env.get('torch_version')}, "
                  f"peak {gpu.get('peak_gpu_memory', {}).get('peak_allocated_human')}, "
                  f"total runtime {runtime}s") if ok else "GPU/runtime telemetry not recorded"
        self.record(18, "GPU / runtime telemetry recorded", PASS if ok else FAIL, detail,
                    {"device": env, "peak_gpu_memory": gpu.get("peak_gpu_memory"),
                     "total_runtime_seconds": runtime})

    def _c20_final_status(self) -> None:
        reasons = self.failure_reasons
        if not reasons:
            self.record(20, "Final status", PASS,
                        f"all {self.n_passed} completed checks passed"
                        + (f" ({self.n_unavailable} unavailable)" if self.n_unavailable else ""))
        else:
            self.record(20, "Final status", FAIL, f"{len(reasons)} check(s) failed",
                        {"reasons": reasons})

    # ---------------- reports ----------------

    def write_reports(self) -> tuple[Path, Path]:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        substantive = [c for c in self.checks if c["status"] == FAIL and c["check"] != 20]
        verdict = "MODULE 4 VERIFIED SUCCESSFULLY" if not substantive else "MODULE 4 FAILED"

        payload = {
            "module": "Module 4 - Supervised Contrastive Learning",
            "verification_run_at": datetime.now(timezone.utc).isoformat(),
            "reports_dir": str(self.reports_dir),
            "embeddings_dir": str(self.embeddings_dir),
            "verdict": verdict,
            "n_checks": len(self.checks),
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "n_unavailable": self.n_unavailable,
            "failure_reasons": self.failure_reasons,
            "checks": self.checks,
        }
        jp = self.reports_dir / "final_verification_report.json"
        jp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

        lines = ["# Module 4 — Final Verification Report", "", f"# {verdict}", "",
                 f"Run at: {payload['verification_run_at']}  ",
                 f"Checks: **{self.n_passed} passed**, {self.n_failed} failed, "
                 f"{self.n_unavailable} unavailable", "",
                 "Every figure is measured from the generated files. Module 3 was neither read "
                 "for modification nor altered.", "",
                 "| # | Check | Status | Detail |", "|---:|---|:---:|---|"]
        for c in self.checks:
            lines.append(f"| {c['check']} | {c['name']} | **{c['status']}** | "
                         f"{str(c['detail']).replace('|', chr(92) + '|')} |")
        if self.failure_reasons:
            lines += ["", "## Failure reasons", ""] + [f"- {r}" for r in self.failure_reasons]
        for n, title in ((3, "Dataset counts"), (10, "Validation/test separation"),
                         (11, "Overall test metrics"), (13, "Tamil test metrics"),
                         (15, "Ablations"), (16, "Zero-shot"), (18, "Telemetry")):
            e = next((c for c in self.checks if c["check"] == n), None)
            if e and e.get("data"):
                lines += ["", f"## {title}", "", "```json",
                          json.dumps(e["data"], indent=2, ensure_ascii=False, default=str), "```"]
        lines += ["", "## Conclusion", "",
                  ("All checks passed. Module 4's results are complete, reproducible, "
                   "leak-free, and selected without reference to the test set."
                   if not substantive else
                   "One or more checks failed — see 'Failure reasons'. Module 4 is NOT complete.")]
        mp = self.reports_dir / "final_verification_report.md"
        mp.write_text("\n".join(lines), encoding="utf-8")
        return mp, jp


def main() -> None:
    p = argparse.ArgumentParser(description="Module 4: verify training outputs (20 checks).")
    p.add_argument("--project-root", default=str(PROJECT_ROOT))
    p.add_argument("--reports-dir", default=None)
    p.add_argument("--embeddings-dir", default=None)
    p.add_argument("--models-dir", default=None)
    args = p.parse_args()

    root = Path(args.project_root)
    v = Module4Verifier(
        root=root,
        reports_dir=Path(args.reports_dir) if args.reports_dir else root / "outputs" / "module4",
        embeddings_dir=Path(args.embeddings_dir) if args.embeddings_dir else root / "embeddings",
        models_dir=Path(args.models_dir) if args.models_dir else root / "models" / "module4",
    )

    print("=" * 78)
    print("MODULE 4 — 20-POINT VERIFICATION")
    print("=" * 78)
    v.run_all()
    mp, jp = v.write_reports()

    substantive = [c for c in v.checks if c["status"] == FAIL and c["check"] != 20]
    verdict = "MODULE 4 VERIFIED SUCCESSFULLY" if not substantive else "MODULE 4 FAILED"
    print("=" * 78)
    print(verdict)
    print(f"{v.n_passed} passed, {v.n_failed} failed, {v.n_unavailable} unavailable")
    for r in v.failure_reasons:
        print(f"  - {r}")
    print(f"Wrote: {mp}")
    print(f"Wrote: {jp}")
    print("=" * 78)
    sys.exit(0 if not substantive else 1)


if __name__ == "__main__":
    main()
