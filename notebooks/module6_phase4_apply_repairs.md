# Phase 4 — Apply the source repairs, then verify. **No training in this phase.**

Five files changed in your connected folder. All 302 existing tests still pass, plus 19 new ones.

| File | Bytes | sha256 (first 16) | Change |
|---|---:|---|---|
| `src/verification/dataset.py` | 10,533 | `2a9bb5ab781ea8c3` | honest `evidence_source`; evidence-language filter |
| `src/verification/trainer.py` | 11,257 | `0fe9d8410333471a` | `weighted_f1` in `classification_report` |
| `scripts/m6_06_train_verifier.py` | 18,628 | `f7e97522e30e57e7` | **hard guard**, **run isolation**, `--no-resume`, `--class-weights`, full manifest |
| `scripts/m6_07_evaluate_verifier.py` | 10,868 | `5737cecd5aea11d3` | `--run-name`, records `evidence_used`, majority baseline, reproducibility block |
| `tests/test_verification_repair.py` | 10,596 | `d2b1a0abadb4e8aa` | **new** — 19 regression tests |

**Nothing else was touched.** No config values changed (`train_subset_size` remains **50000**). No checkpoint, no evaluation artifact, no Module 5 file was read for writing.

---

## 1. The hard guard (your instruction 1)

New function in `m6_06_train_verifier.py`, called immediately after the examples are built and **before** any model is constructed:

```python
def assert_evidence_present(source, stats, claim_only, dataset):
    if claim_only:
        return
    if source == "none":
        raise SystemExit(
            f"REFUSING TO TRAIN: no training evidence for {dataset!r}. ..."
            f"  Fix : python scripts/m6_04_retrieve.py --dataset {dataset} --split train\n"
            f"  Or  : pass --claim-only if a claim-only ablation is what you intended.")
    if stats.get("n_with_evidence", 0) == 0:
        raise SystemExit(
            f"REFUSING TO TRAIN: {dataset!r} loaded candidates from {source!r}, but "
            f"ZERO of {stats.get('n_built', 0):,} examples ended up with any evidence text. ...")
```

Two conditions, not one. The second matters: candidate files could load fine while every `evidence_id` fails to resolve against `evidence.jsonl` (or a language filter removes everything). That would have produced the identical silent claim-only run with `evidence_source` reading `"dense_candidates"`.

A claim-only run is now reachable **only** by asking for it explicitly, and it is recorded as `"claim_only"`.

### The reporting fix that makes the guard trustworthy

`build_examples()` previously hard-coded the manifest field:

```python
"evidence_source": "claim_only" if claim_only else ("gold" if use_gold_evidence else "retrieved")
```

That is what wrote `"evidence_source": "retrieved"` into the FEVER manifest on a run where all 260,032 examples had zero evidence. It is now **derived from what was actually attached**, at the end of the function:

```python
if claim_only:                          stats["evidence_source"] = "claim_only"
elif stats["n_with_evidence"] == 0:     stats["evidence_source"] = "none"
elif use_gold_evidence:                 stats["evidence_source"] = "gold"
else:                                   stats["evidence_source"] = "retrieved"
```

The caller's intent is preserved separately as `requested_mode`, so you can still see what was asked for versus what happened. New fields: `n_with_evidence`, `n_evidence_attached`, `mean_evidence_per_example`, `evidence_coverage`, `by_evidence_language`, `n_dropped_by_language_filter`.

---

## 2. Run isolation and no-resume (your instruction 2)

```python
def run_dirs(cfg, root, dataset, run_name):
    if run_name:
        return (root / cfg["output"]["models_dir"] / "ablations" / dataset / run_name,
                stage_dir(cfg, "verifier_training", root) / "ablations" / dataset / run_name)
    return (root / cfg["output"]["models_dir"] / dataset,
            stage_dir(cfg, "verifier_training", root) / dataset)
```

Plus a refusal that fires **before** the model is built:

```python
existing = (ckpt_dir / "latest.pt").is_file()
if existing and not args.resume:
    raise SystemExit("REFUSING TO START: ... already contains latest.pt ...")
resume = bool(args.resume) and not args.no_resume
```

`resume` is no longer hard-coded to `True`. It is now opt-in, and resuming into a populated directory requires `--resume` explicitly. Without this, launching `--claim-only` after a finished run would restore `state.epoch = 3`, make `range(3, 3)` empty so the training loop never executes, overwrite `best.pt`, and return the *previous* run's `best_metric` as the ablation result — exit code 0, no warning.

**Directory layout after the ablations:**

```
models/module6/x_fact/                                  primary (unweighted claim+evidence)
models/module6/ablations/x_fact/claim_only/
models/module6/ablations/x_fact/oracle/
models/module6/ablations/x_fact/class_weighted/
outputs/module6/verifier_training/ablations/x_fact/<variant>/
outputs/module6/evaluation/ablations/x_fact_test/<variant>/
```

---

## 3. Also fixed while in there

| Item | Change |
|---|---|
| `--init-from` used `latest.pt` | Now prefers **`best.pt`**, falls back to `latest.pt`, and records `init_checkpoint` (path, bytes, source epoch/step/best_metric) in the manifest. Transfer will be provable from the manifest from now on, not inferred from tensors. |
| Stage 7 did not record its evidence source | Now writes `evidence_used` per split: `candidate_file`, `attached_source`, `n_with_evidence`, `mean_evidence_per_example`. This is the field whose absence made the reranker question unanswerable. |
| No baseline in the metrics file | Stage 7 now computes the **majority-class baseline on the same test items** and stores accuracy / macro-F1 / weighted-F1 beside the model's. |
| Only macro-F1 | `weighted_f1` added to `classification_report` — reported *alongside* macro-F1, never instead of it. |
| Unrecorded config lies | Manifest now records `gradient_accumulation_steps_declared_but_unused` and `mixed_precision_declared_but_unused`, so the paper cannot accidentally claim effective batch 32 or AMP. |
| `--class-weights` | Inverse-frequency weights `w_c = N / (K · n_c)`, computed on **train only**, recorded as `class_weight_values` / `class_weight_counts`. |
| `--evidence-languages` | Filters **before** the `max_evidence` slice, so you get top-k *of the permitted language* rather than top-k overall with holes. |

---

## 4. Test results (already run in my sandbox, no GPU needed)

```
tests/test_verification_repair.py ........ 19 passed
tests/  (full suite)  .................... 302 passed, 1 warning
```

The 19 new tests each map to an audit defect. The two that matter most:

- `test_evidence_source_is_none_when_nothing_attached` — fails if P3 ever returns.
- `test_run_dirs_isolate_every_variant` — asserts all five checkpoint directories are distinct and that no ablation can land on the primary.

---

# CELL A — Sync the five files to Drive, then verify by hash

> The five files are in your connected folder. Sync them to Drive by whatever method you normally use, then run this cell. **It writes nothing.**

```python
import hashlib, json
from pathlib import Path
ROOT = Path("/content/drive/MyDrive/Fake_News_Detection")

EXPECT = {
    "src/verification/dataset.py":            ("2a9bb5ab781ea8c3", 10533),
    "src/verification/trainer.py":            ("0fe9d8410333471a", 11257),
    "scripts/m6_06_train_verifier.py":        ("f7e97522e30e57e7", 18628),
    "scripts/m6_07_evaluate_verifier.py":     ("5737cecd5aea11d3", 10868),
    "tests/test_verification_repair.py":      ("d2b1a0abadb4e8aa", 10596),
}
ok = True
for rel, (want_sha, want_bytes) in EXPECT.items():
    p = ROOT / rel
    if not p.is_file():
        print(f"MISSING   {rel}"); ok = False; continue
    b = p.read_bytes()
    got = hashlib.sha256(b).hexdigest()[:16]
    good = (got == want_sha)
    ok &= good
    print(f"{'OK      ' if good else 'STALE   '} {rel}")
    if not good:
        print(f"           want sha={want_sha} bytes={want_bytes}")
        print(f"           got  sha={got} bytes={len(b)}")

print("\nALL FILES CURRENT:", ok)
if not ok:
    print("STOP. Drive still has old copies. Do not run anything else until this is OK.")
```

Line-ending note: if you sync through a tool that rewrites CRLF, the hashes will differ while the code is fine. If a file shows `STALE` but the byte count is close, tell me and I'll give you an AST-level check instead of a hash.

---

# CELL B — Run the test suite on Drive (CPU, ~1 min, writes nothing)

```python
!cd /content/drive/MyDrive/Fake_News_Detection && \
 FAKE_NEWS_PROJECT_ROOT=/content/drive/MyDrive/Fake_News_Detection \
 python -m pytest tests/test_verification_repair.py tests/test_verification.py \
                  tests/test_retrieval.py -q
```

**Expect:** `95 passed`. Anything else — stop and paste the output.

---

# CELL C — Prove the guard fires and the isolation holds (CPU, ~2 min, writes nothing)

> Uses `--dry-run`, which builds examples and writes a manifest but **constructs no model and trains nothing**. Read the notes on writes below.

```python
import subprocess, os
ROOT = "/content/drive/MyDrive/Fake_News_Detection"
env = {**os.environ, "FAKE_NEWS_PROJECT_ROOT": ROOT}

def run(args, expect_fail=False):
    print("\n$", " ".join(args))
    r = subprocess.run(["python"] + args, cwd=ROOT, env=env,
                       capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip().splitlines()
    for ln in out[-14:]:
        print("   ", ln)
    status = "FAILED (expected)" if r.returncode else "exit 0"
    print(f"    -> {status}")
    assert bool(r.returncode) == expect_fail, "UNEXPECTED RESULT - stop and report"

# C.1  X-FACT has no train retrieval yet -> the guard MUST refuse.
run(["scripts/m6_06_train_verifier.py", "--dataset", "x_fact", "--dry-run"],
    expect_fail=True)

# C.2  ... but an explicit claim-only ablation is allowed through.
run(["scripts/m6_06_train_verifier.py", "--dataset", "x_fact", "--claim-only",
     "--run-name", "guardcheck_claimonly", "--dry-run"], expect_fail=False)

# C.3  FEVER train retrieval now exists -> the guard MUST pass.
run(["scripts/m6_06_train_verifier.py", "--dataset", "fever", "--dry-run"],
    expect_fail=False)

# C.4  The primary FEVER dir already holds latest.pt -> isolation MUST refuse.
#      (dry-run returns before the check, so this one runs for real but is
#       stopped by the guard before any model is built.)
run(["scripts/m6_06_train_verifier.py", "--dataset", "fever"], expect_fail=True)
```

**Expected results:**

| Check | Expected | Proves |
|---|---|---|
| C.1 | `SystemExit: REFUSING TO TRAIN: no training evidence for 'x_fact'` | The exact defect is now impossible. X-FACT train retrieval genuinely does not exist yet. |
| C.2 | `exit 0`, prints `evidence source (train): none`, manifest records `"evidence_source": "claim_only"` | Deliberate claim-only still works and is labelled honestly. |
| C.3 | `exit 0`, prints `evidence source (train): dense_candidates` and `train with evidence: NN,NNN/50,000` | The Phase 3 repair is wired into training. |
| C.4 | `SystemExit: REFUSING TO START: ... already contains latest.pt` | Your existing FEVER checkpoint cannot be silently overwritten. |

### ⚠️ What C.2 writes

`--dry-run` calls `write_manifest()`. C.2 therefore creates one small JSON:

```
outputs/module6/verifier_training/ablations/x_fact/guardcheck_claimonly/manifest.json
```

It is a throwaway diagnostic in an isolated directory, touching no existing artifact. Delete it or leave it — it will be overwritten when the real `claim_only` ablation runs under its own name. C.1, C.3 and C.4 all exit before writing anything.

If you would rather have **zero** writes, skip C.2; C.1, C.3 and C.4 are the load-bearing checks.

---

# What comes next (not started — awaiting your go-ahead)

Still **no training**. After you confirm Cells A–C:

| # | Step | Device | Est. |
|---|---|---|---|
| 1 | X-FACT train retrieval (`m6_02`/`m6_03`/`m6_04`, 1,096 claims) | GPU+CPU | ~5 min |
| 2 | Back up the existing X-FACT evaluation, re-run Stage 7 on the dense fallback | GPU | ~3 min |
| 3 | Move `models/module6/{fever,x_fact}` aside as `*.CLAIMONLY_PREREPAIR` — **only on your explicit say-so** | — | — |
| 4 | Retrain FEVER evidence-aware (50,000 subset) | GPU | ~25–40 min |
| 5 | Retrain X-FACT evidence-aware, `--init-from models/module6/fever` | GPU | ~10–15 min |
| 6 | Stage 7 on both | GPU | ~10 min |
| 7 | Four X-FACT ablations: `claim_only`, `claim_evidence`, `oracle`, `class_weighted` | GPU | ~60 min |
| 8 | Collect results, statistics, final audit report | CPU | ~10 min |

## ⚠️ One decision I need from you at step 7

Phase 0 established that **X-FACT multilingual evidence is meaningful** (minority language ≥ 5 %). But your approved arm list has four entries — `claim_only`, `claim+evidence`, `oracle`, `class_weighted` — and does **not** include the English-only vs multilingual comparison.

I have implemented `--evidence-languages` so the arm is available, but I will **not** run it unless you say so. Adding it costs ~30 min GPU (two more arms: `evidence_en` and `evidence_multi`) and would let you report whether Tamil-language evidence actually helps Tamil claims — which is close to the centre of your research claim. Your call; I will not assume.

Also per your instruction 6: **I will not move or overwrite `models/module6/fever` or `models/module6/x_fact` until you explicitly authorise step 3.** Until then the isolation guard will keep refusing to write there, which is the correct behaviour.
