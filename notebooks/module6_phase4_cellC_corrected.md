# Phase 4 — corrected Cell C

## Root cause of the C.2 failure

Not a cell problem. A real bug in `scripts/m6_common.py`:

```python
def write_manifest(stage_path: Path, payload: dict[str, Any]) -> Path:
    payload = dict(payload)
    payload.setdefault("written_at", ...)
    tmp = stage_path / "manifest.tmp.json"      # <-- stage_path may not exist
    tmp.write_text(...)                          # <-- FileNotFoundError
```

`write_manifest()` never created its directory — unlike `write_json()` twelve lines below it, which does `path.parent.mkdir(parents=True, exist_ok=True)`.

It survived until now by luck. `stage_dir()` creates `outputs/module6/<stage>` but **nothing beneath it**, and every previous caller happened to target a `<dataset>/` subdirectory that some earlier step had already made. Your isolated ablation path was the first genuinely new nested directory anyone had ever passed, so it was the first to hit the bug.

**Fix — in the source, not the cell** (`scripts/m6_common.py`):

```python
    stage_path = Path(stage_path)
    stage_path.mkdir(parents=True, exist_ok=True)
```

This is the correct place: it fixes every caller — ablations, primary runs, and any future stage — rather than papering over one symptom. Two regression tests added.

---

## ⚠️ I also found a second defect in my own C.3, before you ran it

`C.3` as written was `--dataset fever --dry-run` with no `--run-name`. That is wrong for two reasons:

1. **It would have failed the assertion.** Order of operations in `m6_06` is: evidence guard (line 173) → `run_dirs` (195) → isolation refusal (201) → dry-run write (249). Since `models/module6/fever/latest.pt` exists, C.3 would have exited at line 201 — non-zero — and my harness expected exit 0.

2. **Worse, had it passed, it would have overwritten `outputs/module6/verifier_training/fever/manifest.json`** — the file that is the primary evidence of defect P1, which you instructed me to preserve. A dry-run still calls `write_manifest()`.

Corrected below: C.3 now runs under its own `--run-name`, so it can neither trip the isolation guard nor touch the audit-evidence manifest.

---

## Files changed since the last hash table

| File | Bytes | sha256 (first 16) | Status |
|---|---:|---|---|
| `scripts/m6_common.py` | 11,006 | `7ac8df3774ee0c1f` | **NEW — the fix** |
| `tests/test_verification_repair.py` | 11,766 | `4de8e7eed69c5681` | **updated — 2 new tests** |
| `scripts/m6_06_train_verifier.py` | 18,628 | `f7e97522e30e57e7` | unchanged |
| `scripts/m6_07_evaluate_verifier.py` | 10,868 | `5737cecd5aea11d3` | unchanged |
| `src/verification/dataset.py` | 10,533 | `2a9bb5ab781ea8c3` | unchanged |
| `src/verification/trainer.py` | 11,257 | `0fe9d8410333471a` | unchanged |

Full suite in my sandbox: **304 passed** (was 302 + 19 new − overlap; the 2 new manifest tests bring it to 304).

---

# CELL A′ — re-verify hashes after syncing the two changed files

```python
import hashlib
from pathlib import Path
ROOT = Path("/content/drive/MyDrive/Fake_News_Detection")

EXPECT = {
    "scripts/m6_common.py":               ("7ac8df3774ee0c1f", 11006),
    "scripts/m6_06_train_verifier.py":    ("f7e97522e30e57e7", 18628),
    "scripts/m6_07_evaluate_verifier.py": ("5737cecd5aea11d3", 10868),
    "src/verification/dataset.py":        ("2a9bb5ab781ea8c3", 10533),
    "src/verification/trainer.py":        ("0fe9d8410333471a", 11257),
    "tests/test_verification_repair.py":  ("4de8e7eed69c5681", 11766),
}
ok = True
for rel, (want, nbytes) in EXPECT.items():
    p = ROOT / rel
    if not p.is_file():
        print(f"MISSING  {rel}"); ok = False; continue
    b = p.read_bytes(); got = hashlib.sha256(b).hexdigest()[:16]
    good = got == want; ok &= good
    print(f"{'OK     ' if good else 'STALE  '} {rel}")
    if not good:
        print(f"          want {want} / {nbytes}B   got {got} / {len(b)}B")
print("\nALL CURRENT:", ok)
assert ok, "Sync the changed files to Drive before continuing."
```

---

# CELL B′ — tests (CPU, ~1 min, writes nothing)

```python
!cd /content/drive/MyDrive/Fake_News_Detection && \
 FAKE_NEWS_PROJECT_ROOT=/content/drive/MyDrive/Fake_News_Detection \
 python -m pytest tests/test_verification_repair.py tests/test_verification.py \
                  tests/test_retrieval.py -q
```

**Expect `97 passed`** (was 95; +2 manifest tests).

---

# CELL C — CORRECTED (CPU, ~2 min)

```python
import os, subprocess

ROOT = "/content/drive/MyDrive/Fake_News_Detection"
env  = {**os.environ, "FAKE_NEWS_PROJECT_ROOT": ROOT}
results = {}

def run(tag, args, expect_fail, must_contain=()):
    print(f"\n{'='*70}\n{tag}\n$ python {' '.join(args)}")
    r = subprocess.run(["python"] + args, cwd=ROOT, env=env,
                       capture_output=True, text=True)
    out = r.stdout + r.stderr
    for ln in out.strip().splitlines()[-16:]:
        print("   ", ln)
    okexit = bool(r.returncode) == expect_fail
    okmsg  = all(s in out for s in must_contain)
    if not okexit:
        print(f"    !! EXPECTED {'non-zero' if expect_fail else 'exit 0'}, "
              f"GOT {r.returncode}")
    for s in must_contain:
        if s not in out:
            print(f"    !! MISSING EXPECTED TEXT: {s!r}")
    results[tag] = okexit and okmsg
    print(f"    -> {'PASS' if results[tag] else 'FAIL'}")

# C.1  X-FACT has no train retrieval yet -> the guard MUST refuse.
run("C.1  guard refuses silent claim-only",
    ["scripts/m6_06_train_verifier.py", "--dataset", "x_fact", "--dry-run"],
    expect_fail=True,
    must_contain=("REFUSING TO TRAIN", "no training evidence for 'x_fact'"))

# C.2  ... but an EXPLICIT claim-only ablation is allowed, in its own directory.
run("C.2  explicit claim-only allowed",
    ["scripts/m6_06_train_verifier.py", "--dataset", "x_fact", "--claim-only",
     "--run-name", "guardcheck_claimonly", "--dry-run"],
    expect_fail=False)

# C.3  FEVER train retrieval now exists -> guard passes.
#      NOTE: --run-name is REQUIRED here. Without it this hits the isolation
#      guard (models/module6/fever/latest.pt exists) AND would overwrite
#      outputs/module6/verifier_training/fever/manifest.json, which is the
#      audit evidence for defect P1.
run("C.3  FEVER evidence is wired into training",
    ["scripts/m6_06_train_verifier.py", "--dataset", "fever",
     "--run-name", "guardcheck_fever_evidence", "--dry-run"],
    expect_fail=False,
    must_contain=("evidence source (train)     : dense_candidates",))

# C.4  Primary FEVER dir holds latest.pt -> isolation MUST refuse.
run("C.4  primary checkpoint is protected",
    ["scripts/m6_06_train_verifier.py", "--dataset", "fever"],
    expect_fail=True,
    must_contain=("REFUSING TO START", "already contains latest.pt"))

print(f"\n{'='*70}\nSUMMARY")
for k, v in results.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
print("\nALL CHECKS PASSED:", all(results.values()))
```

## Expected output

| Check | Expected | Proves |
|---|---|---|
| **C.1** | non-zero, `REFUSING TO TRAIN: no training evidence for 'x_fact'` | The original defect is now unreachable. X-FACT train retrieval genuinely does not exist yet — that is the next step. |
| **C.2** | exit 0. Prints `evidence source (train)     : none`, then `train label distribution: {...}`. Manifest records `"evidence_source": "claim_only"`, **not** `"none"` and **not** `"retrieved"`. | Deliberate claim-only works and is labelled honestly. Confirms the `write_manifest` fix. |
| **C.3** | exit 0. Prints `evidence source (train)     : dense_candidates` and `train with evidence: NN,NNN/50,000 (NN.N%)`. | Phase 3 retrieval is wired into training. **Read that percentage** — see below. |
| **C.4** | non-zero, `REFUSING TO START: ... already contains latest.pt` | Your existing FEVER checkpoint cannot be silently overwritten or resumed into. |

### On C.3's coverage percentage

FEVER train `recall@5` was **0.2619**, but coverage here is a different quantity: recall asks whether the *gold* evidence was retrieved, coverage asks whether *any* candidate resolved to text. Expect coverage near **100 %** — nearly every claim gets 5 candidates whether or not they are the right ones.

- **~100 %** → correct, proceed.
- **Well below 100 %** → retrieved `evidence_id`s are not resolving against `evidence.jsonl`. Stop and report; do not train.
- **0 %** → the guard's second condition fires instead and C.3 exits non-zero.

### Files these checks write

Two throwaway manifests, both in isolated directories, touching nothing existing:

```
outputs/module6/verifier_training/ablations/x_fact/guardcheck_claimonly/manifest.json
outputs/module6/verifier_training/ablations/fever/guardcheck_fever_evidence/manifest.json
```

C.1 and C.4 exit before writing. Optional cleanup once the checks pass:

```python
import shutil
for d in ["outputs/module6/verifier_training/ablations/x_fact/guardcheck_claimonly",
          "outputs/module6/verifier_training/ablations/fever/guardcheck_fever_evidence"]:
    shutil.rmtree(f"/content/drive/MyDrive/Fake_News_Detection/{d}", ignore_errors=True)
    print("removed", d)
```

Leaving them is harmless — the real ablations use different `--run-name` values.

---

**Nothing proceeds past this cell until all four checks read PASS.** No X-FACT retrieval, no training, and no move of `models/module6/{fever,x_fact}` — that last one still waits on your explicit authorisation.
