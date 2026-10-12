# Phase 3 — Generate the missing FEVER train retrieval artifacts

**Approved scope:** FEVER verifier training subset = **50,000** claims (`configs/module6_retrieval_verification.yaml` → `verifier.fever.train_subset_size: 50000`, unchanged).
**Source code changes required: NONE.** Every command below runs against the existing scripts as they are on Drive.

---

## Why no code change is needed

Three things I verified in the code rather than assumed:

1. **`m6_04_retrieve.py` already accepts `--split train`** — it is in the argparse `choices`, and it already enforces `build_evidence_scope(...) → assert_index_within_scope(...)` against `leakage.index_scope.train: [train]`. Train claims can retrieve **only** evidence cited by train claims. Your leakage requirement is satisfied by existing code.

2. **`m6_02_build_index.py` already derives a train index from existing embeddings.** It contains `FALLBACK_ORDER = {"train": ["train", "validation", "test"], ...}` and selects the permitted subset from a wider phase's vectors. Because train-scope evidence ⊂ test-scope evidence and the vectors are identical (same model, same text), **no evidence re-encoding is needed.** `assert_index_within_scope()` still audits the result. This removes ~1 h of LaBSE encoding.

3. **Retrieving top-10 for train is provably equivalent to top-50 for training purposes.** `build_examples()` does `list(ev_ids)[:max_evidence]` with `evidence_per_claim: 5`, so only the first 5 ranked candidates ever reach the tokenizer. The first 5 of a top-10 list and the first 5 of a top-50 list are the same 5 items. Retrieving top-10 changes nothing methodologically and cuts `dense_candidates.json` from ~938 MB to ~190 MB — which matters, because `m6_06` calls `json.loads()` on that whole file.

### ⚠️ One trade-off you are accepting

`m6_03` will encode **all 228,090** FEVER train claims, but `m6_06` trains on a stratified **50,000** subset. About 78 % of the encoding and retrieval work is discarded.

Avoiding that waste would need a `--claim-ids-file` flag on `m6_03`/`m6_04` plus a subset-exporter script (~80 lines, additive). You asked me not to modify source code unless necessary, and it is **not** necessary — so the cells below take the zero-change path. Cost of that choice: roughly **+50 minutes**, and one non-resumable 60–90 min CPU job (see the disconnect warning at Cell 4).

---

## Total resource budget

| Item | Value |
|---|---|
| GPU time | ~25–40 min (Cell 3 only) |
| CPU time | ~70–100 min (Cells 2 and 4) |
| Wall clock | **~1 h 45 m – 2 h 20 m** |
| New Drive storage | **~1.1 GB** |
| Existing files modified | **none** |
| Existing files deleted/renamed | **none** |

---

# CELL 1 — Preflight (READ-ONLY, ~30 s)

> **Writes nothing.** Confirms every prerequisite exists and that no train artifact exists yet, so nothing can be silently overwritten.

```python
import json, os, subprocess
from pathlib import Path

ROOT = Path("/content/drive/MyDrive/Fake_News_Detection")
os.environ["FAKE_NEWS_PROJECT_ROOT"] = str(ROOT)      # scripts/m6_common.py reads this
os.chdir(ROOT)
print("project root :", ROOT, "| exists:", ROOT.is_dir())

print("\n--- PREREQUISITES (must all be True) ---")
need = {
    "module5 claims.jsonl":   ROOT/"datasets/fact_verification/fever/claims.jsonl",
    "module5 evidence.jsonl": ROOT/"datasets/fact_verification/fever/evidence.jsonl",
    "module5 claim_splits":   ROOT/"datasets/fact_verification/fever/claim_splits.json",
    "encode_evidence test":   ROOT/"outputs/module6/encode_evidence/fever_test/manifest.json",
    "scripts/__init__.py":    ROOT/"scripts/__init__.py",
    "m6_02":                  ROOT/"scripts/m6_02_build_index.py",
    "m6_03":                  ROOT/"scripts/m6_03_encode_claims.py",
    "m6_04":                  ROOT/"scripts/m6_04_retrieve.py",
}
for k, p in need.items():
    print(f"  {k:<24} {p.is_file()}")

m = json.loads((ROOT/"outputs/module6/encode_evidence/fever_test/manifest.json").read_text())
print(f"\n  encode_evidence/fever_test status = {m.get('status')!r}  "
      f"n_items = {m.get('n_items')}   <-- must be 'complete'")

print("\n--- TARGETS (must all be False: nothing exists yet) ---")
for p in ["outputs/module6/build_index/fever_train",
          "outputs/module6/encode_claims/fever_train",
          "outputs/module6/retrieve/fever_train"]:
    print(f"  {p:<45} {(ROOT/p).exists()}")

print("\n--- RERANKER NEUTRALISATION (must stay as shown) ---")
rr = ROOT/"outputs/module6/rerank/x_fact_test"
print(f"  reranked_top5.json                 exists = {(rr/'reranked_top5.json').is_file()}  <-- must be False")
print(f"  reranked_top5.UNTRAINED_DO_NOT_USE exists = {(rr/'reranked_top5.UNTRAINED_DO_NOT_USE.json').is_file()}  <-- must be True")

print("\n--- CONFIRM TRAIN SUBSET SCOPE = 50,000 ---")
import yaml
cfg = yaml.safe_load((ROOT/"configs/module6_retrieval_verification.yaml").read_text())
print("  verifier.fever.train_subset_size =", cfg["verifier"]["fever"]["train_subset_size"])
print("  leakage.index_scope.train        =", cfg["leakage"]["index_scope"]["train"])
print("  retrieval.top_k_candidates       =", cfg["retrieval"]["top_k_candidates"])

print("\n--- MODULE 5 FINGERPRINT (baseline for the untouched check) ---")
import sys; sys.path.insert(0, str(ROOT))
from scripts.m6_common import module5_fingerprint
BASELINE = module5_fingerprint(ROOT)
print(json.dumps(BASELINE, indent=2)[:600], "...")
```

**Stop and report to me if:** any prerequisite is `False`, any target already `exists = True`, `reranked_top5.json` is `True`, or `train_subset_size` is not `50000`.

---

# CELL 2 — Build the FEVER train index (CPU, ~5–10 min)

> **Reads:** `datasets/fact_verification/fever/*` (Module 5, read-only) · `outputs/module6/encode_evidence/fever_test/*`
> **Writes:** `outputs/module6/build_index/fever_train/` **only**

```python
!cd /content/drive/MyDrive/Fake_News_Detection && \
 FAKE_NEWS_PROJECT_ROOT=/content/drive/MyDrive/Fake_News_Detection \
 python scripts/m6_02_build_index.py --dataset fever --phase train
```

### Files created

```
outputs/module6/build_index/fever_train/vectors.npy        ~190 MB   (N_train_evidence x 768 float32)
outputs/module6/build_index/fever_train/index_meta.json    ~4 MB     (evidence_ids, dim, normalized, n)
outputs/module6/build_index/fever_train/manifest.json      ~1 KB
```

### Expected output — all three lines must appear

```
derived from 'test' embeddings: selected N of 67,388 vectors for this phase's scope
indexed N evidence vectors (dim 768)
scope audit: PASS
```

`N` will be **less than 67,388** — that is correct and is the leakage control working: only evidence cited by *train* claims is indexed.

### What failure would mean

- `No completed evidence embeddings usable for fever/train` → `encode_evidence/fever_test` is not `complete`. Stop; do not run `m6_01`, report to me first.
- `Cannot derive the 'train' index from 'test' embeddings: K permitted evidence item(s) were never encoded` → the test-phase encoding is incomplete. Stop and report `K`.
- `assert_index_within_scope` raising → a genuine leakage bug. **Stop immediately and report.**

---

# CELL 3 — Encode FEVER train claims (GPU, ~25–40 min) 🔴 EXPENSIVE

> **Reads:** `datasets/fact_verification/fever/claims.jsonl`, `claim_splits.json` (read-only)
> **Writes:** `outputs/module6/encode_claims/fever_train/` **only**

**Confirm a GPU is attached first:**

```python
import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU - fix runtime type first")
```

```python
!cd /content/drive/MyDrive/Fake_News_Detection && \
 FAKE_NEWS_PROJECT_ROOT=/content/drive/MyDrive/Fake_News_Detection \
 python scripts/m6_03_encode_claims.py --dataset fever --split train
```

### Files created

```
outputs/module6/encode_claims/fever_train/shard_00000.npy ... shard_00022.npy   ~700 MB total
outputs/module6/encode_claims/fever_train/ids.json                              ~15 MB
outputs/module6/encode_claims/fever_train/manifest.json                         ~1 KB
```

23 shards at `shard_size: 10000` (228,090 claims). Encoder: `sentence-transformers/LaBSE`, `max_seq_length: 256`, `encode_batch_size: 64`.

### Expected output

```
claims to encode: 228,090
... (per-shard progress)
status: complete   n_items: 228090   n_shards: 23
```

### ✅ This step is resumable

Shards are written temp-then-rename and driven by the manifest. **If Colab disconnects, just re-run the identical command** — it resumes at the first incomplete shard. Nothing is lost and nothing is corrupted.

### What failure would mean

- `exit -9` / silent kill → RAM exhaustion. Re-run with `--batch-size 32 --shard-size 5000`. The memory-safety work in `src/retrieval/encoders.py` (micro-batching, `inference_mode`, `gc.collect`, thread limit) should prevent this, but the guard is there.
- `STALE src/retrieval/encoders.py` → the encoder file on Drive is out of date. Stop and report; do not work around it.

---

# CELL 4 — Retrieve FEVER train candidates (CPU, ~60–90 min) 🔴 EXPENSIVE + NOT RESUMABLE

> **Reads:** `build_index/fever_train/*`, `encode_claims/fever_train/*`, Module 5 corpus (read-only)
> **Writes:** `outputs/module6/retrieve/fever_train/` **only**

```python
!cd /content/drive/MyDrive/Fake_News_Detection && \
 FAKE_NEWS_PROJECT_ROOT=/content/drive/MyDrive/Fake_News_Detection \
 python scripts/m6_04_retrieve.py --dataset fever --split train --retriever dense --top-k 10
```

### Flag choices, and why

| Flag | Value | Reason |
|---|---|---|
| `--retriever` | `dense` | `load_retrieved()` in `m6_06` looks only for `reranked_top5.json` then `dense_candidates.json`. **BM25 train candidates would never be read**, and generating them costs an extra ~20–30 min. The BM25-vs-dense comparison is a *test-split* analysis and already exists. |
| `--top-k` | `10` | Only the first 5 reach the tokenizer (`evidence_per_claim: 5`), so 10 is methodologically identical to 50 while cutting the file from ~938 MB to ~190 MB. The extra 5 are cheap insurance in case any id fails to resolve in `evidence_texts`. |

### Files created

```
outputs/module6/retrieve/fever_train/dense_candidates.json   ~190 MB   {claim_id: [ev_id x 10]}
outputs/module6/retrieve/fever_train/dense_metrics.json      ~1 KB
outputs/module6/retrieve/fever_train/manifest.json           ~1 KB
```

### Expected output

```
dense  : {'recall@1': ..., 'recall@5': ..., 'recall@10': ..., 'mrr': ...}
         scored ~227,7xx, excluded (no gold) ~3xx
wrote -> outputs/module6/retrieve/fever_train
```

**Sanity check:** train `recall@5` should land in the same neighbourhood as the known test value **0.3227**. A train recall near 0 or near 1 means something is wrong — stop and report rather than proceeding to training.

The ~427 FEVER claims with no gold evidence (mostly `NOT ENOUGH INFO`) are excluded from recall denominators and counted in `accounting.n_excluded_no_gold_evidence`, not silently dropped. Only the train share of those appears here.

### ⚠️ Disconnect risk — read before starting

`m6_04` performs the whole search in memory and writes only at the end. **A disconnect at minute 70 loses all of it** (the GPU encoding from Cell 3 survives — only this CPU search is lost). Mitigations:

- Start it when you can leave the tab open and the machine awake.
- Free Colab idles out on an unfocused tab; keep the browser tab visible.
- If it dies repeatedly, tell me — that is the point at which the `--claim-ids-file` code change (restricting to the 50,000 subset, cutting this to ~15 min) becomes genuinely *necessary* rather than merely convenient, and I will implement it.

### Cosmetic note

Internally `m6_04` scores `tuple(k for k in (1, 5, 10, top_k))`. With `--top-k 10` that is `(1, 5, 10, 10)` — the `recall@10` key simply appears twice with the same value. Harmless. Train retrieval metrics are a sanity check, not a reported paper result.

---

# CELL 5 — Verify and confirm Module 5 untouched (READ-ONLY, ~1 min)

> **Writes nothing.** Run this before we go anywhere near training.

```python
import json
from pathlib import Path
ROOT = Path("/content/drive/MyDrive/Fake_News_Detection")

print("--- ARTIFACTS CREATED ---")
for p in ["outputs/module6/build_index/fever_train/manifest.json",
          "outputs/module6/encode_claims/fever_train/manifest.json",
          "outputs/module6/retrieve/fever_train/manifest.json"]:
    f = ROOT/p
    print(f"\n{p}\n  exists={f.is_file()}")
    if f.is_file():
        print(" ", json.dumps(json.loads(f.read_text()))[:400])

print("\n--- RETRIEVAL METRICS (train) ---")
dm = ROOT/"outputs/module6/retrieve/fever_train/dense_metrics.json"
if dm.is_file():
    m = json.loads(dm.read_text())
    print("  overall   :", m["overall"])
    print("  accounting:", m["accounting"])

print("\n--- WOULD m6_06 NOW FIND TRAIN EVIDENCE? (this is the whole point) ---")
import sys; sys.path.insert(0, str(ROOT))
from scripts.m6_common import load_config
from scripts.m6_06_train_verifier import load_retrieved
cfg = load_config(None)
ret, src = load_retrieved(cfg, ROOT, "fever", "train")
print(f"  evidence_source = {src!r}   (MUST be 'dense_candidates', NOT 'none')")
print(f"  claims with candidates = {len(ret):,}")
if ret:
    k = next(iter(ret)); print(f"  example: {k} -> {ret[k][:3]} ...")

print("\n--- MODULE 5 UNTOUCHED? ---")
from scripts.m6_common import module5_fingerprint
after = module5_fingerprint(ROOT)
print("  IDENTICAL TO BASELINE:", after == BASELINE)   # BASELINE from Cell 1
if after != BASELINE:
    print("  *** STOP AND REPORT - Module 5 changed ***")
```

**The single line that matters:** `evidence_source = 'dense_candidates'`. If it still prints `'none'`, the repair has not landed and **we must not proceed to training** — that would reproduce the exact defect this whole exercise is fixing.

---

## What happens next (not yet — after you paste Cell 5's output back)

1. I add the `evidence_source == "none"` **hard guard** to `m6_06` so claim+evidence training can never again proceed silently without evidence. *(This one IS a necessary source change — it is repair item 6 in your instruction list.)*
2. I add `--run-name` / `--no-resume` isolation so ablations cannot resume from the primary checkpoint. *(Necessary — repair item 7.)*
3. X-FACT train retrieval (~2 min; 1,096 claims).
4. Re-run X-FACT Stage 7 on the dense fallback, after backing up the existing evaluation as audit evidence.
5. Retrain FEVER and X-FACT evidence-aware, preserving the claim-only checkpoints as the Ablation-1 arm.
6. The four X-FACT ablation arms.

**Nothing in steps 1–6 starts until you have shown me Cell 5's output.**
