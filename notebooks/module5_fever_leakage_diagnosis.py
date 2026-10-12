"""FEVER leakage diagnosis — STRICTLY READ-ONLY. Run in Colab.

Answers what the build report cannot: the gate's near-duplicate check
stops after 10 examples, so the true violation count is unknown, and the
gate has no evidence-leakage check at all. This recomputes both
independently, reusing the SHIPPED candidate_pairs/jaccard functions
without modifying them.

Opens every file in read mode. Writes nothing, deletes nothing,
downloads nothing, rebuilds nothing, trains nothing.

    python notebooks/module5_fever_leakage_diagnosis.py

Memory note: section 5 holds char-5-gram sets for all 260k claims
(~2-3 GB). If it is killed, use Runtime > Change runtime type > High-RAM.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

PROJECT = Path("/content/drive/MyDrive/Fake_News_Detection")
sys.path.insert(0, str(PROJECT))

FEVER_DIR = PROJECT / "datasets/fact_verification/fever"
CLAIMS_PATH = FEVER_DIR / "claims.jsonl"
EVIDENCE_PATH = FEVER_DIR / "evidence.jsonl"
SPLITS_PATH = FEVER_DIR / "claim_splits.json"
REPORT_PATH = PROJECT / "outputs/module5/corpus_build_report.json"

THRESHOLD = 0.90
NGRAM = 5
SOURCE_SPLIT_ROWS = {"train": 228277, "validation": 15935, "test": 16039}

SEP = "=" * 72


def head(t):
    print(f"\n{SEP}\n{t}\n{SEP}")


# ----------------------------------------------------------- 10. safety first
head("[10] SAFETY PRECHECK (read-only)")
required = [CLAIMS_PATH, EVIDENCE_PATH, SPLITS_PATH, REPORT_PATH]
missing = [p for p in required if not p.is_file()]
if missing:
    print("STOP — required file(s) missing:")
    for p in missing:
        print(f"  {p}")
    raise SystemExit(1)
for p in required:
    print(f"  found  {p.relative_to(PROJECT)}  ({p.stat().st_size:,} bytes)")
print("\n  This script opens files with mode='r' only.")
print("  No writes, no deletes, no downloads, no training, no rebuild.")

# ----------------------------------------------------------- schema inspection
head("[0] ACTUAL SCHEMA (observed, not assumed)")
with CLAIMS_PATH.open(encoding="utf-8") as f:
    first_claim = json.loads(f.readline())
with EVIDENCE_PATH.open(encoding="utf-8") as f:
    first_ev = json.loads(f.readline())
splits_doc = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
print(f"  claims.jsonl keys   : {sorted(first_claim)}")
print(f"  metadata keys       : {sorted(first_claim.get('metadata', {}))}")
print(f"  evidence.jsonl keys : {sorted(first_ev)}")
print(f"  claim_splits keys   : {sorted(splits_doc)}")
print(f"  evidence field type : {type(first_claim.get('evidence')).__name__}")

report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
lk = report.get("corpora", {}).get("fever", {}).get("leakage") or {}
print(f"\n  gate result from report: passed={lk.get('passed')} "
      f"{lk.get('n_passed')}/{lk.get('n_checks')} runtime={lk.get('seconds')}s")
for c in lk.get("checks", []):
    print(f"    {'PASS' if c.get('passed') else 'FAIL'}  {c.get('name')}")
    print(f"        {str(c.get('detail'))[:200]}")

# --------------------------------------------------------------- load corpus
t0 = time.perf_counter()
claims = []
with CLAIMS_PATH.open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            claims.append(json.loads(line))
evidence = []
with EVIDENCE_PATH.open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            evidence.append(json.loads(line))
print(f"\n  loaded {len(claims):,} claims and {len(evidence):,} evidence "
      f"records in {time.perf_counter()-t0:.1f}s")

split_of = {}
for s, idxs in splits_doc.get("splits", {}).items():
    for i in idxs:
        split_of[i] = s
SPLITS = sorted(splits_doc.get("splits", {}))
PAIRS = list(combinations(SPLITS, 2))

# ------------------------------------------------------ 1. FEVER row structure
head("[1] FEVER ROW STRUCTURE")
texts = Counter(c["text"] for c in claims)
claim_ids = Counter(c["claim_id"] for c in claims)
rec_ids = Counter(str(c.get("record_id")) for c in claims)
dup_text_rows = sum(v for v in texts.values() if v > 1)
print(f"  total JSONL rows            : {len(claims):,}")
print(f"  distinct claim_id           : {len(claim_ids):,}")
print(f"  distinct record_id          : {len(rec_ids):,}")
print(f"  distinct claim TEXT         : {len(texts):,}")
print(f"  rows whose text repeats     : {dup_text_rows:,} "
      f"({100*dup_text_rows/len(claims):.2f}%)")
print(f"  max rows sharing one text   : {max(texts.values()):,}")
print(f"  max rows sharing one rec_id : {max(rec_ids.values()):,}")
print(f"  mean rows per distinct text : {len(claims)/max(len(texts),1):.3f}")
dupdist = Counter(v for v in texts.values() if v > 1)
print(f"  duplicate-frequency (text repeats -> how many texts): "
      f"{dict(sorted(dupdist.items())[:10])}")
if len(claim_ids) < len(claims):
    print(f"  !! {len(claims)-len(claim_ids):,} DUPLICATE claim_id values")
if len(rec_ids) < len(claims):
    print("  => source rows are claim-EVIDENCE pairs: one record_id spans many rows.")
else:
    print("  => every row has a unique record_id: rows are NOT claim-evidence pairs;")
    print("     repeated texts are distinct source records sharing wording.")

# ---------------------------------------------------- 2. official split structure
head("[2] OFFICIAL SPLIT STRUCTURE")
print(f"  counts from claim_splits.json : {splits_doc.get('counts')}")
built = Counter(c.get("metadata", {}).get("official_split") for c in claims)
print(f"  counts from claim metadata    : {dict(built)}")
assigned = Counter()
for s, idxs in splits_doc.get("splits", {}).items():
    for i in idxs:
        assigned[i] += 1
n_missing = sum(1 for i in range(len(claims)) if assigned[i] == 0)
n_multi = sum(1 for i, v in assigned.items() if v > 1)
print(f"  claims assigned exactly once  : "
      f"{'YES' if n_missing == 0 and n_multi == 0 else 'NO'} "
      f"(missing={n_missing}, multiple={n_multi})")
print("\n  dropped-row accounting:")
tot_drop = 0
for s, src in SOURCE_SPLIT_ROWS.items():
    got = built.get(s, 0)
    tot_drop += src - got
    print(f"    {s:<11} source={src:>7,}  built={got:>7,}  dropped={src-got:>4,}")
print(f"    total dropped = {tot_drop} (build reported 219) -> "
      f"{'CONSISTENT' if tot_drop == 219 else 'MISMATCH'}")
print("  Dropped rows are below-min-length rows removed BEFORE splitting;")
print("  they remove claims, so they cannot create cross-split overlap.")

# ------------------------------------------------- 3. exact cross-split overlap
head("[3] EXACT CROSS-SPLIT CLAIM OVERLAP")
text_to_splits = defaultdict(set)
text_to_idx = defaultdict(list)
for i, c in enumerate(claims):
    s = split_of.get(i)
    if s:
        text_to_splits[c["text"]].add(s)
    text_to_idx[c["text"]].append(i)

for a, b in PAIRS:
    shared = [t for t, ss in text_to_splits.items() if a in ss and b in ss]
    rows = sum(len(text_to_idx[t]) for t in shared)
    print(f"  {a:<11} vs {b:<11}: {len(shared):,} shared texts, {rows:,} rows")

train_texts = {claims[i]["text"] for i in splits_doc["splits"].get("train", [])}
test_idx = splits_doc["splits"].get("test", [])
val_idx = splits_doc["splits"].get("validation", [])
test_exact = [i for i in test_idx if claims[i]["text"] in train_texts]
val_exact = [i for i in val_idx if claims[i]["text"] in train_texts]
print(f"\n  TEST claims exactly overlapping TRAIN       : {len(test_exact):,} "
      f"of {len(test_idx):,} ({100*len(test_exact)/max(len(test_idx),1):.2f}%)")
print(f"  VALIDATION claims exactly overlapping TRAIN : {len(val_exact):,} "
      f"of {len(val_idx):,} ({100*len(val_exact)/max(len(val_idx),1):.2f}%)")
print("\n  example overlapping claim_ids (test-side, with a train partner):")
for i in test_exact[:10]:
    partners = [j for j in text_to_idx[claims[i]["text"]] if split_of.get(j) == "train"][:1]
    p = partners[0] if partners else None
    print(f"    TEST  {claims[i]['claim_id']}  rec={claims[i].get('record_id')} "
          f"label={claims[i].get('label')}")
    if p is not None:
        print(f"    TRAIN {claims[p]['claim_id']}  rec={claims[p].get('record_id')} "
              f"label={claims[p].get('label')}")
    print(f"      text: {claims[i]['text'][:110]!r}")

# ----------------------------------------------------------- 4. group leakage
head("[4] GROUP LEAKAGE (near-duplicate groups spanning official splits)")
gid_splits = defaultdict(set)
gid_idx = defaultdict(list)
for i, c in enumerate(claims):
    g = c.get("claim_group_id")
    s = split_of.get(i)
    if s:
        gid_splits[g].add(s)
    gid_idx[g].append(i)
spanning = {g: ss for g, ss in gid_splits.items() if len(ss) > 1}
print(f"  total groups                : {len(gid_idx):,}")
print(f"  groups spanning >1 split    : {len(spanning):,} "
      f"({100*len(spanning)/max(len(gid_idx),1):.2f}%)")
for a, b in PAIRS:
    n = sum(1 for ss in spanning.values() if a in ss and b in ss)
    print(f"    spanning {a:<11} & {b:<11}: {n:,} groups")

train_groups = {claims[i].get("claim_group_id") for i in splits_doc["splits"].get("train", [])}
test_in_train_group = [i for i in test_idx if claims[i].get("claim_group_id") in train_groups]
val_in_train_group = [i for i in val_idx if claims[i].get("claim_group_id") in train_groups]
affected_groups = {claims[i].get("claim_group_id") for i in test_in_train_group}
train_side = [i for i in splits_doc["splits"].get("train", [])
              if claims[i].get("claim_group_id") in affected_groups]
print(f"\n  *** TEST claims whose group also contains a TRAIN claim: "
      f"{len(test_in_train_group):,} of {len(test_idx):,} "
      f"({100*len(test_in_train_group)/max(len(test_idx),1):.2f}%)")
print(f"  *** corresponding TRAIN claims involved: {len(train_side):,} "
      f"across {len(affected_groups):,} groups")
print(f"  VALIDATION claims sharing a group with TRAIN: {len(val_in_train_group):,} "
      f"({100*len(val_in_train_group)/max(len(val_idx),1):.2f}%)")
print("\n  representative spanning groups:")
for g in list(affected_groups)[:5]:
    members = gid_idx[g]
    print(f"    group {g} (size {len(members)}) splits={sorted(gid_splits[g])}")
    for i in members[:3]:
        print(f"      [{split_of.get(i):<10}] {claims[i]['claim_id']}  "
              f"{claims[i]['text'][:80]!r}")

# ------------------------------------------- 5. TRUE near-duplicate leakage
head("[5] COMPLETE CROSS-SPLIT NEAR-DUPLICATE SCAN (uncapped)")
print(f"  threshold={THRESHOLD}  ngram={NGRAM}  (identical to the shipped gate)")
print("  importing the SHIPPED candidate_pairs/jaccard — gate code unmodified")
from src.fact_verification.leakage_gate import candidate_pairs, char_ngrams, jaccard

t0 = time.perf_counter()
grams = [char_ngrams(c["text"], NGRAM) for c in claims]
langs = [str(c.get("language")) for c in claims]
print(f"  built {len(grams):,} n-gram sets in {time.perf_counter()-t0:.1f}s; scanning...")

t1 = time.perf_counter()
pair_counts = Counter()
sims = []
test_hit = set()
train_hit = set()
examples = []
n_cand = 0
for i, j in candidate_pairs(grams, langs, THRESHOLD):
    n_cand += 1
    a, b = split_of.get(i), split_of.get(j)
    if a is None or b is None or a == b:
        continue
    sim = jaccard(grams[i], grams[j])
    if sim < THRESHOLD:
        continue
    key = tuple(sorted((a, b)))
    pair_counts[key] += 1
    sims.append(sim)
    if key == ("test", "train"):
        t_i, tr_i = (i, j) if a == "test" else (j, i)
        test_hit.add(t_i)
        train_hit.add(tr_i)
    if len(examples) < 10:
        examples.append((i, j, a, b, sim))
scan_s = time.perf_counter() - t1

print(f"\n  candidate pairs examined : {n_cand:,}")
print(f"  scan runtime             : {scan_s:.1f}s")
print(f"  TOTAL cross-split violating pairs: {sum(pair_counts.values()):,}")
for a, b in PAIRS:
    print(f"    {a:<11} vs {b:<11}: {pair_counts[tuple(sorted((a,b)))]:,} pairs")
print(f"\n  TEST claims with >=1 near-duplicate in TRAIN : {len(test_hit):,} "
      f"({100*len(test_hit)/max(len(test_idx),1):.2f}% of test)")
print(f"  TRAIN claims involved                        : {len(train_hit):,}")
if sims:
    print(f"  maximum similarity among violations : {max(sims):.6f}")
    print(f"  minimum similarity among violations : {min(sims):.6f}")
    print(f"  mean similarity                     : {sum(sims)/len(sims):.6f}")
    print(f"  violations at similarity == 1.0     : {sum(1 for s in sims if s >= 0.999999):,}")
else:
    print("  no cross-split violations found")

print("\n  10 representative violating pairs:")
for i, j, a, b, sim in examples:
    print(f"    sim={sim:.4f}")
    print(f"      [{a:<10}] {claims[i]['claim_id']}  {claims[i]['text'][:95]!r}")
    print(f"      [{b:<10}] {claims[j]['claim_id']}  {claims[j]['text'][:95]!r}")

del grams

# ------------------------------------------------------------ 6. evidence overlap
head("[6] EVIDENCE OVERLAP ACROSS SPLITS (gate does NOT check this)")
ev_splits = defaultdict(set)
ev_claims = defaultdict(list)
for i, c in enumerate(claims):
    s = split_of.get(i)
    for eid in (c.get("evidence") or []):
        if s:
            ev_splits[eid].add(s)
        ev_claims[eid].append(i)
multi = {e: ss for e, ss in ev_splits.items() if len(ss) > 1}
print(f"  distinct evidence ids referenced : {len(ev_splits):,}")
print(f"  evidence appearing in >1 split   : {len(multi):,} "
      f"({100*len(multi)/max(len(ev_splits),1):.2f}%)")
for a, b in PAIRS:
    n = sum(1 for ss in ev_splits.values() if a in ss and b in ss)
    print(f"    shared {a:<11} & {b:<11}: {n:,} evidence items")

train_ev = {e for e, ss in ev_splits.items() if "train" in ss}
test_ev = {e for e, ss in ev_splits.items() if "test" in ss}
shared_tt = test_ev & train_ev
print(f"\n  TEST evidence items also used by TRAIN : {len(shared_tt):,} "
      f"of {len(test_ev):,} ({100*len(shared_tt)/max(len(test_ev),1):.2f}%)")
test_claims_shared_ev = [i for i in test_idx
                         if any(e in train_ev for e in (claims[i].get("evidence") or []))]
print(f"  TEST claims citing evidence also used by TRAIN: {len(test_claims_shared_ev):,} "
      f"of {len(test_idx):,} ({100*len(test_claims_shared_ev)/max(len(test_idx),1):.2f}%)")
ev_text = {e["evidence_id"]: e.get("text", "") for e in evidence}
print("\n  10 examples of evidence shared between train and test:")
for e in list(shared_tt)[:10]:
    users = ev_claims[e]
    tr = next((k for k in users if split_of.get(k) == "train"), None)
    te = next((k for k in users if split_of.get(k) == "test"), None)
    print(f"    {e[:46]}...  used by {len(users)} claims")
    print(f"      text: {ev_text.get(e,'')[:95]!r}")
    if tr is not None:
        print(f"      TRAIN {claims[tr]['claim_id']}")
    if te is not None:
        print(f"      TEST  {claims[te]['claim_id']}")

# ------------------------------------------------------- 7. evidence-less claims
head("[7] CLAIMS WITHOUT EVIDENCE")
no_ev = [i for i, c in enumerate(claims) if not (c.get("evidence") or [])]
by_split = Counter(split_of.get(i) for i in no_ev)
print(f"  total claims without evidence : {len(no_ev):,} "
      f"({100*len(no_ev)/len(claims):.3f}% of corpus)")
for s in SPLITS:
    tot = len(splits_doc["splits"].get(s, []))
    print(f"    {s:<11}: {by_split.get(s,0):>5,} of {tot:>7,} "
          f"({100*by_split.get(s,0)/max(tot,1):.3f}%)")
print(f"  labels: {Counter(str(claims[i].get('label')) for i in no_ev).most_common()}")
print(f"  extraction methods: "
      f"{Counter(claims[i].get('metadata',{}).get('evidence_extraction_method') for i in no_ev).most_common()}")
print("\n  10 examples:")
for i in no_ev[:10]:
    print(f"    [{split_of.get(i):<10}] {claims[i]['claim_id']} "
          f"label={claims[i].get('label')}  {claims[i]['text'][:80]!r}")
print("\n  NOTE: the gate defines no evidence-count check; these are NOT leakage.")

# --------------------------------------------------------------- 8. summary
head("FEVER LEAKAGE DIAGNOSIS — SUMMARY")
uniq_ratio = len(rec_ids) / max(len(claims), 1)
print(f"Source-layout finding : {len(claims):,} rows / {len(rec_ids):,} distinct record_id "
      f"/ {len(texts):,} distinct texts "
      f"({'claim-evidence pairs' if uniq_ratio < 0.999 else 'one row per record'})")
print(f"Exact overlap         : {len(test_exact):,} test claims share exact text with train "
      f"({100*len(test_exact)/max(len(test_idx),1):.2f}%)")
print(f"Group leakage         : {len(spanning):,} spanning groups; "
      f"{len(test_in_train_group):,} test claims share a group with train "
      f"({100*len(test_in_train_group)/max(len(test_idx),1):.2f}%)")
print(f"Near-duplicate leakage: {sum(pair_counts.values()):,} cross-split pairs >= {THRESHOLD}; "
      f"{len(test_hit):,} test claims affected")
print(f"Evidence leakage      : {len(shared_tt):,} evidence items shared train/test; "
      f"{len(test_claims_shared_ev):,} test claims affected "
      f"({100*len(test_claims_shared_ev)/max(len(test_idx),1):.2f}%)")
print(f"Evidence-less claims  : {len(no_ev):,} ({dict(by_split)})")
print("\nRead-only diagnosis complete. Nothing was modified.")
print(SEP)
