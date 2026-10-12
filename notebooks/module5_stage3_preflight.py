"""Module 5 Stage 3 — PREFLIGHT ONLY. Paste into a Colab cell and run.

Verifies the Stage 3 files, compiles them, runs the four suites, checks for
duplicates, re-hashes the Module 1-4 CSVs, and confirms the real X-FACT and
FEVER inputs exist with their sizes.

Builds nothing. Downloads nothing. Modifies nothing. Trains nothing.
Every check prints PASS or FAIL; the cell exits non-zero if any FAIL.
"""

import hashlib
import re
import subprocess
import sys
from pathlib import Path

PROJECT = Path("/content/drive/MyDrive/Fake_News_Detection")
FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  |  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)
    return ok


def run(cmd):
    r = subprocess.run(cmd, cwd=PROJECT, shell=True, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{n:,} B"
        n /= 1024


print("=" * 66)
print("MODULE 5 STAGE 3 — PREFLIGHT (no build, no download, no training)")
print("=" * 66)

if not PROJECT.exists():
    raise SystemExit(f"FAIL: project not found at {PROJECT}. Mount Drive first.")

# ---------------------------------------------------------------- 1. files exist
print("\n[1] Required files exist")
REQUIRED = [
    "src/fact_verification/leakage_gate.py",
    "src/fact_verification/corpus_builder.py",
    "scripts/build_claim_corpus.py",
    "scripts/audit_stage3_corpus.py",
    "tests/test_fact_verification.py",
]
for rel in REQUIRED:
    p = PROJECT / rel
    check(rel, p.is_file(), human(p.stat().st_size) if p.is_file() else "MISSING")

# ------------------------------------------------------------- 2. py_compile
print("\n[2] py_compile (implementation files)")
impl = ("src/fact_verification/leakage_gate.py src/fact_verification/corpus_builder.py "
        "scripts/build_claim_corpus.py scripts/audit_stage3_corpus.py")
rc, out = run(f"python -m py_compile {impl}")
check("py_compile all four implementation files", rc == 0, out.strip()[:300] or "clean")

# ------------------------------------------- 2b. content sanity (the known fixes)
print("\n[2b] Expected fixes present")
try:
    lg = (PROJECT / "src/fact_verification/leakage_gate.py").read_text(encoding="utf-8")
    cb = (PROJECT / "src/fact_verification/corpus_builder.py").read_text(encoding="utf-8")
    tf = (PROJECT / "tests/test_fact_verification.py").read_text(encoding="utf-8")
    check("leakage_gate defines candidate_pairs", "def candidate_pairs(" in lg)
    check("old coarse blocking path removed", "buckets.setdefault(bkey" not in lg)
    check("corpus_builder uses SHA-256 evidence_id",
          "hashlib.sha256(normalized_text.encode" in cb)
    check("evidence_id is not a raw-text prefix", "hash_key(e_norm)[:64]" not in cb)
    check("regression test present (64-char prefix)",
          "test_evidence_ids_differ_for_shared_64_char_prefix" in tf)
    check("regression test present (filter exactness)",
          "test_candidate_filter_is_exact_vs_brute_force" in tf)
except OSError as e:
    check("read Stage 3 sources", False, str(e))

# ------------------------------------------------------------- 3. no duplicates
print("\n[3] Exactly one copy of each Stage 3 file")
for folder, pattern in [("scripts", "build_claim_corpus*.py"),
                        ("scripts", "audit_stage3_corpus*.py"),
                        ("src/fact_verification", "corpus_builder*.py"),
                        ("src/fact_verification", "leakage_gate*.py"),
                        ("tests", "test_fact_verification*.py")]:
    d = PROJECT / folder
    hits = [q.name for q in d.glob(pattern) if "__pycache__" not in str(q)] if d.is_dir() else []
    check(f"{folder}/{pattern}", len(hits) == 1, f"found {hits}")

# ----------------------------------------------------------------- 4. test suites
print("\n[4] Test suites (executed against the Drive copies)")
EXPECTED = {"tests/test_fact_verification.py": 43,
            "tests/test_module5_corpus_schema.py": 21,
            "tests/test_contrastive.py": 22,
            "tests/test_embeddings.py": 5}
for t, exp in EXPECTED.items():
    if not (PROJECT / t).is_file():
        check(f"{t} = {exp}/{exp}", False, "SUITE FILE MISSING")
        continue
    rc, out = run(f"python {t}")
    m = None
    for line in out.strip().splitlines():
        hit = re.search(r"(\d+)/(\d+) passed", line)
        if hit:
            m = hit
    if m is None:
        check(f"{t} = {exp}/{exp}", False, "no result line: " + out.strip()[-200:])
    else:
        got, tot = int(m.group(1)), int(m.group(2))
        check(f"{t} = {exp}/{exp}", rc == 0 and got == tot == exp, f"reported {got}/{tot}")

# --------------------------------------------------- 5. Module 1-4 CSV integrity
print("\n[5] Module 1-4 processed CSVs unchanged (read-only)")
FROZEN = {"datasets/processed/train.csv": "fc26454e56cb9704509ffe3216d52a22",
          "datasets/processed/validation.csv": "081f7b828477ab9f27725d17c34b525a",
          "datasets/processed/test.csv": "66599a4f9d71bd5cee1927c3a573f0ce"}
for rel, exp in FROZEN.items():
    p = PROJECT / rel
    if not p.is_file():
        check(rel, False, "FILE MISSING")
        continue
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    check(f"{rel} md5", got == exp, f"{got}  ({human(p.stat().st_size)})")

# ------------------------------------------- 6. real corpora exist + actual sizes
print("\n[6] Real X-FACT / FEVER inputs (existence + actual sizes)")
print("     (row counts stream the files; FEVER may take a minute over Drive)")
SOURCES = {"x_fact": ("datasets/module5_external/x_fact", 1463),
           "fever": ("datasets/module5_external/fever_gold_evidence", 260251)}
for name, (rel, expected_rows) in SOURCES.items():
    d = PROJECT / rel
    if not d.is_dir():
        check(f"{name}: source directory {rel}", False, "DIRECTORY MISSING")
        continue
    files = sorted(d.glob("*.jsonl"))
    if not files:
        check(f"{name}: .jsonl files present", False, f"none found in {rel}")
        continue
    total = 0
    print(f"     {name}:")
    for fp in files:
        n = 0
        with fp.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
        total += n
        print(f"       {fp.name:<32} {human(fp.stat().st_size):>12}   {n:,} rows")
    check(f"{name}: total rows == {expected_rows:,}", total == expected_rows,
          f"counted {total:,}")

# ------------------------------------------------------------------- verdict
print("\n" + "=" * 66)
if FAILURES:
    print(f"PREFLIGHT: FAIL  ({len(FAILURES)} check(s))")
    for f in FAILURES:
        print(f"  - {f}")
    print("\nDo NOT run scripts/build_claim_corpus.py.")
else:
    print("PREFLIGHT: PASS")
    print("Corpus build was NOT executed. Nothing downloaded, nothing modified.")
print("=" * 66)
if FAILURES:
    raise SystemExit(1)
