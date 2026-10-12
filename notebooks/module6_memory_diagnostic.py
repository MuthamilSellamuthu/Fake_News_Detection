# DIAGNOSTIC — run BEFORE any encoding stage. ~1 minute. Downloads LaBSE once.
#
# A runtime that dies right after "Loading weights" with no traceback was
# OOM-killed: the OS terminates the kernel, so Python never gets to raise.
# This measures where the memory actually goes instead of guessing.
import shutil, sys, time
from pathlib import Path

PROJECT = Path("/content/drive/MyDrive/Fake_News_Detection")
sys.path.insert(0, str(PROJECT))


def mem():
    try:
        with open("/proc/meminfo") as f:
            d = {}
            for line in f:
                k, v = line.split(":", 1)
                d[k] = float(v.split()[0]) / 1024.0
        return d.get("MemAvailable", 0.0), d.get("MemTotal", 0.0)
    except Exception:
        return 0.0, 0.0


avail, total = mem()
print("RAM  : %.0f MB available of %.0f MB total" % (avail, total))
du = shutil.disk_usage("/content")
print("disk : %.1f GB free of %.1f GB" % (du.free / 1e9, du.total / 1e9))
try:
    import torch
    print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
except Exception as e:
    print("torch: NOT IMPORTABLE", e)

from scripts.m6_common import load_config, load_corpus

cfg = load_config(PROJECT / "configs/module6_retrieval_verification.yaml")
corpus = load_corpus(cfg, "x_fact", root=PROJECT)
lens = sorted(len(t) for t in corpus["evidence_texts"].values())
n = len(lens)
print()
print("x_fact evidence: %d items" % n)
print("  chars  median=%d  p95=%d  max=%d  total=%.1f MB"
      % (lens[n // 2], lens[int(n * 0.95)], lens[-1], sum(lens) / 1e6))
if lens[-1] > 100000:
    print("  !! Some evidence items are very large. Tokenisation truncates them")
    print("     at max_seq_length, but they still inflate peak RAM while batching.")

a0, _ = mem()
t0 = time.perf_counter()
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/LaBSE"
print()
print("loading", MODEL, "...")
try:
    m = SentenceTransformer(MODEL, device="cpu")
    a1, _ = mem()
    print("  loaded in %.1fs, RAM consumed by load: %.0f MB (available now %.0f MB)"
          % (time.perf_counter() - t0, a0 - a1, a1))
    m.max_seq_length = 256
    t1 = time.perf_counter()
    v = m.encode(list(corpus["evidence_texts"].values())[:64], batch_size=8,
                 convert_to_numpy=True, show_progress_bar=False)
    a2, _ = mem()
    dt = max(time.perf_counter() - t1, 1e-6)
    print("  encoded 64 texts in %.1fs -> %s, available now %.0f MB" % (dt, v.shape, a2))
    rate = 64 / dt
    print("  throughput ~%.1f texts/sec  =>  x_fact 6,968 in ~%.0f min,"
          "  FEVER 67,388 in ~%.0f min" % (rate, 6968 / rate / 60, 67388 / rate / 60))
    if a2 < 1500:
        print()
        print("  !! Under 1.5 GB free after a 64-text batch: LaBSE does not fit")
        print("     this runtime. Use the smaller encoder (see below).")
except Exception as e:
    print("  FAILED:", type(e).__name__, e)
    print("  -> LaBSE cannot load here; use the smaller encoder.")

print()
print("If LaBSE is too heavy, re-run Stage 1 with the lighter multilingual model:")
print("  !python scripts/m6_01_encode_evidence.py --dataset x_fact --phase test \\")
print("     --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \\")
print("     --batch-size 16 --max-seq-length 128 --shard-size 2000")
