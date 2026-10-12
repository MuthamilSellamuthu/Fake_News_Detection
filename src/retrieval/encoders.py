"""Module 6 — sharded, resumable LaBSE/E5 encoding.

Designed around free Colab disconnecting mid-run. Work is split into shards
of `shard_size` texts; each finished shard is written `.tmp.npy` then
atomically renamed, and recorded in a manifest. Re-running skips completed
shards, so a disconnect costs one shard, not the whole stage.

The `.tmp.npy` suffix is not cosmetic. `np.save` appends `.npy` unless the
path already ends in it, so a temp name like `shard.npy.tmp` silently becomes
`shard.npy.tmp.npy` and every rename fails. That exact bug was found and
fixed in Module 3; the same convention is used here deliberately.

Model loading is lazy: importing this module must not pull ~1.8 GB of LaBSE
weights, so tests and CPU stages can import it freely. Encoding is
deterministic -- eval mode, no dropout, no sampling, fixed shard order.
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


# Capability marker. A stale copy of this file on Drive is otherwise invisible:
# the CLI still accepts every flag, the run still starts, and the memory-safety
# behaviour is simply absent. Callers assert against this set so a partial sync
# fails loudly instead of silently OOM-killing the runtime.
ENCODERS_VERSION = "m6-cpu-safe-1"
ENCODER_CAPABILITIES = frozenset({
    "micro_batching",     # shard walked in batch_size chunks, not one call
    "inference_mode",     # torch.inference_mode() around the forward pass
    "gc_collect",         # explicit collection between batches and shards
    "thread_limit",       # torch.set_num_threads / set_num_interop_threads
    "ram_floor",          # LowMemoryStop before the kernel OOM-kills us
    "settings_guard",     # refuse to resume under different encoder settings
    "lazy_text_fn",       # per-shard text materialisation
})


def encoder_runtime_info() -> dict[str, Any]:
    """What the encoder actually did, for the run log.

    Reports the *effective* torch thread counts rather than what was requested,
    so the log is evidence rather than intention.
    """
    info: dict[str, Any] = {"encoders_version": ENCODERS_VERSION,
                            "capabilities": sorted(ENCODER_CAPABILITIES)}
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["torch_num_threads"] = torch.get_num_threads()
        try:
            info["torch_num_interop_threads"] = torch.get_num_interop_threads()
        except Exception:  # noqa: BLE001
            pass
        info["inference_mode_available"] = hasattr(torch, "inference_mode")
    except Exception:  # noqa: BLE001
        info["torch"] = "not importable"
    return info


class LowMemoryStop(RuntimeError):
    """Raised to stop cleanly BEFORE the OS kills the process.

    A kernel OOM-kill produces no traceback and no manifest update, so the
    shard in flight is lost and the cause is invisible. Stopping voluntarily
    keeps every completed shard and says why.
    """


def available_mb() -> float:
    """System MemAvailable in MB; 0.0 where /proc is unavailable."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def memory_report() -> dict[str, float]:
    """Process RSS and system-available RAM, in MB.

    A Colab runtime that dies immediately after model load with no traceback
    has almost certainly been OOM-killed: the kernel is terminated, so Python
    never gets to raise. Logging memory per shard turns that silent death into
    a visible trend.
    """
    out: dict[str, float] = {}
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kB, macOS bytes.
        out["peak_rss_mb"] = round(ru / 1024.0, 1) if ru > 1 << 20 else round(ru / 1024.0, 1)
    except Exception:  # noqa: BLE001
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            info = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = float(parts[1])
        if "MemAvailable" in info:
            out["available_mb"] = round(info["MemAvailable"] / 1024.0, 1)
        if "MemTotal" in info:
            out["total_mb"] = round(info["MemTotal"] / 1024.0, 1)
    except Exception:  # noqa: BLE001
        pass
    return out


def text_fingerprint(texts: Sequence[str]) -> str:
    """Hash of the exact input list, so a manifest cannot be reused for
    different data. Order-sensitive on purpose: shard i must mean the same
    rows on resume."""
    h = hashlib.sha256()
    h.update(str(len(texts)).encode())
    for t in texts:
        h.update(b"\x00")
        h.update(t.encode("utf-8"))
    return h.hexdigest()


@dataclass
class ShardPlan:
    shard_size: int
    n_items: int

    @property
    def n_shards(self) -> int:
        return (self.n_items + self.shard_size - 1) // self.shard_size if self.n_items else 0

    def bounds(self, i: int) -> tuple[int, int]:
        start = i * self.shard_size
        return start, min(start + self.shard_size, self.n_items)


class ShardedEncoder:
    """Resumable encode-to-disk driver.

    `encode_fn` maps a list of strings to an (n, dim) float32 array. Injecting
    it keeps this class testable without downloading any model.
    """

    def __init__(self, out_dir: Path, shard_size: int = 10000,
                 encode_fn: Callable[[Sequence[str]], np.ndarray] | None = None,
                 model_name: str = "sentence-transformers/LaBSE") -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = int(shard_size)
        self.encode_fn = encode_fn
        self.model_name = model_name
        self.manifest_path = self.out_dir / "manifest.json"

    # ------------------------------------------------------------ manifest
    def load_manifest(self) -> dict[str, Any]:
        if self.manifest_path.is_file():
            try:
                return json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
        return {}

    def _write_manifest(self, m: dict[str, Any]) -> None:
        tmp = self.manifest_path.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.manifest_path)

    def shard_path(self, i: int) -> Path:
        return self.out_dir / f"shard_{i:05d}.npy"

    # -------------------------------------------------------------- encode
    def encode(self, texts: Sequence[str], ids: Sequence[str] | None = None,
               log: Callable[[str], None] = print,
               min_available_mb: float = 0.0,
               settings: dict[str, Any] | None = None,
               text_fn: Callable[[int, int], list[str]] | None = None) -> dict[str, Any]:
        """Encode shard by shard, saving each one before starting the next.

        `min_available_mb` stops cleanly if system RAM drops below the
        threshold, instead of letting the kernel OOM-kill the process (which
        yields no traceback and loses the in-flight shard).

        `settings` (model / batch / seq len) is stored in the manifest and
        re-checked on resume: finishing a corpus half at seq-128 and half at
        seq-256 would silently produce two incompatible embedding spaces.

        `text_fn(lo, hi)` supplies one shard's texts lazily, so the full text
        list never has to be materialised alongside the corpus.
        """
        if self.encode_fn is None:
            raise RuntimeError("no encode_fn: pass one, or call build_labse_encode_fn()")
        if ids is not None and len(ids) != len(texts):
            raise ValueError(f"{len(ids)} ids but {len(texts)} texts")

        plan = ShardPlan(self.shard_size, len(texts))
        fp = text_fingerprint(texts)
        man = self.load_manifest()

        if man.get("fingerprint") and man["fingerprint"] != fp:
            raise RuntimeError(
                "Manifest fingerprint does not match these inputs. The corpus changed "
                "since the last run; delete the stage directory to re-encode rather "
                "than mixing embeddings from two different corpora.")

        if settings:
            prev = man.get("settings")
            if prev and prev != settings:
                raise RuntimeError(
                    f"Encoder settings changed between runs: {prev} -> {settings}. "
                    f"Resuming would mix embeddings produced under different settings. "
                    f"Delete {self.out_dir} to re-encode from scratch, or restore the "
                    f"previous settings.")
            man["settings"] = settings

        completed = set(man.get("completed_shards", []))
        # Trust the file, not just the manifest: a crash between rename and
        # manifest write would otherwise lose a good shard.
        for i in range(plan.n_shards):
            if self.shard_path(i).is_file():
                completed.add(i)

        man.update({"model": self.model_name, "fingerprint": fp, "n_items": len(texts),
                    "shard_size": self.shard_size, "n_shards": plan.n_shards,
                    "completed_shards": sorted(completed), "status": "in_progress"})
        self._write_manifest(man)

        t0 = time.perf_counter()
        ram_seen: list[float] = []
        for i in range(plan.n_shards):
            if i in completed:
                continue
            lo, hi = plan.bounds(i)

            avail = available_mb()
            if avail:
                ram_seen.append(avail)
            if min_available_mb and avail and avail < min_available_mb:
                man["status"] = "stopped_low_memory"
                man["stopped_at_shard"] = i
                man["available_mb_at_stop"] = avail
                man["min_available_mb_observed"] = min(ram_seen) if ram_seen else None
                self._write_manifest(man)
                raise LowMemoryStop(
                    f"Stopping before shard {i}: only {avail:.0f} MB RAM available "
                    f"(threshold {min_available_mb:.0f} MB). "
                    f"{len(completed)}/{plan.n_shards} shards are saved and will be "
                    f"skipped on resume. Re-run the same command after restarting the "
                    f"runtime, or lower --batch-size / --shard-size.")

            batch_texts = text_fn(lo, hi) if text_fn else list(texts[lo:hi])
            vecs = np.asarray(self.encode_fn(batch_texts), dtype=np.float32)
            if vecs.shape[0] != hi - lo:
                raise RuntimeError(f"shard {i}: encoder returned {vecs.shape[0]} rows "
                                   f"for {hi - lo} texts")
            tmp = self.out_dir / f"shard_{i:05d}.tmp.npy"   # .tmp.npy, not .npy.tmp
            np.save(tmp, vecs)
            tmp.replace(self.shard_path(i))
            completed.add(i)
            man["completed_shards"] = sorted(completed)
            man["last_shard_seconds"] = round(time.perf_counter() - t0, 2)
            man["min_available_mb_observed"] = min(ram_seen) if ram_seen else None
            self._write_manifest(man)

            # Release the shard's tensors/arrays before the next one starts.
            del batch_texts, vecs
            gc.collect()

            after = available_mb()
            log(f"  shard {i + 1}/{plan.n_shards} [{lo}:{hi}] -> {self.shard_path(i).name}"
                + (f"  | RAM {avail:.0f}->{after:.0f} MB" if avail else ""))

        if ids is not None and not (self.out_dir / "ids.json").is_file():
            tmp = self.out_dir / "ids.tmp.json"
            tmp.write_text(json.dumps(list(ids), ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.out_dir / "ids.json")

        man["status"] = "complete"
        man["total_seconds"] = round(time.perf_counter() - t0, 2)
        self._write_manifest(man)
        return man

    # --------------------------------------------------------------- load
    def load_all(self) -> np.ndarray:
        man = self.load_manifest()
        n_shards = int(man.get("n_shards", 0))
        missing = [i for i in range(n_shards) if not self.shard_path(i).is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} shard(s) missing, e.g. {missing[:5]}; "
                                    f"re-run the encode stage to resume")
        if n_shards == 0:
            return np.zeros((0, 0), dtype=np.float32)
        parts = [np.load(self.shard_path(i)) for i in range(n_shards)]
        out = np.concatenate(parts, axis=0)
        if out.shape[0] != int(man.get("n_items", out.shape[0])):
            raise RuntimeError(f"loaded {out.shape[0]} rows, manifest says {man.get('n_items')}")
        return out

    def load_ids(self) -> list[str]:
        p = self.out_dir / "ids.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else []


def build_labse_encode_fn(model_name: str = "sentence-transformers/LaBSE",
                          batch_size: int = 64, max_seq_length: int = 256,
                          device: str | None = None, num_threads: int | None = None,
                          gc_every_batches: int = 8,
                          log: Callable[[str], None] | None = None
                          ) -> Callable[[Sequence[str]], np.ndarray]:
    """Memory-bounded sentence-transformers encoder.

    The original version handed the whole shard to `model.encode()`. With the
    default 10,000-item shard that meant one call covering the entire 6,968-item
    X-FACT pool: sentence-transformers length-sorts the input, builds batches of
    64 at seq-256, and holds every output embedding plus internal buffers until
    the call returns. On a free CPU runtime that peak is what gets the process
    OOM-killed -- silently, because the kernel terminates it before Python can
    raise.

    Here the shard is walked in micro-batches, each result copied into a
    pre-allocated output array and then released, so peak memory is set by
    `batch_size` rather than by shard size.

    `num_threads` matters more than it looks: torch defaults to one thread per
    core and each carries its own workspace, so an unbounded thread count
    multiplies peak RAM on a small container. 2 is a sane default for Colab.

    Embeddings are byte-identical to the previous implementation for the same
    model and max_seq_length -- micro-batching changes only when memory is
    freed, not what is computed. Normalisation still happens in the index.
    """
    import torch
    from sentence_transformers import SentenceTransformer  # local import on purpose

    if num_threads:
        torch.set_num_threads(int(num_threads))
        try:
            torch.set_num_interop_threads(1)
        except Exception:  # noqa: BLE001 - only settable before parallel work starts
            pass

    model = SentenceTransformer(model_name, device=device or "cpu")
    model.max_seq_length = int(max_seq_length)
    model.eval()

    dim = int(model.get_sentence_embedding_dimension())
    bs = max(1, int(batch_size))

    def _encode(texts: Sequence[str]) -> np.ndarray:
        items = list(texts)
        out = np.empty((len(items), dim), dtype=np.float32)   # allocated once
        n_batches = (len(items) + bs - 1) // bs
        with torch.inference_mode():                          # no autograd graph
            for b in range(n_batches):
                lo = b * bs
                hi = min(lo + bs, len(items))
                vecs = model.encode(items[lo:hi], batch_size=bs, convert_to_numpy=True,
                                    normalize_embeddings=False, show_progress_bar=False)
                out[lo:hi] = np.asarray(vecs, dtype=np.float32)
                del vecs
                if gc_every_batches and (b + 1) % gc_every_batches == 0:
                    gc.collect()
                    if log and (b + 1) % (gc_every_batches * 4) == 0:
                        log(f"    batch {b + 1}/{n_batches}  RAM {available_mb():.0f} MB")
        gc.collect()
        return out

    return _encode
