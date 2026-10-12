"""Module 3 — runtime, device, and throughput telemetry.

Collects the execution-environment and performance metrics that a
publication-quality methods section needs (GPU model, CUDA/PyTorch
versions, peak GPU memory, mixed-precision status, per-batch timings,
throughput) and that cannot be reconstructed after the fact — peak GPU
memory in particular is only observable *while* the run is happening,
which is why this lives in the extraction pipeline rather than in the
post-hoc verifier.

Everything here degrades gracefully when torch is missing or no GPU is
present: a CPU-only run still produces a complete, honest record that
says so, rather than failing or silently omitting fields. That matters
because the report consumer needs to be able to tell "ran on CPU" apart
from "GPU info wasn't recorded".
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field
from typing import Any


def collect_device_info(device: Any = None) -> dict[str, Any]:
    """Snapshot the execution environment.

    Returns a dict that always contains the same keys, so downstream
    reports never have to guard on their presence. `gpu_available` is
    the field to branch on; when it's False the GPU-specific fields are
    explicitly `None` rather than absent.
    """
    info: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "cudnn_version": None,
        "gpu_available": False,
        "gpu_name": None,
        "gpu_count": 0,
        "gpu_total_memory_bytes": None,
        "gpu_total_memory_human": None,
        "gpu_capability": None,
        "device_used": str(device) if device is not None else None,
    }

    try:
        import torch
    except ImportError:
        info["torch_import_error"] = "torch is not installed in this environment"
        return info

    info["torch_version"] = torch.__version__
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["cuda_version"] = getattr(torch.version, "cuda", None)
    try:
        info["cudnn_version"] = torch.backends.cudnn.version()
    except Exception:  # noqa: BLE001 - cudnn may be unavailable; not worth failing a run over
        info["cudnn_version"] = None

    if info["cuda_available"]:
        try:
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            info.update(
                {
                    "gpu_available": True,
                    "gpu_name": props.name,
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_total_memory_bytes": int(props.total_memory),
                    "gpu_total_memory_human": human_bytes(props.total_memory),
                    "gpu_capability": f"{props.major}.{props.minor}",
                }
            )
        except Exception as exc:  # noqa: BLE001
            info["gpu_probe_error"] = str(exc)

    return info


def reset_peak_memory() -> None:
    """Zero CUDA's peak-memory counter so `peak_gpu_memory()` measures
    this run only, not whatever else the process did earlier. No-op
    without CUDA.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:  # noqa: BLE001
        pass


def peak_gpu_memory() -> dict[str, Any]:
    """Peak CUDA memory since the last reset.

    `allocated` is memory actually held by tensors; `reserved` is what
    the caching allocator took from the driver (always >= allocated, and
    the figure closer to what `nvidia-smi` displays). Both are reported
    because they answer different questions — "how big is my model+
    activations" vs. "how much GPU do I need to provision".
    """
    result: dict[str, Any] = {
        "peak_allocated_bytes": None,
        "peak_allocated_human": None,
        "peak_reserved_bytes": None,
        "peak_reserved_human": None,
        "measured": False,
    }
    try:
        import torch

        if not torch.cuda.is_available():
            return result
        allocated = int(torch.cuda.max_memory_allocated())
        reserved = int(torch.cuda.max_memory_reserved())
        result.update(
            {
                "peak_allocated_bytes": allocated,
                "peak_allocated_human": human_bytes(allocated),
                "peak_reserved_bytes": reserved,
                "peak_reserved_human": human_bytes(reserved),
                "measured": True,
            }
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
    return result


@dataclass
class BatchTimer:
    """Accumulates per-batch wall-clock timings for one split.

    Keeps every individual batch duration (they're small — a few
    thousand floats at most) so the report can give median/p95 rather
    than only a mean, which would hide the long tail caused by
    variable-length batches under dynamic padding.
    """

    batch_seconds: list[float] = field(default_factory=list)
    n_documents: int = 0
    _split_start: float | None = None
    total_seconds: float = 0.0

    def start_split(self) -> None:
        self._split_start = time.perf_counter()

    def record_batch(self, seconds: float, n_docs: int) -> None:
        self.batch_seconds.append(float(seconds))
        self.n_documents += int(n_docs)

    def end_split(self) -> None:
        if self._split_start is not None:
            self.total_seconds = time.perf_counter() - self._split_start
            self._split_start = None

    def summary(self) -> dict[str, Any]:
        if not self.batch_seconds:
            return {
                "n_batches": 0,
                "n_documents": self.n_documents,
                "total_seconds": round(self.total_seconds, 3),
                "throughput_docs_per_second": None,
            }

        ordered = sorted(self.batch_seconds)
        n = len(ordered)

        def _percentile(p: float) -> float:
            idx = min(n - 1, max(0, int(round((p / 100.0) * (n - 1)))))
            return ordered[idx]

        total = self.total_seconds if self.total_seconds > 0 else sum(self.batch_seconds)
        return {
            "n_batches": n,
            "n_documents": self.n_documents,
            "total_seconds": round(total, 3),
            "total_human": human_duration(total),
            "batch_seconds_mean": round(sum(ordered) / n, 5),
            "batch_seconds_median": round(_percentile(50), 5),
            "batch_seconds_p95": round(_percentile(95), 5),
            "batch_seconds_min": round(ordered[0], 5),
            "batch_seconds_max": round(ordered[-1], 5),
            "throughput_docs_per_second": round(self.n_documents / total, 2) if total > 0 else None,
        }


def human_bytes(n: float | None) -> str | None:
    if n is None:
        return None
    n = float(n)
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def human_duration(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.0f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes}m {sec:.0f}s"
