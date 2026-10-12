#!/usr/bin/env python
"""Module 10 Stage 1 — inference on one news article. CPU or GPU.

Frozen end to end: no training, no fitting, no write to any Module 1-9 artifact.
Fusion parameters are LOADED from Module 7's validation artifact; they are not
literals anywhere in this module.

    python scripts/m10_01_inference.py --text "Your news article here"
    python scripts/m10_01_inference.py --title "Headline" --text-file article.txt
    python scripts/m10_01_inference.py --text-file a.txt --language ta --no-explain
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.m10_common import (  # noqa: E402
    PROJECT_ROOT, banner, load_m10_config, m10_dir, provenance_block,
    set_all_seeds, sha256_file, write_json, write_manifest)
from src.inference.pipeline import run_inference  # noqa: E402
from src.inference.report import render_report, validate_provenance  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyse one news article (frozen models).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="Article text.")
    src.add_argument("--text-file", help="Path to a UTF-8 file holding the article.")
    ap.add_argument("--title", default=None, help="Headline, if separate from the body.")
    ap.add_argument("--language", default=None, choices=["en", "ta"],
                    help="Override detection. Default: detect from Tamil script ratio.")
    ap.add_argument("--no-explain", action="store_true",
                    help="Skip occlusion (it costs one forward pass per token).")
    ap.add_argument("--out", default=None, help="Write the JSON here instead of outputs/.")
    ap.add_argument("--quiet", action="store_true", help="JSON only, no text report.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args()

    root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    cfg = load_m10_config(args.config, root=root)
    set_all_seeds(int(cfg["reproducibility"]["seed"]))
    if not args.quiet:
        banner("Stage 1 inference")

    text = (Path(args.text_file).read_text(encoding="utf-8") if args.text_file
            else args.text)

    t0 = time.perf_counter()
    result = run_inference(text, root, cfg, title=args.title,
                           language=args.language, explain=not args.no_explain)

    broken = validate_provenance(result)
    if broken:
        raise SystemExit(
            "Reason sentences cite fields that do not exist in the result: "
            f"{broken}. Refusing to emit an explanation that cannot be traced.")

    m7cfg_p = root / cfg["inputs"]["module7_config"]
    fusion_p = root / cfg["inputs"]["fusion_parameters"]
    result["provenance"].update(provenance_block(cfg, {
        "fusion_parameters_file": str(fusion_p.relative_to(root)),
        "fusion_parameters_sha256": sha256_file(fusion_p) if fusion_p.is_file() else None,
        "module7_config": str(m7cfg_p.relative_to(root)),
        "wall_seconds": round(time.perf_counter() - t0, 3),
        "fitting_performed": False,
        "note": ("All decision parameters were loaded from Module 7's frozen "
                 "validation artifact. Nothing was fitted during inference."),
    }))

    rid = result.get("claim", {}).get("claim_id", "inference:unknown")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    slug = rid.split(":")[-1][:16]
    out_dir = m10_dir(cfg, "inference", root)
    out_p = Path(args.out) if args.out else out_dir / f"{stamp}_{slug}.json"
    write_json(out_p, result)

    if cfg["output"].get("save_text_report", True) and not args.quiet:
        txt = render_report(result)
        out_p.with_suffix(".txt").write_text(txt, encoding="utf-8")
        print("\n" + txt)

    write_manifest(m10_dir(cfg, "manifests", root), {
        "stage": "m10_01_inference", "status": "complete",
        "result_file": str(out_p.relative_to(root)) if out_p.is_relative_to(root)
        else str(out_p),
        "language": result.get("language", {}).get("code"),
        "display_state": result.get("fusion", {}).get("display_state"),
        "accepted": result.get("input", {}).get("accepted"),
        "provenance": result.get("provenance")})

    if not args.quiet:
        print(f"\n  wrote -> {out_p}")


if __name__ == "__main__":
    main()
