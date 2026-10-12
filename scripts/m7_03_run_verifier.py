#!/usr/bin/env python
"""
Module 7 Stage 3 — run the Module 6 verifier over project claims. GPU.

BLOCKED until Module 6 Stage 6 produces a verifier checkpoint. This stage
REFUSES to run without one rather than falling back to an untrained head: an
untrained XLM-R classifier emits arbitrary logits, and fusing those would
produce numbers that look like results and mean nothing.

    python scripts/m7_03_run_verifier.py --split validation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from scripts.m6_common import load_config as load_m6_config  # noqa: E402
from scripts.m6_common import load_corpus  # noqa: E402
from scripts.m7_common import (  # noqa: E402
    PROJECT_ROOT,
    banner,
    detect_device,
    load_m7_config,
    m7_dir,
    reproducibility_block,
    set_all_seeds,
    sha256_file,
    write_json,
    write_manifest,
)
from src.verification.dataset import get_label_space  # noqa: E402
from src.verification.trainer import CheckpointManager  # noqa: E402
from src.verification.verifier_model import (  # noqa: E402
    VerifierConfig,
    build_verifier,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Verify project claims with the Module 6 verifier."
    )
    ap.add_argument(
        "--split",
        required=True,
        choices=["validation", "test"],
    )
    ap.add_argument("--config", default=None)
    ap.add_argument("--project-root", default=None)
    ap.add_argument(
        "--check-only",
        action="store_true",
        help="Report dependency status and exit without loading a model.",
    )
    args = ap.parse_args()

    root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    cfg = load_m7_config(args.config, root=root)
    m6cfg = load_m6_config(
        root / cfg["inputs"]["module6_config"]
    )

    banner(f"Stage 3 verification [{args.split}]")
    set_all_seeds(int(cfg["reproducibility"]["seed"]))

    pools = cfg["inputs"]["evidence_datasets"]
    status = {}

    for lang, dataset in pools.items():

        # Explicitly select the intended Module 6 checkpoint.
        if dataset == "x_fact":
            ck = (
                root
                / "models/module6/ablations/x_fact"
                / "xfact_evidence_dense_transfer"
            )
        elif dataset == "fever":
            ck = root / "models/module6/fever"
        else:
            raise RuntimeError(
                f"Unsupported verifier dataset: {dataset}"
            )

        best = ck / "best.pt"
        latest = ck / "latest.pt"

        have = best.is_file() or latest.is_file()

        selected_path = (
            best if best.is_file() else latest
        )

        status[lang] = {
            "dataset": dataset,
            "checkpoint_dir": str(ck),
            "checkpoint_present": have,
            "path": str(selected_path) if have else None,
        }

        print(
            f"  [{lang}] {dataset} verifier checkpoint: "
            f"{'FOUND' if have else 'MISSING'}"
        )

        if have:
            print(
                f"      USING CHECKPOINT: {selected_path}"
            )

    missing = [
        lang
        for lang, s in status.items()
        if not s["checkpoint_present"]
    ]

    if missing or args.check_only:
        out = m7_dir(cfg, "verification", root)

        write_manifest(
            out,
            {
                "stage": "m7_03_run_verifier",
                "split": args.split,
                "status": (
                    "blocked"
                    if missing
                    else "check_only"
                ),
                "checkpoint_status": status,
                "blocking_dependency": (
                    "Module 6 Stage 6 verifier training "
                    f"({', '.join(missing)})"
                    if missing
                    else None
                ),
                "reproducibility": reproducibility_block(cfg),
            },
        )

        if missing:
            raise SystemExit(
                "\nBLOCKED: Module 6 Stage 6 has not produced "
                "a verifier checkpoint for: "
                f"{missing}.\n"
                "  Module 7 will NOT substitute an untrained verifier: "
                "arbitrary logits\n"
                "  fused into a decision would look like a result "
                "and mean nothing.\n"
                "  Run first:\n"
                "    python scripts/m6_06_train_verifier.py "
                "--dataset fever\n"
                "    python scripts/m6_06_train_verifier.py "
                "--dataset x_fact "
                "--init-from models/module6/fever"
            )

        return

    import torch
    from transformers import AutoTokenizer

    claims_doc = json.loads(
        (
            m7_dir(cfg, "claims", root)
            / f"{args.split}_claims.json"
        ).read_text(encoding="utf-8")
    )

    retr_doc = json.loads(
        (
            m7_dir(cfg, "retrieval", root)
            / f"{args.split}_retrieval.json"
        ).read_text(encoding="utf-8")
    )

    primary = cfg["retrieval"]["primary"]
    v = m6cfg["verifier"]

    out_records: list[dict] = []

    for lang, dataset in pools.items():
        subset = [
            c
            for c in claims_doc["claims"]
            if c["language"] == lang
        ]

        if not subset:
            continue

        space = get_label_space(dataset)
        corpus = load_corpus(
            m6cfg,
            dataset,
            root=root,
        )

        vcfg = VerifierConfig(
            encoder_name=v["encoder"],
            num_labels=len(space),
            hidden_dim=int(v["hidden_dim"]),
            dropout=float(v["dropout"]),
            max_length=int(v["max_length"]),
            label_space=space.name,
        )

        model = build_verifier(vcfg)

        cm = CheckpointManager(
            Path(status[lang]["checkpoint_dir"])
        )

        if (
            cm.load(
                model,
                path=cm.best if cm.best.is_file() else None,
            )
            is None
        ):
            raise SystemExit(
                f"could not load checkpoint for {dataset}"
            )

        dev = detect_device()
        model.to(dev).eval()

        tok = AutoTokenizer.from_pretrained(
            v["encoder"]
        )

        lang_res = retr_doc["results"].get(
            lang,
            {},
        )

        with torch.inference_mode():

            for i in range(
                0,
                len(subset),
                int(v["batch_size"]),
            ):
                chunk = subset[
                    i : i + int(v["batch_size"])
                ]

                texts = []
                evid = []

                for c in chunk:
                    r = (
                        lang_res.get(
                            c["claim_id"],
                            {},
                        )
                        or {}
                    ).get(
                        primary,
                        {},
                    )

                    ids = (
                        r.get("evidence_ids")
                        or []
                    )[
                        : int(
                            v["evidence_per_claim"]
                        )
                    ]

                    evid.append(ids)

                    texts.append(
                        " </s> ".join(
                            corpus[
                                "evidence_texts"
                            ].get(e, "")
                            for e in ids
                        )
                    )

                enc = tok(
                    [c["text"] for c in chunk],
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=int(
                        v["max_length"]
                    ),
                    return_tensors="pt",
                ).to(dev)

                probs = (
                    torch.softmax(
                        model(
                            enc["input_ids"],
                            enc["attention_mask"],
                        ),
                        dim=-1,
                    )
                    .float()
                    .cpu()
                    .numpy()
                )

                for c, p, ids in zip(
                    chunk,
                    probs,
                    evid,
                ):
                    out_records.append(
                        {
                            "claim_id": c["claim_id"],
                            "record_id": c["record_id"],
                            "language": lang,
                            "dataset": dataset,
                            "label_space": space.labels,
                            "probs": {
                                lab: float(x)
                                for lab, x in zip(
                                    space.labels,
                                    p,
                                )
                            },
                            "verdict": space.decode(
                                int(
                                    np.argmax(p)
                                )
                            ),
                            "evidence_ids": ids,
                            "has_evidence": bool(ids),
                        }
                    )

    out = m7_dir(
        cfg,
        "verification",
        root,
    )

    write_json(
        out / f"{args.split}_verification.json",
        {
            "split": args.split,
            "n_records": len(out_records),
            "primary_retriever": primary,
            "records": out_records,
        },
    )

    write_manifest(
        out,
        {
            "stage": "m7_03_run_verifier",
            "split": args.split,
            "status": "complete",
            "n_records": len(out_records),
            "checkpoint_status": status,
            "reproducibility": reproducibility_block(
                cfg
            ),
        },
    )

    print(
        f"\n  verified {len(out_records):,} claims -> {out}"
    )


if __name__ == "__main__":
    main()