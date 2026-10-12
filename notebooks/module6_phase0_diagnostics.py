#!/usr/bin/env python
"""Module 6 — PHASE 0 DIAGNOSTICS.  STRICTLY READ-ONLY.

Paste this whole file into ONE Colab cell and run it.

It writes NOTHING. It creates no files, renames nothing, deletes nothing,
trains nothing and loads no dataset into the verifier. It only opens files
for reading and prints a report.

Answers three questions the audit could not settle from Drive metadata alone:

  C0   Are the four Module 6 checkpoints actually readable zip archives,
       and do their on-mount sizes match what Drive recorded?
  C1   Was the X-FACT verifier really initialised from the FEVER encoder,
       i.e. did `--init-from models/module6/fever` actually run?
  C2b  What is the language composition of the evidence pool? This decides
       whether the English-only vs multilingual evidence ablation is
       meaningful or degenerate.

Run this BEFORE any repair step.
"""

import json
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path("/content/drive/MyDrive/Fake_News_Detection")

# Byte sizes Google Drive recorded for each checkpoint (audit evidence,
# 2026-08-21). A smaller size on the mount means the FUSE layer served a
# truncated stream -- that is a mount problem, not on-disk corruption.
EXPECTED_BYTES = {
    "models/module6/fever/best.pt":   3_336_812_189,
    "models/module6/fever/latest.pt": 3_336_813_433,
    "models/module6/x_fact/best.pt":  3_336_811_549,
    "models/module6/x_fact/latest.pt": 3_336_813_561,
}

line = "=" * 78
print(line)
print("MODULE 6 - PHASE 0 DIAGNOSTICS (READ-ONLY)")
print(line)
print(f"project root : {ROOT}")
print(f"exists       : {ROOT.is_dir()}")
if not ROOT.is_dir():
    sys.exit("Drive not mounted, or the project root is wrong. Mount Drive first.")


# ----------------------------------------------------------------- C0
print(f"\n{line}\nC0  CHECKPOINT INTEGRITY\n{line}")
c0 = {}
for rel, expected in EXPECTED_BYTES.items():
    p = ROOT / rel
    print(f"\n{rel}")
    if not p.is_file():
        print("  MISSING")
        c0[rel] = "missing"
        continue
    size = p.stat().st_size
    match = "OK" if size == expected else f"MISMATCH (Drive recorded {expected:,})"
    print(f"  size on mount : {size:,}  [{match}]")
    try:
        with zipfile.ZipFile(p) as z:
            members = z.namelist()
        print(f"  zip archive   : OK, {len(members)} members")
        c0[rel] = "ok" if size == expected else "size_mismatch"
    except Exception as e:                                   # noqa: BLE001
        print(f"  zip archive   : FAILED -> {type(e).__name__}: {e}")
        print("  HINT: copy to local disk and retry -- Drive FUSE often serves")
        print("        3 GB files incompletely:")
        print(f"        !cp '{p}' /content/tmp.pt  &&  then re-test /content/tmp.pt")
        c0[rel] = "zip_failed"

print(f"\n  C0 SUMMARY: {c0}")


# ----------------------------------------------------------------- C1
print(f"\n{line}\nC1  DID FEVER -> X-FACT TRANSFER ACTUALLY HAPPEN?\n{line}")
print("""
Logic of this test
------------------
The X-FACT training manifest is missing, so `init_from` was never recorded.
We infer it from the weights instead, using a THIRD reference point: a fresh
pretrained xlm-roberta-base.

  d(fever,  base)  = how far FEVER training moved the encoder
                     (50,000 claims x 3 epochs -> expect a clear drift)
  d(x_fact, base)  = how far the X-FACT encoder sits from raw pretrained
  d(fever, x_fact) = distance between the two fine-tuned encoders

  If --init-from WAS used : x_fact inherited all of FEVER's drift, so
                            d(x_fact,base) is LARGE, comparable to
                            d(fever,base), and d(fever,x_fact) is SMALL.
  If it was NOT used      : x_fact was fine-tuned from base on only 1,096
                            examples, which barely moves a 278M encoder, so
                            d(x_fact,base) is NEAR ZERO and much smaller
                            than d(fever,base).

The ratio d(x_fact,base) / d(fever,base) is the decisive number.
""")

try:
    import torch
    from transformers import AutoModel

    PROBES = [
        "encoder.encoder.layer.0.attention.self.query.weight",
        "encoder.encoder.layer.6.attention.self.query.weight",
        "encoder.encoder.layer.11.output.dense.weight",
        "encoder.embeddings.word_embeddings.weight",
    ]

    def probe_tensors(ckpt_path):
        """Load a checkpoint, keep ONLY the probe tensors, free the rest."""
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = payload["model_state"]
        out = {k: sd[k].detach().float().clone() for k in PROBES if k in sd}
        head = sorted(k for k in sd if k.startswith("head."))
        train_state = payload.get("train_state", {})
        del payload, sd
        import gc; gc.collect()
        return out, head, train_state

    print("  loading models/module6/fever/latest.pt ...")
    fever_t, fever_head, fever_state = probe_tensors(ROOT / "models/module6/fever/latest.pt")
    print("  loading models/module6/x_fact/best.pt ...")
    xfact_t, xfact_head, xfact_state = probe_tensors(ROOT / "models/module6/x_fact/best.pt")
    print("  loading pretrained xlm-roberta-base for reference ...")
    base_model = AutoModel.from_pretrained("xlm-roberta-base")
    base_sd = {f"encoder.{k}": v for k, v in base_model.state_dict().items()}
    base_t = {k: base_sd[k].detach().float().clone() for k in PROBES if k in base_sd}
    del base_model, base_sd

    def rel_delta(a, b):
        return ((a - b).norm() / b.norm()).item()

    print(f"\n  {'probe tensor':<52} {'d(fev,base)':>12} {'d(xf,base)':>12} {'d(fev,xf)':>11}")
    ratios = []
    for k in PROBES:
        if k not in fever_t or k not in xfact_t or k not in base_t:
            print(f"  {k:<52}  -- not present in all three --")
            continue
        d_fb = rel_delta(fever_t[k], base_t[k])
        d_xb = rel_delta(xfact_t[k], base_t[k])
        d_fx = rel_delta(fever_t[k], xfact_t[k])
        ratios.append(d_xb / d_fb if d_fb > 0 else float("nan"))
        print(f"  {k:<52} {d_fb:12.6f} {d_xb:12.6f} {d_fx:11.6f}")

    if ratios:
        mean_ratio = sum(ratios) / len(ratios)
        print(f"\n  mean ratio d(x_fact,base) / d(fever,base) = {mean_ratio:.4f}")
        if mean_ratio > 0.60:
            verdict = ("TRANSFER CONFIRMED -- the X-FACT encoder carries FEVER's drift. "
                       "`--init-from models/module6/fever` was used.")
        elif mean_ratio < 0.20:
            verdict = ("NO TRANSFER -- the X-FACT encoder sits close to raw pretrained "
                       "weights. It was NOT initialised from FEVER. Drop every "
                       "transfer-learning claim, and re-run Stage 6b with --init-from.")
        else:
            verdict = ("INCONCLUSIVE -- ratio falls between the thresholds. Report as "
                       "unverified; do not claim transfer.")
        print(f"  VERDICT: {verdict}")

    print(f"\n  fever head keys  : {fever_head}")
    print(f"  x_fact head keys : {xfact_head}")
    print(f"  fever  train_state: epoch={fever_state.get('epoch')} "
          f"step={fever_state.get('global_step')} best={fever_state.get('best_metric')}")
    print(f"  x_fact train_state: epoch={xfact_state.get('epoch')} "
          f"step={xfact_state.get('global_step')} best={xfact_state.get('best_metric')} "
          f"best_step={xfact_state.get('best_step')}")

except Exception as e:                                       # noqa: BLE001
    print(f"  C1 COULD NOT RUN -> {type(e).__name__}: {e}")
    print("  If this is an OOM, restart the runtime and run C1 on its own.")


# ----------------------------------------------------------------- C2b
print(f"\n{line}\nC2b  EVIDENCE LANGUAGE CENSUS\n{line}")
print("""
Decision rule, fixed IN ADVANCE so the choice cannot be made after seeing
the numbers:
    minority language >= 5% of evidence items  ->  run ablations 3/4
    minority language <  5%                    ->  degenerate; drop and say why
""")
for ds in ("x_fact", "fever"):
    ev_p = ROOT / "datasets/fact_verification" / ds / "evidence.jsonl"
    cl_p = ROOT / "datasets/fact_verification" / ds / "claims.jsonl"
    sp_p = ROOT / "datasets/fact_verification" / ds / "claim_splits.json"
    print(f"\n  --- {ds} ---")
    if not ev_p.is_file():
        print(f"  MISSING: {ev_p}")
        continue

    lang, url_like, n, total_chars = Counter(), 0, 0, 0
    with ev_p.open(encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            e = json.loads(ln)
            n += 1
            lang[e.get("language", "unknown")] += 1
            t = (e.get("text") or "").strip()
            total_chars += len(t)
            if t.startswith(("http://", "https://", "www.")):
                url_like += 1
    print(f"  evidence items      : {n:,}")
    print(f"  language counts     : {dict(lang)}")
    for lg, c in lang.most_common():
        print(f"      {lg:<10} {c:>8,}  ({100*c/n:5.2f} %)")
    minority = 100 * min(lang.values()) / n if len(lang) > 1 else 0.0
    print(f"  minority share      : {minority:.2f} %  -> "
          f"{'ABLATION 3/4 MEANINGFUL' if minority >= 5 else 'DEGENERATE - DROP IT'}")
    print(f"  mean evidence chars : {total_chars/max(n,1):.0f}")
    print(f"  URL-like items      : {url_like:,} ({100*url_like/max(n,1):.2f} %)")
    if url_like > 0.5 * n:
        print("  WARNING: most evidence 'texts' look like bare URLs. A cross-encoder")
        print("           cannot verify a claim against a URL string. If so, the")
        print("           evidence-aware repair will not help and the corpus needs")
        print("           re-examination BEFORE spending GPU time.")

    # train-split label distribution (previously unverified for x_fact)
    if cl_p.is_file() and sp_p.is_file():
        splits = json.loads(sp_p.read_text(encoding="utf-8")).get("splits", {})
        labels = [json.loads(l).get("label") for l in cl_p.open(encoding="utf-8") if l.strip()]
        for sname in ("train", "validation", "test"):
            idxs = splits.get(sname, [])
            c = Counter(labels[i] for i in idxs if i < len(labels))
            print(f"  {sname:<11} n={len(idxs):>7,}  {dict(c)}")

print(f"\n{line}\nPHASE 0 COMPLETE - NOTHING WAS WRITTEN, RENAMED OR DELETED.\n{line}")
