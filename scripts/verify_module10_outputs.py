#!/usr/bin/env python
"""Module 10 verification. Read-only.

Checks a stored inference result against the invariants Module 10 claims to
hold. Naming follows the project convention (verify_module3/4/6_outputs.py),
not an m10_02_* stage number.

    python scripts/verify_module10_outputs.py --result outputs/module10/inference/x.json
    python scripts/verify_module10_outputs.py --latest
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.m10_common import (  # noqa: E402
    PROJECT_ROOT, banner, load_m10_config, m10_dir)
from src.inference.report import validate_provenance  # noqa: E402

FEVER = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}
X_FACT = {"false", "true", "partly true/misleading"}


def verify(result: dict, root: Path, cfg: dict) -> list[dict]:
    checks: list[dict] = []

    def add(name: str, passed: bool | None, detail: str) -> None:
        checks.append({"name": name,
                       "status": "SKIP" if passed is None else ("PASS" if passed else "FAIL"),
                       "detail": detail})

    # ---- schema ----
    required = ["schema_version", "input", "language", "claim", "classifier",
                "retrieval", "verification", "fusion", "reason", "limitations",
                "provenance"]
    missing = [k for k in required if k not in result]
    add("schema_fields_present", not missing, f"missing={missing}")

    if not result.get("input", {}).get("accepted"):
        add("input_rejected_cleanly", "error" in result,
            f"rejected: {result.get('error')}")
        return checks

    f, c, v, r = (result["fusion"], result["classifier"],
                  result["verification"], result["retrieval"])

    # ---- probabilities ----
    probs = {"classifier.p_fake": c["p_fake"], "classifier.p_real": c["p_real"],
             "fusion.fused_p_fake": f["fused_p_fake"],
             "fusion.confidence": f["confidence"]}
    bad = {k: x for k, x in probs.items() if not (0.0 <= float(x) <= 1.0)}
    add("probabilities_in_unit_interval", not bad, f"out of range: {bad}")
    add("classifier_probs_sum_to_one",
        abs(c["p_fake"] + c["p_real"] - 1.0) < 1e-9,
        f"p_fake+p_real={c['p_fake'] + c['p_real']:.12f}")
    add("fused_probs_sum_to_one",
        abs(f["fused_p_fake"] + f["fused_p_real"] - 1.0) < 1e-9,
        f"sum={f['fused_p_fake'] + f['fused_p_real']:.12f}")

    # ---- verdict consistency ----
    expected = 1 if f["fused_p_fake"] >= f["threshold"] else 0
    add("verdict_consistent_with_threshold", f["final_prediction"] == expected,
        f"fused={f['fused_p_fake']:.6f} thr={f['threshold']} -> "
        f"expected {expected}, got {f['final_prediction']}")
    band = float(f["uncertain_band"])
    exp_state = ("UNCERTAIN" if abs(f["fused_p_fake"] - f["threshold"]) < band
                 else ("FAKE" if expected == 1 else "REAL"))
    add("display_state_matches_rule", f["display_state"] == exp_state,
        f"expected {exp_state}, got {f['display_state']}")
    add("confidence_is_not_p_fake",
        f["display_state"] == "UNCERTAIN" or
        abs(f["confidence"] - f["fused_p_fake"]) > 1e-9 or f["fused_p_fake"] in (0.0, 1.0),
        f"confidence={f['confidence']:.6f} fused_p_fake={f['fused_p_fake']:.6f}")

    # ---- fusion parameters came from Module 7's artifact ----
    fp = root / cfg["inputs"]["fusion_parameters"]
    add("fusion_params_file_exists", fp.is_file(), str(fp))
    if fp.is_file():
        doc = json.loads(fp.read_text(encoding="utf-8"))["parameters"]
        same = all(abs(float(doc[k]) - float(f[k])) < 1e-12
                   for k in ("alpha", "threshold", "temperature") if k in doc)
        add("fusion_params_match_artifact", same,
            f"artifact alpha={doc.get('alpha')} thr={doc.get('threshold')} "
            f"T={doc.get('temperature')} | result alpha={f['alpha']} "
            f"thr={f['threshold']} T={f['temperature']}")
        add("fusion_fitted_on_validation", doc.get("fit_split") == "validation",
            f"fit_split={doc.get('fit_split')}")
    add("no_fitting_during_inference",
        result["provenance"].get("fitting_performed") is False,
        f"fitting_performed={result['provenance'].get('fitting_performed')}")

    # ---- language and Unicode ----
    add("language_valid", result["language"]["code"] in ("en", "ta"),
        f"language={result['language']['code']}")
    txt = result["input"]["text"]
    add("no_replacement_characters", "�" not in txt,
        f"U+FFFD count={txt.count(chr(0xFFFD))}")
    add("claim_text_nfc_stable",
        unicodedata.normalize("NFC", result["claim"]["text"]) == result["claim"]["text"],
        "claim text is NFC-stable")

    # ---- verifier label space ----
    if v.get("available"):
        space = set(v["label_space"])
        expected_space = FEVER if v["dataset"] == "fever" else X_FACT
        add("verifier_label_space_correct", space == expected_space,
            f"{v['dataset']} -> {sorted(space)}")
        add("verifier_verdict_in_own_space", v["verdict"] in expected_space,
            f"verdict={v['verdict']!r}")
        add("no_cross_dataset_label_leak",
            not (v["dataset"] == "x_fact" and space & FEVER),
            "X-FACT result must never carry FEVER labels")
        s = sum(v["probs"].values())
        add("verifier_probs_normalised", abs(s - 1.0) < 1e-5, f"sum={s:.8f}")
    else:
        add("verifier_label_space_correct", None, f"unavailable: {v.get('reason')}")
        add("verifier_verdict_in_own_space", None, "unavailable")
        add("no_cross_dataset_label_leak", None, "unavailable")
        add("verifier_probs_normalised", None, "unavailable")

    # ---- evidence provenance, no fabrication ----
    cands = r.get("candidates", [])
    if cands:
        need = {"evidence_id", "corpus", "score", "rank", "text", "retriever"}
        incomplete = [e.get("rank") for e in cands if not need <= set(e)]
        add("evidence_provenance_complete", not incomplete,
            f"items missing fields: {incomplete}")
        add("evidence_count_matches", r["n_retrieved"] == len(cands),
            f"n_retrieved={r['n_retrieved']} len={len(cands)}")
        add("evidence_within_top_k", len(cands) <= r["top_k"],
            f"{len(cands)} <= {r['top_k']}")
        if v.get("available"):
            ret_ids = {e["evidence_id"] for e in cands}
            used = set(v.get("evidence_ids", []))
            add("verified_evidence_was_retrieved", used <= ret_ids,
                f"verifier used ids not in the retrieved set: {sorted(used - ret_ids)}")
        else:
            add("verified_evidence_was_retrieved", None, "no verifier output")
    else:
        add("no_evidence_reported_honestly", r["status"] == "NO EVIDENCE RETRIEVED",
            f"status={r['status']}")
        add("evidence_provenance_complete", None, "no candidates")
        add("evidence_count_matches", r["n_retrieved"] == 0, f"{r['n_retrieved']}")
        add("evidence_within_top_k", None, "no candidates")
        add("verified_evidence_was_retrieved", None, "no candidates")

    # ---- explainability ----
    x = result.get("explainability", {})
    if x.get("method"):
        add("explanation_method_labelled_occlusion", x["method"] == "occlusion",
            f"method={x['method']}")
        add("explanation_target_is_classifier",
            "classifier" in str(x.get("target", "")).lower(), f"target={x.get('target')}")
        add("attribution_length_matches_tokens",
            len(x.get("tokens", [])) == len(x.get("attributions", [])),
            f"{len(x.get('tokens', []))} tokens, {len(x.get('attributions', []))} attrs")
        pos_ok = all(t["attribution"] > 0 for t in x.get("top_positive_tokens", []))
        neg_ok = all(t["attribution"] < 0 for t in x.get("top_negative_tokens", []))
        add("attribution_sign_semantics_correct", pos_ok and neg_ok,
            "positive tokens must have attribution>0, negative <0")
    else:
        for n in ("explanation_method_labelled_occlusion",
                  "explanation_target_is_classifier",
                  "attribution_length_matches_tokens",
                  "attribution_sign_semantics_correct"):
            add(n, None, "explanation not computed")

    # ---- reason provenance ----
    broken = validate_provenance(result)
    add("reason_provenance_resolves", not broken, f"broken={broken}")
    add("reason_has_no_free_text_generator",
        "no language model" in result["reason"].get("generator", ""),
        result["reason"].get("generator", ""))

    # ---- encoder provenance ----
    enc = result["provenance"].get("encoder", {})
    add("encoder_is_xlm_roberta_base", enc.get("encoder") == "xlm-roberta-base",
        f"encoder={enc.get('encoder')}")
    add("encoder_pooling_is_mean", enc.get("pooling") == "mean",
        f"pooling={enc.get('pooling')}")
    add("encoder_is_l2_normalised", enc.get("normalized") is True,
        f"normalized={enc.get('normalized')}")
    add("encoder_max_length_512", int(enc.get("max_length", -1)) == 512,
        f"max_length={enc.get('max_length')}")
    sc = enc.get("self_check") or {}
    add("encoder_self_check_passed", sc.get("passed") is True,
        f"cosine={sc.get('cosine')} |dP|={sc.get('abs_p_fake_delta')}")
    add("embedding_is_unit_norm", abs(c.get("embedding_l2_norm", 0) - 1.0) < 1e-3,
        f"||v||={c.get('embedding_l2_norm')}")

    return checks


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify a Module 10 inference result.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--result", help="Path to an inference JSON.")
    g.add_argument("--latest", action="store_true", help="Newest file in outputs/.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--project-root", default=None)
    args = ap.parse_args()

    root = Path(args.project_root) if args.project_root else PROJECT_ROOT
    cfg = load_m10_config(args.config, root=root)
    banner("verification")

    if args.latest:
        files = sorted(m10_dir(cfg, "inference", root).glob("*.json"),
                       key=lambda p: p.stat().st_mtime)
        if not files:
            raise SystemExit("No inference results found. Run m10_01_inference.py first.")
        path = files[-1]
    else:
        path = Path(args.result)
    print(f"  result: {path}")

    checks = verify(json.loads(path.read_text(encoding="utf-8")), root, cfg)
    for ch in checks:
        print(f"  [{ch['status']:<4}] {ch['name']:<40} {ch['detail'][:80]}")

    n_pass = sum(1 for c in checks if c["status"] == "PASS")
    n_fail = sum(1 for c in checks if c["status"] == "FAIL")
    n_skip = sum(1 for c in checks if c["status"] == "SKIP")
    print(f"\n  {n_pass} pass / {n_fail} fail / {n_skip} skip")
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
