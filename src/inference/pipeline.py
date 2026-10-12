"""Module 10 — inference pipeline. Frozen and read-only end to end.

Every stage delegates to the Module 1-9 implementation that produced the
research results. Nothing here fits, trains or writes a model artifact.

    text -> normalise -> language -> claim -> encode -> classify
         -> BM25 retrieve -> verify -> fuse (frozen params) -> attribute -> report

DEPENDENCY INJECTION. Each heavy stage is a callable with a real default, so
the orchestration can be tested without torch, checkpoints or a GPU -- the same
pattern `VerifierTrainer` uses for `step_fn`/`eval_fn`.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

FAKE_CLASS_INDEX = 1        # Module 2 label encoding; Module 4 predict_proba[:, 1]


class InferenceConfigError(RuntimeError):
    """A frozen artifact required for inference is missing or inconsistent."""


@dataclass
class Dependencies:
    """Injectable stages. None -> build the real implementation lazily."""
    encode_fn: Callable[[str], np.ndarray] | None = None
    classify_fn: Callable[[np.ndarray], float] | None = None
    retrieve_fn: Callable[[str, str], list[dict[str, Any]]] | None = None
    verify_fn: Callable[[str, list[dict[str, Any]], str], dict[str, Any] | None] | None = None
    encoder_provenance: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------- utilities
def validate_input(text: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Accept or reject. Never silently alters the submitted text."""
    v = cfg["input_validation"]
    raw = text if isinstance(text, str) else ""
    n = len(raw.strip())
    if n == 0:
        return {"accepted": False, "reason": "empty input", "n_chars": 0,
                "min_chars": int(v["min_chars"])}
    if n < int(v["min_chars"]):
        return {"accepted": False,
                "reason": f"too short: {n} characters, minimum {v['min_chars']}",
                "n_chars": n, "min_chars": int(v["min_chars"])}
    if n > int(v["max_chars"]):
        return {"accepted": False,
                "reason": f"too long: {n} characters, maximum {v['max_chars']}",
                "n_chars": n, "max_chars": int(v["max_chars"])}
    return {"accepted": True, "reason": None, "n_chars": n,
            "min_chars": int(v["min_chars"])}


def detect_language(text: str) -> dict[str, Any]:
    """Module 5's rule, reused unchanged: Tamil script ratio > 0.3 -> 'ta'."""
    from src.fact_verification.claim_selection import tamil_ratio
    r = float(tamil_ratio(text))
    return {"code": "ta" if r > 0.3 else "en", "tamil_ratio": round(r, 4),
            "method": "tamil_ratio>0.3 (src/fact_verification/claim_selection.py)"}


def select_claim(title: str | None, text: str, language: str | None,
                 root: Path, m6_cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Module 5's claim selector, reused unchanged."""
    from scripts.m6_common import load_config as load_m6
    from src.fact_verification.claim_selection import select_claim_from_record

    m5 = load_m6(root / m6_cfg["inputs"]["module5_config"])
    cfg = {"claim_selection": m5["claim_selection"], "normalization": m5["normalization"]}
    rid = "inference:" + hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]
    record = {"record_id": rid, "title": title or "", "text": text,
              "language": language or "", "source": "user_input"}
    claim = select_claim_from_record(record, cfg, dataset="inference")
    if claim is None:
        return None
    return {"claim_id": claim.claim_id, "text": claim.text, "raw_text": claim.raw_text,
            "language": claim.language, "selection_rule": claim.selection_rule,
            "metadata": claim.metadata}


def load_frozen_fusion(root: Path, m10_cfg: dict[str, Any]) -> Any:
    """Module 7's fitted parameters, loaded from the artifact. Never literals."""
    import json

    from src.integration.fusion import FusionParameters

    p = root / m10_cfg["inputs"]["fusion_parameters"]
    if not p.is_file():
        raise InferenceConfigError(
            f"Frozen fusion parameters not found: {p}\n"
            "  Module 10 will NOT substitute default or literal values: the "
            "decision threshold, alpha and temperature were fitted on Module 7's "
            "VALIDATION split and are the only admissible ones.\n"
            "  Run: python scripts/m7_04_fit_fusion.py")
    doc = json.loads(p.read_text(encoding="utf-8"))
    params = FusionParameters.from_dict(doc["parameters"])
    if getattr(params, "fit_split", None) != "validation":
        raise InferenceConfigError(
            f"Fusion parameters were fitted on {params.fit_split!r}, not 'validation'. "
            "Refusing to use them.")
    return params


def compute_confidence(fused_p: float, threshold: float) -> float:
    """Certainty of the DECISION, not a restatement of P(fake).

    How far into the decided region the probability sits, as a fraction of that
    region: 0.0 exactly at the threshold, 1.0 at either extreme. Defined in
    configs/module10_inference.yaml (display.confidence_definition).
    """
    p, t = float(fused_p), float(threshold)
    return float(min(1.0, max(0.0, (p - t) / (1.0 - t) if p >= t else (t - p) / t)))


def display_state(fused_p: float, threshold: float, band: float) -> str:
    """Threshold decides; the band only annotates. Never overrides."""
    if abs(float(fused_p) - float(threshold)) < float(band):
        return "UNCERTAIN"
    return "FAKE" if float(fused_p) >= float(threshold) else "REAL"


# ----------------------------------------------------------- real defaults
def build_default_dependencies(root: Path, m10_cfg: dict[str, Any]) -> Dependencies:
    """Construct the real stages once. Heavy imports stay inside."""
    import torch

    from src.contrastive.projection_head import build_ce_mlp_baseline
    from src.inference.text_encoder import InferenceEncoder
    from src.utils.config_loader import load_yaml

    m7 = load_yaml(root / m10_cfg["inputs"]["module7_config"])
    m6 = load_yaml(root / m10_cfg["inputs"]["module6_config"])
    inp = m7["inputs"]

    ckpt = root / inp["module4_models_dir"] / inp["module4_checkpoint"]
    if not ckpt.is_file():
        raise InferenceConfigError(f"Module 4 checkpoint not found: {ckpt}")
    clf = build_ce_mlp_baseline(input_dim=int(inp["module4_input_dim"]),
                                hidden_dims=list(inp["module4_hidden_dims"]),
                                n_classes=int(inp["module4_n_classes"]),
                                dropout=float(inp["module4_dropout"]))
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    missing, unexpected = clf.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise InferenceConfigError(
            f"Module 4 checkpoint does not match the architecture. "
            f"missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}")
    clf.eval()

    def classify_fn(vec: np.ndarray) -> float:
        with torch.inference_mode():
            x = torch.from_numpy(np.asarray(vec, dtype=np.float32).reshape(1, -1))
            return float(torch.softmax(clf(x), dim=-1)[0, FAKE_CLASS_INDEX])

    encoder = InferenceEncoder(root, m10_cfg, classify_fn=classify_fn, self_check=True)

    def encode_fn(text: str) -> np.ndarray:
        return encoder.encode(text)

    # ---- BM25 retrieval over the fixed pool, Module 7's configuration ----
    pools = m7["inputs"]["evidence_datasets"]
    ret_cfg, cache = m10_cfg["retrieval"], {}

    def retrieve_fn(claim_text: str, language: str) -> list[dict[str, Any]]:
        from scripts.m6_common import load_corpus
        from src.integration.fusion import rank_normalise
        from src.retrieval.bm25_baseline import BM25Retriever
        from src.retrieval.index_scope import build_evidence_scope

        dataset = pools.get(language)
        if dataset is None:
            return []
        if dataset not in cache:
            corpus = load_corpus(m6, dataset, root=root)
            scope = build_evidence_scope(corpus["claims"], corpus["splits"],
                                         ret_cfg["phase"], m6["leakage"]["index_scope"])
            ev_ids = sorted(scope.evidence_ids)
            cache[dataset] = (BM25Retriever(ev_ids,
                                            [corpus["evidence_texts"][e] for e in ev_ids],
                                            k1=float(m6["bm25"]["k1"]),
                                            b=float(m6["bm25"]["b"])),
                              corpus, len(ev_ids))
        bm, corpus, pool_size = cache[dataset]
        ids, scores = bm.retrieve(claim_text, int(ret_cfg["top_k"]))
        conf = rank_normalise(scores)
        return [{"rank": i + 1, "evidence_id": e, "corpus": dataset,
                 "pool_size": pool_size, "retriever": "bm25",
                 "score": float(s), "confidence": float(c),
                 "text": corpus["evidence_texts"].get(e, "")}
                for i, (e, s, c) in enumerate(zip(ids, scores, conf))]

    # ---- Module 6 verifier, per language, native label space --------------
    vcache: dict[str, Any] = {}

    def verify_fn(claim_text: str, evidence: list[dict[str, Any]],
                  language: str) -> dict[str, Any] | None:
        from transformers import AutoTokenizer

        from src.verification.dataset import get_label_space
        from src.verification.trainer import CheckpointManager
        from src.verification.verifier_model import VerifierConfig, build_verifier

        dataset = pools.get(language)
        if dataset is None or not evidence:
            return None
        if dataset not in vcache:
            ck = root / m7["inputs"]["module6_models_dir"] / dataset
            best, latest = ck / "best.pt", ck / "latest.pt"
            if not (best.is_file() or latest.is_file()):
                return None
            space = get_label_space(dataset)
            v = m6["verifier"]
            model = build_verifier(VerifierConfig(
                encoder_name=v["encoder"], num_labels=len(space),
                hidden_dim=int(v["hidden_dim"]), dropout=float(v["dropout"]),
                max_length=int(v["max_length"]), label_space=space.name))
            cm = CheckpointManager(ck)
            if cm.load(model, path=best if best.is_file() else latest) is None:
                return None
            model.eval()
            vcache[dataset] = (model, AutoTokenizer.from_pretrained(v["encoder"]),
                               space, v, str(best if best.is_file() else latest))
        model, tok, space, v, ckpt_path = vcache[dataset]

        n_ev = int(v["evidence_per_claim"])
        used = evidence[:n_ev]
        joined = " </s> ".join(e.get("text", "") for e in used)
        with torch.inference_mode():
            enc = tok([claim_text], [joined], padding=True, truncation=True,
                      max_length=int(v["max_length"]), return_tensors="pt")
            probs = torch.softmax(model(enc["input_ids"], enc["attention_mask"]),
                                  dim=-1).float().cpu().numpy()[0]
        return {"dataset": dataset, "label_space": list(space.labels),
                "probs": {lab: float(p) for lab, p in zip(space.labels, probs)},
                "verdict": space.decode(int(np.argmax(probs))),
                "verdict_probability": float(np.max(probs)),
                "n_evidence_used": len(used),
                "evidence_ids": [e["evidence_id"] for e in used],
                "checkpoint": ckpt_path}

    return Dependencies(encode_fn=encode_fn, classify_fn=classify_fn,
                        retrieve_fn=retrieve_fn, verify_fn=verify_fn,
                        encoder_provenance=encoder.provenance())


# ------------------------------------------------------------- the pipeline
def run_inference(text: str, root: Path, m10_cfg: dict[str, Any],
                  title: str | None = None, language: str | None = None,
                  deps: Dependencies | None = None,
                  explain: bool = True) -> dict[str, Any]:
    """One article -> the full structured result. Fits nothing."""
    from src.utils.config_loader import load_yaml

    t0 = time.perf_counter()
    m6 = load_yaml(root / m10_cfg["inputs"]["module6_config"])
    m7 = load_yaml(root / m10_cfg["inputs"]["module7_config"])
    deps = deps or build_default_dependencies(root, m10_cfg)

    result: dict[str, Any] = {"schema_version": "m10.1"}

    # 1. input ----------------------------------------------------------
    val = validate_input(text, m10_cfg)
    result["input"] = {**val, "text": text, "title": title}
    if not val["accepted"]:
        result["error"] = val["reason"]
        return result

    # 2. language -------------------------------------------------------
    lang = {"code": language, "tamil_ratio": None, "method": "user-supplied"} \
        if language in ("en", "ta") else detect_language(text)
    result["language"] = lang

    # 3. claim ----------------------------------------------------------
    claim = select_claim(title, text, lang["code"], root, m6)
    if claim is None:
        result["error"] = "claim selection returned no usable claim"
        result["claim"] = None
        return result
    result["claim"] = claim

    # 4. classifier -----------------------------------------------------
    vec = deps.encode_fn(text)
    p_clf = float(deps.classify_fn(vec))
    result["classifier"] = {
        "p_fake": p_clf, "p_real": 1.0 - p_clf,
        "prediction": int(p_clf >= 0.5),
        "prediction_label": "FAKE" if p_clf >= 0.5 else "REAL",
        "note": ("Classifier-only reading at 0.5. The FINAL decision uses the "
                 "frozen Module 7 threshold, which is not 0.5."),
        "embedding_l2_norm": float(np.linalg.norm(vec)),
        "embedding_dim": int(np.asarray(vec).shape[-1])}

    # 5. retrieval ------------------------------------------------------
    evidence = list(deps.retrieve_fn(claim["text"], lang["code"]) or [])
    pools = m7["inputs"]["evidence_datasets"]
    result["retrieval"] = {
        "pool": pools.get(lang["code"]), "retriever": m10_cfg["retrieval"]["primary"],
        "top_k": int(m10_cfg["retrieval"]["top_k"]), "n_retrieved": len(evidence),
        "status": "NO EVIDENCE RETRIEVED" if not evidence else "ok",
        "label": "CANDIDATE EVIDENCE — RETRIEVED AUTOMATICALLY",
        "candidates": evidence}

    # 6. verifier -------------------------------------------------------
    ver = deps.verify_fn(claim["text"], evidence, lang["code"]) if evidence else None
    result["verification"] = ({"available": False,
                               "reason": ("no evidence retrieved" if not evidence
                                          else "no verifier checkpoint for this pool"),
                               "dataset": pools.get(lang["code"])}
                              if ver is None else {"available": True, **ver})

    # 7. fusion ---------------------------------------------------------
    from src.integration.fusion import apply_fusion, verifier_to_fake_prob

    params = load_frozen_fusion(root, m10_cfg)
    if ver is not None:
        roles = (m7["fusion"]["fever_label_roles"] if ver["dataset"] == "fever"
                 else m7["fusion"]["x_fact_label_roles"])
        p = ver["probs"]
        p_ver = float(verifier_to_fake_prob(p[roles["fake"]], p[roles["real"]],
                                            p.get(roles["neutral"])))
        has_ver = True
    else:
        p_ver, has_ver = 0.5, False

    out = apply_fusion([p_clf], [p_ver], [has_ver], params)
    fused = float(out["p_fake"][0])
    pre_temp = (params.alpha * p_clf + (1 - params.alpha) * p_ver) if has_ver else p_clf
    band = float(m10_cfg["display"]["uncertain_band"])
    state = display_state(fused, params.threshold, band)

    result["fusion"] = {
        "alpha": float(params.alpha), "threshold": float(params.threshold),
        "temperature": float(params.temperature), "fit_split": params.fit_split,
        "params_source": m10_cfg["inputs"]["fusion_parameters"],
        "p_fake_classifier": p_clf, "p_fake_verifier": p_ver if has_ver else None,
        "has_verifier": has_ver,
        "fused_p_fake_before_temperature": float(pre_temp),
        "fused_p_fake": fused, "fused_p_real": 1.0 - fused,
        "final_prediction": int(out["prediction"][0]),
        "final_label": "FAKE" if int(out["prediction"][0]) == 1 else "REAL",
        "margin_to_threshold": float(fused - params.threshold),
        "confidence": compute_confidence(fused, params.threshold),
        "confidence_definition": m10_cfg["display"]["confidence_definition"],
        "uncertain_band": band, "display_state": state,
        "note": ("Temperature is applied in binary logit space. T < 1 sharpens, "
                 "so the fused probability is more extreme than the pre-temperature "
                 "value. Confidence is NOT P(fake).")}

    # 8. explainability -------------------------------------------------
    result["explainability"] = (_attribute(text, deps, m10_cfg) if explain
                                else {"method": None, "reason": "explain=False"})

    # 9. reason ---------------------------------------------------------
    from src.inference.report import build_reason_block
    result["reason"] = build_reason_block(result)
    result["limitations"] = _limitations(result, m7)
    result["provenance"] = {"encoder": deps.encoder_provenance,
                            "seconds": round(time.perf_counter() - t0, 3)}
    return result


def _attribute(text: str, deps: Dependencies, cfg: dict[str, Any]) -> dict[str, Any]:
    """Occlusion over the classifier's P(fake). Module 8's implementation."""
    from src.explainability.token_attribution import (
        deletion_test, insertion_test, occlusion_attribution)

    e = cfg["explainability"]

    def predict_fn(texts):
        return np.asarray([deps.classify_fn(deps.encode_fn(t)) for t in texts],
                          dtype=np.float64)

    attr = occlusion_attribution(text, predict_fn, mask_token=str(e["mask_token"]),
                                 max_tokens=int(e["max_tokens"]),
                                 batch_size=int(e["batch_size"]))
    top = attr.top_tokens(int(e["top_k_tokens"]))
    out = {"method": e["method"], "target": "classifier P(fake)",
           "mask_token": e["mask_token"], "n_tokens": attr.n_tokens,
           "truncated": attr.truncated, "base_probability": attr.base_probability,
           "tokens": attr.tokens, "attributions": attr.attributions,
           "error": attr.error, **top,
           "interpretation": {
               "positive": ("removing the token LOWERED P(fake) -> the token pushed "
                            "the classifier toward FAKE"),
               "negative": ("removing the token RAISED P(fake) -> the token pushed "
                            "the classifier toward REAL"),
               "scope": ("Explains the CLASSIFIER only. It does not explain the "
                         "verifier or the fused decision.")}}
    f = e.get("faithfulness", {})
    if f.get("enabled") and attr.n_tokens:
        fr = tuple(float(x) for x in f["fractions"])
        out["faithfulness"] = {"label": f["label"], "scope": "this input only",
                               "deletion": deletion_test(text, attr, predict_fn, fr),
                               "insertion": insertion_test(text, attr, predict_fn, fr)}
    return out


def _limitations(result: dict[str, Any], m7: dict[str, Any]) -> list[str]:
    """Measured facts, not boilerplate."""
    f, r = result.get("fusion", {}), result.get("retrieval", {})
    out = [
        "This is a model prediction, not a verified truth claim.",
        (f"Fusion weight alpha = {f.get('alpha')}: with verifier evidence available, "
         f"the classifier supplies about {100 * float(f.get('alpha', 1)):.0f}% of the "
         "pre-temperature fused probability."),
        (f"Decision threshold is {f.get('threshold')}, not 0.5. A fused P(fake) above "
         "0.5 but below the threshold is still REAL."),
        (f"Temperature T = {f.get('temperature')} is applied in binary logit space; "
         "T < 1 sharpens, so the displayed probability is more extreme than the "
         "pre-temperature value."),
        ("Candidate evidence is retrieved from a FIXED corpus "
         f"({r.get('pool')}), not from the live internet. An article about anything "
         "outside that corpus cannot have its evidence retrieved."),
        ("Token attributions explain the classifier's probability. They do not "
         "establish factual truth and do not explain the verifier."),
        ("The verifier was trained on FEVER/X-FACT, not on user-submitted news; "
         "its verdict is an auxiliary signal."),
    ]
    if not result.get("verification", {}).get("available"):
        out.append("No verifier output for this input: the decision rests on the "
                   "classifier alone, by Module 7's missing_verifier policy.")
    return out
