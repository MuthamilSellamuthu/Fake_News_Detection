#!/usr/bin/env python
"""Module 10 — Streamlit demonstration UI. Read-only over frozen artifacts.

    streamlit run scripts/m10_app.py

This is a PRESENTATION LAYER ONLY. It contains no inference mathematics: every
number shown comes from `src.inference.pipeline.run_inference`, which is the
same code path `scripts/m10_01_inference.py` uses. alpha, threshold and
temperature are loaded from Module 7's frozen validation artifact, never
written here.

Stage progress is produced by WRAPPING the injectable `Dependencies` callables
with thin pass-throughs that update the status widget. Nothing is recomputed or
duplicated -- the wrappers call the real functions and return their results
unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from scripts.m10_common import PROJECT_ROOT, load_m10_config, set_all_seeds  # noqa: E402
from src.inference.pipeline import (  # noqa: E402
    Dependencies, InferenceConfigError, build_default_dependencies, run_inference)

st.set_page_config(page_title="Fake News Detection System",
                   page_icon="🔍", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1100px;}
  .verdict-card {padding: 1.6rem; border-radius: 12px; text-align: center;
                 margin: 0.5rem 0 1.2rem 0;}
  .verdict-fake {background: #3b1113; border: 2px solid #b3242c;}
  .verdict-real {background: #0f2e18; border: 2px solid #1e8e3e;}
  .verdict-unc  {background: #33290b; border: 2px solid #b8860b;}
  .verdict-title {font-size: 2.4rem; font-weight: 700; margin: 0;}
  .verdict-sub {opacity: 0.85; margin-top: 0.3rem; font-size: 0.94rem;}
  .tok {display: inline-block; padding: 0.2rem 0.55rem; margin: 0.18rem;
        border-radius: 6px; font-family: monospace; font-size: 0.9rem;}
  .tok-pos {background: rgba(179,36,44,0.28); border: 1px solid #b3242c;}
  .tok-neg {background: rgba(30,142,62,0.25); border: 1px solid #1e8e3e;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;margin-bottom:0'>FAKE NEWS DETECTION SYSTEM</h1>",
            unsafe_allow_html=True)
st.markdown("<p style='text-align:center;opacity:0.75;margin-top:0.2rem'>"
            "Multilingual AI-Based News Verification &nbsp;·&nbsp; English &amp; Tamil"
            "</p>", unsafe_allow_html=True)
st.divider()


# ------------------------------------------------------------ model loading
@st.cache_resource(show_spinner=False)
def load_pipeline(root_str: str):
    root = Path(root_str)
    cfg = load_m10_config(root=root)
    set_all_seeds(int(cfg["reproducibility"]["seed"]))
    return cfg, build_default_dependencies(root, cfg)


root = Path(st.sidebar.text_input("Project root", str(PROJECT_ROOT)))

try:
    with st.spinner("Loading frozen models (first run only)…"):
        cfg, deps = load_pipeline(str(root))
except InferenceConfigError as exc:
    st.error("**Required model artifacts are missing. "
             "Please verify the project Drive files.**")
    st.code(str(exc), language="text")
    st.stop()
except Exception as exc:                                    # noqa: BLE001
    name = type(exc).__name__
    if name == "EncoderProvenanceError":
        st.error("**Encoder provenance self-check FAILED — inference refused.**")
        st.code(str(exc), language="text")
        st.info("This is a deliberate refusal, not a crash. The encoder could not "
                "reproduce a known stored embedding, which is how a silent "
                "encoder mismatch is prevented.")
    else:
        st.error("**Required model artifacts are missing. "
                 "Please verify the project Drive files.**")
        st.code(f"{name}: {exc}", language="text")
    st.stop()

enc = deps.encoder_provenance or {}
sc = enc.get("self_check") or {}
with st.sidebar:
    st.success("Encoder self-check PASSED")
    st.caption(
        f"`{enc.get('encoder')}` · {enc.get('pooling')} pooling · L2-normalised · "
        f"max_length {enc.get('max_length')}\n\n"
        f"cosine {sc.get('cosine', float('nan')):.8f}\n\n"
        f"|ΔP(fake)| {sc.get('abs_p_fake_delta', float('nan')):.2e}")
    st.divider()
    st.caption("All models are frozen. Nothing is trained or fitted by this app.")


# ------------------------------------------------------------------- input
st.subheader("Enter a news article")
title = st.text_input("Headline (optional)", placeholder="Optional headline")
body = st.text_area("News article / content", height=210,
                    placeholder="Paste the complete news article here "
                                "(English, Tamil, or mixed)…")
c1, c2, c3 = st.columns([2, 2, 3])
lang_choice = c1.selectbox("Language", ["Auto-detect", "English (en)", "Tamil (ta)"])
want_explain = c2.checkbox("Token explanation", value=True,
                           help="Occlusion: one forward pass per token (max 128). "
                                "Adds time but is required for the FAKE explanation panel.")
go = c3.button("🔍  VERIFY NEWS", type="primary", use_container_width=True)


def instrumented(deps: Dependencies, status) -> Dependencies:
    """Wrap the real callables so stage progress is visible.

    Pure pass-throughs: each wrapper updates the status label, calls the
    original function and returns its result unchanged. No mathematics here.
    """
    def wrap(fn, label):
        def inner(*a, **k):
            status.update(label=label)
            return fn(*a, **k)
        return inner

    return Dependencies(
        encode_fn=wrap(deps.encode_fn, "Generating XLM-RoBERTa embedding…"),
        classify_fn=wrap(deps.classify_fn, "Running the classifier…"),
        retrieve_fn=wrap(deps.retrieve_fn, "Retrieving candidate evidence…"),
        verify_fn=wrap(deps.verify_fn, "Running fact verification…"),
        encoder_provenance=deps.encoder_provenance)


if go:
    if not body.strip():
        st.warning("Paste a news article first.")
        st.stop()

    language = {"Auto-detect": None, "English (en)": "en", "Tamil (ta)": "ta"}[lang_choice]
    try:
        with st.status("Starting…", expanded=True) as status:
            res = run_inference(body, root, cfg, title=title or None,
                                language=language,
                                deps=instrumented(deps, status),
                                explain=want_explain)
            status.update(label="Analysis complete", state="complete", expanded=False)
    except InferenceConfigError as exc:
        st.error("**Required model artifacts are missing. "
                 "Please verify the project Drive files.**")
        st.code(str(exc), language="text")
        st.stop()

    if not res["input"]["accepted"]:
        st.error(f"Input rejected: {res['error']}")
        st.stop()

    f, clf = res["fusion"], res["classifier"]
    ver, ret, xai = res["verification"], res["retrieval"], res.get("explainability", {})
    state = f["display_state"]

    # ------------------------------------------------------- final verdict
    css, icon, sub = {
        "FAKE": ("verdict-fake", "🔴 FAKE NEWS", "Fused probability is at or above the threshold"),
        "REAL": ("verdict-real", "🟢 REAL NEWS", "Fused probability is below the threshold"),
        "UNCERTAIN": ("verdict-unc", "⚠️ UNCERTAIN",
                      f"Within {f['uncertain_band']} of the threshold — "
                      f"threshold outcome alone would be {f['final_label']}"),
    }[state]
    st.markdown(f"<div class='verdict-card {css}'>"
                f"<p class='verdict-title'>{icon}</p>"
                f"<p class='verdict-sub'>{sub}</p></div>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("P(Fake) — final", f"{100 * f['fused_p_fake']:.2f}%",
              help="Temperature-scaled fused probability. This is the decision quantity.")
    m2.metric("Confidence", f"{100 * f['confidence']:.2f}%",
              help="Distance from the threshold, normalised to [0,1]. "
                   "A DIFFERENT quantity from P(fake).")
    m3.metric("Decision threshold", f"{100 * f['threshold']:.2f}%",
              help="Loaded from outputs/module7/validation/fusion_parameters.json")
    lang_name = {"en": "English", "ta": "Tamil"}.get(res["language"]["code"], "—")
    tr = res["language"].get("tamil_ratio")
    m4.metric("Language", lang_name,
              delta=f"Tamil ratio {tr:.3f}" if tr is not None else "user-specified",
              delta_color="off")

    st.divider()
    left, right = st.columns(2)

    # -------------------------------------------------------- classifier
    with left:
        st.subheader("Classifier")
        a, b = st.columns(2)
        a.metric("P(Fake)", f"{100 * clf['p_fake']:.2f}%")
        b.metric("P(Real)", f"{100 * clf['p_real']:.2f}%")
        st.caption("Frozen Module 4 CE-MLP over an xlm-roberta-base mean-pooled, "
                   "L2-normalised 768-d embedding. **Confidence above is not this "
                   "number** — see the fusion breakdown.")
        with st.expander("Fusion breakdown (Module 7 frozen parameters)"):
            st.code(
                f"alpha        = {f['alpha']}          fitted on {f['fit_split']}\n"
                f"threshold    = {f['threshold']}\n"
                f"temperature  = {f['temperature']}    binary logit space; T<1 sharpens\n"
                f"\nclassifier P(fake)            = {f['p_fake_classifier']:.6f}\n"
                f"verifier-derived P(fake)      = "
                f"{f['p_fake_verifier'] if f['p_fake_verifier'] is not None else 'n/a'}\n"
                f"pre-temperature fused P(fake) = "
                f"{f['fused_p_fake_before_temperature']:.6f}\n"
                f"temperature-scaled P(fake)    = {f['fused_p_fake']:.6f}",
                language="text")

    # ------------------------------------------------------- verification
    with right:
        st.subheader("Fact verification")
        if ver.get("available"):
            st.write(f"**Dataset** `{ver['dataset']}`")
            st.write(f"**Verdict** `{ver['verdict']}` &nbsp; "
                     f"(p = {ver['verdict_probability']:.3f})")
            st.caption("Label space: " + ", ".join(f"`{x}`" for x in ver["label_space"]))
            st.bar_chart(ver["probs"], height=180)
        else:
            st.info(f"No verifier output — {ver.get('reason')}. "
                    "The decision rests on the classifier alone.")

    # ------------------------------------------------------------ evidence
    st.divider()
    st.subheader("Candidate evidence — retrieved automatically")
    if ret["status"] == "NO EVIDENCE RETRIEVED":
        st.warning("**NO EVIDENCE RETRIEVED** for this claim from the fixed corpus.")
    else:
        st.caption(f"Pool `{ret['pool']}` · retriever **{ret['retriever'].upper()}** · "
                   f"{ret['n_retrieved']} of top-{ret['top_k']}. Retrieved "
                   "automatically from a fixed corpus — not verified, and not "
                   "necessarily the correct evidence.")
        for e in ret["candidates"][: int(cfg["display"]["max_evidence_shown"])]:
            with st.expander(f"**{e['rank']}.** [{e['corpus']}] · score "
                             f"{e['score']:.3f} · confidence {e['confidence']:.2f}"):
                st.write(e["text"][: int(cfg["display"]["truncate_evidence_chars"])])
                st.caption(f"`evidence_id: {e['evidence_id']}` · "
                           f"pool size {e['pool_size']:,} · retriever {e['retriever']}")

    # ------------------------------------------------------- explanation
    st.divider()

    def render_explanation() -> None:
        if not xai.get("method"):
            st.caption("Token explanation was not computed for this run.")
            return
        st.caption(f"Method **{xai['method']}** · target **{xai['target']}** · "
                   f"base P(fake) = {xai['base_probability']:.4f}")
        e1, e2 = st.columns(2)
        with e1:
            st.markdown("**🔴 Pushed the classifier toward FAKE**")
            st.caption("positive attribution — removing the token *lowers* P(fake)")
            toks = (xai.get("top_positive_tokens") or [])[:10]
            st.markdown("".join(
                f"<span class='tok tok-pos'>{t['token']} {t['attribution']:+.3f}</span>"
                for t in toks) or "_none_", unsafe_allow_html=True)
        with e2:
            st.markdown("**🟢 Pushed the classifier toward REAL**")
            st.caption("negative attribution — removing the token *raises* P(fake)")
            toks = (xai.get("top_negative_tokens") or [])[:10]
            st.markdown("".join(
                f"<span class='tok tok-neg'>{t['token']} {t['attribution']:+.3f}</span>"
                for t in toks) or "_none_", unsafe_allow_html=True)
        if "faithfulness" in xai:
            st.caption(xai["faithfulness"]["label"])
        st.warning("These attributions explain the **classifier's P(fake)** only. "
                   "They do **not** establish factual truth, and they do **not** "
                   "explain the verifier's decision or the fused verdict.")

    if state == "FAKE":
        st.subheader("Why was it classified this way?")
        st.markdown("**Classifier-focused explanation**")
        render_explanation()
    else:
        with st.expander("Classifier-focused explanation", expanded=False):
            render_explanation()

    # --------------------------------------------------------------- reason
    st.divider()
    st.subheader("Reason")
    for s in res["reason"]["sentences"]:
        badge = {"MEASURED": "🔵", "INTERPRETATION": "🟡", "LIMITATION": "⚪"}.get(s["kind"], "•")
        st.markdown(f"{badge} **[{s['kind']}]** {s['text']}")
    st.caption(res["reason"]["generator"])

    with st.expander("Limitations"):
        for t in res["limitations"]:
            st.markdown(f"- {t}")
    with st.expander("Full machine-readable result (JSON)"):
        st.json(res)
