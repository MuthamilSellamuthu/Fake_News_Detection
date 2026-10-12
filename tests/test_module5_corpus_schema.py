"""Module 5 Stage 2 — schema and integrity tests for downloaded corpora.

Two modes, deliberately:

  * The audit HELPERS (field discovery, evidence extraction, script
    detection, ID preservation) are tested against synthetic records in
    every known X-FACT and FEVER shape. These run anywhere, including
    before any download, and are what prove the audit will read the real
    files correctly rather than silently reporting zeros.

  * The DOWNLOADED-DATA tests skip cleanly when
    `datasets/module5_external/` is absent, and run for real in Colab
    after acquisition.

Written this way because the audit's biggest risk is not crashing — it
is quietly failing to find the evidence field and reporting
"0 evidence" as though that were a property of the corpus.

    python tests/test_module5_corpus_schema.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "audit_m5", PROJECT_ROOT / "scripts" / "audit_module5_corpora.py")
audit_m5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit_m5)

from src.fact_verification.claim_selection import hash_key, tamil_ratio  # noqa: E402
from src.fact_verification.leakage_gate import group_claims, verify_no_leakage  # noqa: E402
from src.utils.config_loader import load_yaml  # noqa: E402

CONFIG = load_yaml(PROJECT_ROOT / "configs" / "module5_fact_verification.yaml")
DATA_DIR = PROJECT_ROOT / "datasets" / "module5_external"
TAMIL = "பஹல்காம் தாக்குதல் குறித்த தவறான கூற்று பரவுகிறது"

VALID_LANG_PATTERN = str
FROZEN_PATHS = ["datasets/processed", "embeddings", "outputs/module3", "outputs/module4", "models/module4"]


# ------------------------------------------------- field discovery

def test_field_discovery_handles_xfact_shape():
    """X-FACT TSV-style row: numbered evidence columns."""
    row = {"language": "ta", "site": "boomlive", "claim": TAMIL, "label": "false",
           "evidence_1": "Some Tamil evidence", "evidence_2": "English evidence text", "evidence_3": ""}
    assert audit_m5.first_present(row, audit_m5.CLAIM_FIELDS) == "claim"
    assert audit_m5.first_present(row, audit_m5.LABEL_FIELDS) == "label"
    assert audit_m5.first_present(row, audit_m5.LANG_FIELDS) == "language"
    ev, how = audit_m5.extract_evidence(row)
    assert len(ev) == 2, f"numbered evidence not extracted: {ev}"
    assert how.startswith("numbered")


def test_field_discovery_handles_fever_list_shape():
    """FEVER gold-evidence: list of [page, sent_id, text] or dicts."""
    row = {"id": 1, "claim": "Paris is in France.", "label": "SUPPORTS",
           "evidence": [["Paris", 0, "Paris is the capital of France."],
                        ["France", 3, "France is in Europe."]]}
    ev, how = audit_m5.extract_evidence(row)
    assert len(ev) == 2 and how.startswith("list")
    assert "capital of France" in ev[0]

    row_d = {"id": 2, "claim": "X", "label": "REFUTES",
             "evidence": [{"page": "Foo", "sentence_id": 1, "text": "Foo is a bar."}]}
    ev2, _ = audit_m5.extract_evidence(row_d)
    assert ev2 == ["Foo is a bar."]


def test_evidence_absence_is_reported_not_silently_zero():
    ev, how = audit_m5.extract_evidence({"claim": "no evidence here", "label": "NOT ENOUGH INFO"})
    assert ev == [] and how == "none_found"


def test_original_evidence_identifiers_preserved():
    """FEVER's link back to Wikipedia must survive the audit."""
    ids = audit_m5.extract_evidence_ids(
        {"evidence": [["Paris", 0, "text"], {"page": "France", "sentence_id": 3, "text": "t"}]})
    assert any("Paris" in i for i in ids)
    assert any("France#3" == i for i in ids)


# ------------------------------------------------- script detection

def test_script_detection_separates_tamil_from_latin():
    assert audit_m5.detect_script(TAMIL) == "tamil"
    assert audit_m5.detect_script("Paris is the capital of France.") == "latin"
    assert audit_m5.detect_script("") == "empty"
    assert audit_m5.detect_script("12345 !!!") == "non_alphabetic"


def test_script_detection_on_mixed_text_prefers_tamil():
    """Tamil claims often carry English named entities; a low Tamil
    threshold keeps them classified as Tamil rather than Latin."""
    assert audit_m5.detect_script(TAMIL + " Trump RSS") == "tamil"


def test_tamil_unicode_preserved_through_audit_helpers():
    tamil_in = [c for c in TAMIL if "஀" <= c <= "௿"]
    assert [c for c in hash_key(TAMIL) if "஀" <= c <= "௿"] == tamil_in
    assert tamil_ratio(TAMIL) > 0.5


# ------------------------------------------------- stats helpers

def test_length_stats_hand_computed():
    s = audit_m5.length_stats([1, 2, 3, 4, 5])
    assert s["n"] == 5 and s["min"] == 1 and s["max"] == 5
    assert s["mean"] == 3.0 and s["median"] == 3
    assert audit_m5.length_stats([])["n"] == 0


# ------------------------------------------------- corpus identity

def test_corpora_remain_distinguishable():
    assert CONFIG["corpora"]["merge_datasets"] is False
    assert CONFIG["corpora"]["preserve_source_identity"] is True
    assert CONFIG["corpora"]["x_fact"]["licence"] == "MIT"
    assert "CC BY-SA" in CONFIG["corpora"]["fever"]["licence"]


def test_fever_gold_evidence_limitation_is_declared():
    """The closed-pool caveat must be recorded in code, not just prose.

    Whitespace is normalised before matching: the caveat is written
    across wrapped docstring lines, and an exact-substring assertion
    would fail on the line break rather than on a missing declaration.
    """
    src = (PROJECT_ROOT / "scripts" / "download_module5_corpora.py").read_text(encoding="utf-8")
    flat = " ".join(src.split()).lower()

    # Machine-readable marker written into the manifest.
    assert "gold_evidence_only" in flat, "evidence_scope marker absent from the manifest record"
    # Human-readable caveat, in some form.
    assert "not all of wikipedia" in flat, "closed-pool caveat not stated"
    assert "closed-pool" in flat, "'closed-pool' not stated explicitly"
    # The caveat must also travel with the data, not only live in a comment.
    assert "evidence_scope_note" in flat, "caveat not recorded in the download manifest"


# ------------------------------------------------- leakage integration

def test_leakage_gate_integrates_with_audit_claims():
    from src.fact_verification.claim_selection import Claim

    claims = [Claim(claim_id=f"c{i}", text=f"claim number {i} with sufficient length here",
                    raw_text="", language="en" if i % 4 else "ta", dataset="x_fact")
              for i in range(40)]
    g = group_claims(claims, near_duplicate=True, threshold=0.90)
    splits = {"train": list(range(0, 28)), "validation": list(range(28, 34)), "test": list(range(34, 40))}
    r = verify_no_leakage(claims, splits, g.group_ids, near_duplicate_threshold=0.90)
    assert r["passed"] is True

    splits_bad = {"train": list(range(0, 28)) + [39], "validation": list(range(28, 34)),
                  "test": list(range(34, 40))}
    assert verify_no_leakage(claims, splits_bad, g.group_ids)["passed"] is False


# ------------------------------------------------- frozen modules

def test_modules_1_to_4_paths_exist_and_are_not_targets():
    """Neither Stage 2 script may write to a frozen path."""
    for script in ("download_module5_corpora.py", "audit_module5_corpora.py"):
        src = (PROJECT_ROOT / "scripts" / script).read_text(encoding="utf-8")
        for frozen in ("datasets/processed", "embeddings/", "outputs/module3", "outputs/module4"):
            for verb in ("to_csv", "to_json(", "write_text", "np.save"):
                bad = f'{verb}("{frozen}'
                assert bad not in src, f"{script} appears to write to frozen path {frozen}"


def test_download_requires_explicit_approval():
    src = (PROJECT_ROOT / "scripts" / "download_module5_corpora.py").read_text(encoding="utf-8")
    assert "--approve-download" in src
    assert "DOWNLOAD NOT APPROVED" in src


# ------------------------------------------- X-FACT config requirement

def test_xfact_config_is_declared_in_yaml():
    """The HF dataset requires an explicit config; loading without one
    raises 'Config name is missing'. The chosen config must be pinned in
    the YAML rather than left to a library default."""
    xf = CONFIG["corpora"]["x_fact"]
    assert "hf_config" in xf, "corpora.x_fact.hf_config missing from the YAML"
    assert xf["hf_config"] == "ta", f"expected Tamil config, got {xf['hf_config']!r}"
    assert xf["hf_config"] in xf["hf_config_available"]
    # all_languages is ~25x the data and is not needed for the current
    # research question; selecting it should be a conscious change.
    assert xf["hf_config"] != "all_languages"


def test_downloader_always_passes_an_explicit_config():
    """Every real `load_dataset("utahnlp/x-fact", ...)` CALL must pass a config.

    Checked by walking the AST rather than grepping the source: the
    module docstring legitimately quotes the failing config-less form
    while explaining the error, and a substring test flags that as a
    violation. Inspecting actual call nodes distinguishes a call from a
    mention.
    """
    import ast

    src = (PROJECT_ROOT / "scripts" / "download_module5_corpora.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    xfact_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
        if name != "load_dataset":
            continue
        if node.args and isinstance(node.args[0], ast.Constant) \
                and node.args[0].value == "utahnlp/x-fact":
            xfact_calls.append(node)

    assert xfact_calls, "no load_dataset call for utahnlp/x-fact found"
    for call in xfact_calls:
        has_positional_config = len(call.args) >= 2
        has_keyword_config = any(k.arg in ("name", "config_name") for k in call.keywords)
        assert has_positional_config or has_keyword_config, (
            f"load_dataset('utahnlp/x-fact') at line {call.lineno} passes no config — "
            "this raises 'Config name is missing'")

    assert "def download_x_fact(dest: Path, hf_config: str" in " ".join(src.split()), \
        "download_x_fact must accept the config name as a parameter"


def test_downloader_records_config_and_revision_in_manifest():
    """Config and revision must reach the manifest, or a later audit
    cannot tell which language subset produced a given statistic."""
    src = (PROJECT_ROOT / "scripts" / "download_module5_corpora.py").read_text(encoding="utf-8")
    flat = " ".join(src.split())
    assert '"hf_config": hf_config' in flat, "hf_config not recorded in the per-corpus record"
    assert '"dataset_revision"' in flat, "dataset revision not recorded"
    assert 'manifest["x_fact_config_selected"]' in flat, "selected config not recorded at manifest level"
    assert '"hf_config": hf_config,' in flat, "per-file config provenance not recorded"


def test_downloader_rejects_unknown_config():
    """An unavailable config must fail loudly, not fall back silently."""
    src = (PROJECT_ROOT / "scripts" / "download_module5_corpora.py").read_text(encoding="utf-8")
    flat = " ".join(src.split())
    assert "is not available" in flat, "no guard against an unknown config name"
    assert "if hf_config not in available" in flat


def test_xfact_statistics_scope_is_declared():
    """Tamil-only statistics must not be reportable as X-FACT overall."""
    src = (PROJECT_ROOT / "scripts" / "download_module5_corpora.py").read_text(encoding="utf-8")
    flat = " ".join(src.split()).lower()
    assert "hf_config_rationale" in flat
    assert "must not be reported as x-fact overall" in flat


# ------------------------------------------------- downloaded data (Colab)

def _corpus_files(name: str) -> list[Path]:
    d = DATA_DIR / name
    return sorted(d.glob("*.jsonl")) if d.exists() else []


def test_downloaded_xfact_schema_if_present():
    files = _corpus_files("x_fact")
    if not files:
        print("    (skipped: X-FACT not downloaded)")
        return
    for fp in files:
        rows = audit_m5.load_jsonl(fp, limit=200)
        assert rows, f"{fp.name} is empty"
        cf = audit_m5.first_present(rows[0], audit_m5.CLAIM_FIELDS)
        assert cf is not None, f"{fp.name}: no claim field; columns={list(rows[0].keys())}"
        assert sum(1 for r in rows if str(r.get(cf, "")).strip()) > 0


def test_downloaded_fever_schema_if_present():
    files = _corpus_files("fever_gold_evidence")
    if not files:
        print("    (skipped: FEVER not downloaded)")
        return
    valid = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO", "NOTENOUGHINFO", "NEI"}
    for fp in files:
        rows = audit_m5.load_jsonl(fp, limit=200)
        assert rows, f"{fp.name} is empty"
        lf = audit_m5.first_present(rows[0], audit_m5.LABEL_FIELDS)
        if lf:
            seen = {str(r.get(lf, "")).upper().replace("_", " ").strip() for r in rows}
            unknown = seen - valid
            assert not unknown, f"{fp.name}: unexpected labels {unknown}"


def test_no_silent_row_loss_if_present():
    """Audited counts must equal the raw line counts."""
    mp = PROJECT_ROOT / "outputs" / "module5" / "download_manifest.json"
    if not mp.exists():
        print("    (skipped: no download manifest)")
        return
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    for rec in manifest.get("corpora", {}).values():
        for f in rec.get("files", []):
            p = PROJECT_ROOT / f["path"]
            if not p.exists():
                continue
            with p.open(encoding="utf-8") as fh:
                actual = sum(1 for line in fh if line.strip())
            assert actual == f["n_rows"], f"{f['path']}: manifest {f['n_rows']} vs file {actual}"


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed, failed = 0, []
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"[FAIL] {t.__name__}: {exc}")
            failed.append(t.__name__)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {t.__name__}: {type(exc).__name__}: {exc}")
            failed.append(t.__name__)
    print(f"\n{passed}/{len(tests)} passed")
    if failed:
        print("FAILED:", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
