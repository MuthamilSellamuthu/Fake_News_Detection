"""Module 4 mechanics tests.

Runs without torch or scikit-learn, mirroring `tests/test_embeddings.py`:
the components that can be verified without a GPU are verified
thoroughly, and the torch-dependent paths are exercised in Colab.

The SupCon tests compare against values computed BY HAND rather than
asserting "the loss goes down". That distinction matters: the first
implementation of this loss returned NaN for every batch (`-inf * False`
is NaN, not 0), and a "loss decreases" test would have passed it,
because NaN comparisons are always False and the curve simply looked
flat.

Run directly (no pytest needed):
    python tests/test_contrastive.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.contrastive.batch_sampler import (  # noqa: E402
    LanguageAwareBatchSampler,
    UniformBatchSampler,
    build_sampler,
)
from src.contrastive.evaluation import (  # noqa: E402
    compute_metrics,
    confusion_matrix,
    evaluate_predictions,
    majority_class_baseline,
    roc_auc,
    summarise_across_seeds,
)
from src.contrastive.embedding_dataset import SplitEmbeddings, verify_frozen_inputs  # noqa: E402
from src.contrastive.supcon_loss import (  # noqa: E402
    batch_positive_statistics,
    supcon_loss_reference,
)

# Module 2's verified distribution, used so the sampler is tested against
# the real class/language imbalance rather than a toy one.
REAL_TRAIN_EN, REAL_TRAIN_TA = 44520, 3785


# ---------------------------------------------------------------- SupCon

def test_supcon_matches_hand_computation():
    """Four unit vectors, tau=1.0, worked out on paper.

    Anchor 0: A(0)={1,2,3}, P(0)={1}. sim=[.,1,0,0]; after subtracting
    the row max (1): [.,0,-1,-1]; denom = 1 + 2e^-1; loss contribution
    = log(1 + 2e^-1). All four anchors are symmetric.
    """
    z = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    expected = math.log(1 + 2 * math.exp(-1))
    assert abs(supcon_loss_reference(z, y, temperature=1.0) - expected) < 1e-9


def test_supcon_degenerate_closed_form():
    """n identical same-label vectors, no negatives -> exactly log(n-1).

    Verifying against a closed form at several n proves the
    normalisation over |P(i)| and the exclusion of self-similarity are
    both correct.
    """
    for n in (3, 5, 9, 17):
        got = supcon_loss_reference(np.tile([1.0, 0.0], (n, 1)), np.zeros(n, dtype=int), 0.07)
        assert abs(got - math.log(n - 1)) < 1e-9, f"n={n}: {got} != log({n-1})"


def test_supcon_separated_beats_entangled():
    rng = np.random.default_rng(0)
    labels = np.array([0] * 16 + [1] * 16)
    separated = np.vstack([np.tile([1.0, 0.0], (16, 1)), np.tile([0.0, 1.0], (16, 1))])
    entangled = rng.normal(size=(32, 2))
    entangled /= np.linalg.norm(entangled, axis=1, keepdims=True)
    assert supcon_loss_reference(separated, labels, 0.07) < supcon_loss_reference(entangled, labels, 0.07)


def test_supcon_edge_cases_and_stability():
    # no positives -> 0.0, and specifically NOT NaN (the original bug)
    v = supcon_loss_reference(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0, 1]), 0.07)
    assert v == 0.0 and not math.isnan(v)
    assert supcon_loss_reference(np.array([[1.0, 0.0]]), np.array([0]), 0.07) == 0.0

    rng = np.random.default_rng(1)
    z = rng.normal(size=(32, 8))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    y = rng.integers(0, 2, 32)
    for tau in (0.01, 0.05, 0.07, 0.5, 1.0):
        assert np.isfinite(supcon_loss_reference(z, y, tau)), f"non-finite at tau={tau}"


def test_positive_statistics_justify_language_aware_sampling():
    """The measurement behind the paper's claim, not an assertion."""
    labels = np.array([0, 1] * 128)
    uniform_like = batch_positive_statistics(labels, np.array(["en"] * 236 + ["ta"] * 20))
    language_aware = batch_positive_statistics(labels, np.array(["en"] * 192 + ["ta"] * 64))
    assert (language_aware["ta_mean_same_language_positives"]
            > 2 * uniform_like["ta_mean_same_language_positives"])
    assert language_aware["anchors_with_zero_positives"] == 0


# --------------------------------------------------------------- Sampler

def _real_languages() -> np.ndarray:
    return np.array(["en"] * REAL_TRAIN_EN + ["ta"] * REAL_TRAIN_TA)


def test_language_aware_holds_tamil_fraction_exactly():
    langs = _real_languages()
    sampler = LanguageAwareBatchSampler(langs, batch_size=256, tamil_fraction=0.25, seed=42)
    fractions = []
    for batch in sampler:
        assert len(batch) == 256
        fractions.append((langs[batch] == "ta").mean())
    assert np.allclose(fractions, 0.25), f"fraction drifted: {min(fractions)}..{max(fractions)}"


def test_language_aware_sees_every_tamil_row_each_epoch():
    langs = _real_languages()
    seen = set()
    for batch in LanguageAwareBatchSampler(langs, 256, 0.25, seed=42):
        seen.update(batch[langs[batch] == "ta"].tolist())
    assert len(seen) == REAL_TRAIN_TA


def test_uniform_tracks_natural_rate_and_starves_tamil():
    """The control, and the reason the intervention exists."""
    langs = _real_languages()
    natural = REAL_TRAIN_TA / len(langs)
    fractions = [(langs[b] == "ta").mean() for b in UniformBatchSampler(len(langs), 256, seed=42)]
    assert abs(float(np.mean(fractions)) - natural) < 0.01
    # Worst-case batches are severely Tamil-starved -- exactly the
    # condition that makes Tamil anchors positive-poor under SupCon.
    assert min(fractions) < 0.05


def test_sampler_determinism_and_epoch_reshuffle():
    langs = _real_languages()
    a = [b.tolist() for b in LanguageAwareBatchSampler(langs, 256, 0.25, seed=42)]
    b = [x.tolist() for x in LanguageAwareBatchSampler(langs, 256, 0.25, seed=42)]
    assert a == b
    c = [x.tolist() for x in LanguageAwareBatchSampler(langs, 256, 0.25, seed=43)]
    assert a != c
    s = LanguageAwareBatchSampler(langs, 256, 0.25, seed=42)
    s.set_epoch(1)
    assert [x.tolist() for x in s] != a


def test_sampler_input_guards():
    langs = _real_languages()
    for kwargs in ({"tamil_fraction": 0.0}, {"tamil_fraction": 1.0}, {"batch_size": 1}):
        try:
            LanguageAwareBatchSampler(langs, **{"batch_size": 256, "tamil_fraction": 0.25, **kwargs})
            raise AssertionError(f"should have rejected {kwargs}")
        except ValueError:
            pass
    try:
        LanguageAwareBatchSampler(np.array(["en"] * 100), 32, 0.25)
        raise AssertionError("should have rejected an empty Tamil pool")
    except ValueError:
        pass
    try:
        build_sampler("nonsense", langs, 256, 0.25, 42)
        raise AssertionError("should have rejected an unknown strategy")
    except ValueError:
        pass


# ------------------------------------------------------------ Evaluation

def test_metrics_match_hand_computation():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    assert confusion_matrix(y_true, y_pred).tolist() == [[1, 1], [1, 2]]
    m = compute_metrics(y_true, y_pred)
    assert abs(m["accuracy"] - 0.6) < 1e-9
    assert abs(m["macro_f1"] - (0.5 + 2 / 3) / 2) < 1e-6
    assert m["per_class"]["0"]["precision"] == 0.5
    assert m["per_class"]["1"]["support"] == 3


def test_roc_auc_known_values():
    assert roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert roc_auc(np.array([0, 0, 1, 1]), np.array([0.5] * 4)) == 0.5
    # Undefined rather than a misleading 0.0 when one class is absent.
    assert roc_auc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])) is None


def test_per_language_slicing_exposes_hidden_tamil_failure():
    """The aggregate can look strong while Tamil is near chance."""
    rng = np.random.default_rng(0)
    y_true = np.concatenate([np.array([0, 1] * 100), np.array([0, 1] * 10)])
    y_pred = np.concatenate([np.array([0, 1] * 100), rng.integers(0, 2, 20)])
    langs = np.array(["en"] * 200 + ["ta"] * 20)
    r = evaluate_predictions(y_true, y_pred, langs)
    assert r["per_language"]["en"]["macro_f1"] > 0.99
    assert r["per_language"]["ta"]["macro_f1"] < 0.75
    assert r["overall"]["macro_f1"] > r["per_language"]["ta"]["macro_f1"] + 0.2


def test_metrics_edge_cases_stay_finite():
    assert compute_metrics(np.array([]), np.array([]))["n"] == 0
    m = compute_metrics(np.array([0, 0, 0]), np.array([0, 0, 0]))
    assert np.isfinite(m["macro_f1"])


def test_majority_baseline_and_seed_summary():
    langs = np.array(["en"] * 8 + ["ta"] * 2)
    b = majority_class_baseline(np.array([0] * 80 + [1] * 20), np.array([0, 1] * 5), langs)
    assert b["majority_class"] == 0
    assert "per_language" in b and "ta" in b["per_language"]

    s = summarise_across_seeds([{"macro_f1": 0.80}, {"macro_f1": 0.82}, {"macro_f1": 0.84}])
    assert abs(s["macro_f1"]["mean"] - 0.82) < 1e-9
    assert abs(s["macro_f1"]["std"] - float(np.std([0.80, 0.82, 0.84]))) < 1e-6
    assert s["macro_f1"]["values"] == [0.8, 0.82, 0.84]


# ------------------------------------------------------ Frozen-input gate

def _fake_split(name: str, n: int, n_ta: int, dim: int = 768, seed: int = 0) -> SplitEmbeddings:
    rng = np.random.default_rng(seed)
    e = rng.normal(size=(n, dim)).astype("float32")
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    return SplitEmbeddings(
        split=name, pooling="cls", embeddings=e,
        labels=rng.integers(0, 2, n).astype("int64"),
        languages=np.array(["en"] * (n - n_ta) + ["ta"] * n_ta, dtype="<U2"),
        record_ids=np.array([f"{name}_{i}" for i in range(n)], dtype="<U64"),
    )


def _good_data() -> dict[str, SplitEmbeddings]:
    return {"train": _fake_split("train", 48305, 3785, seed=1),
            "validation": _fake_split("validation", 10351, 811, seed=2),
            "test": _fake_split("test", 10352, 812, seed=3)}


_EXPECTED = {"train": 48305, "validation": 10351, "test": 10352}


def _gate(data):
    return verify_frozen_inputs(data, _EXPECTED, 69008, 768, 5408)


def test_frozen_gate_accepts_correct_inputs():
    assert _gate(_good_data())["passed"] is True


def test_frozen_gate_rejects_wrong_counts():
    """The 69,552-substitution scenario that slipped through Module 3."""
    d = _good_data()
    d["train"] = _fake_split("train", 48686, 4166, seed=1)
    r = _gate(d)
    assert r["passed"] is False
    assert any(c["name"] == "dataset_counts" and not c["passed"] for c in r["checks"])


def test_frozen_gate_rejects_wrong_dimension():
    d = _good_data()
    d["test"] = _fake_split("test", 10352, 812, dim=512, seed=3)
    assert _gate(d)["passed"] is False


def test_frozen_gate_rejects_leakage():
    d = _good_data()
    d["test"].record_ids[0] = d["train"].record_ids[0]
    r = _gate(d)
    assert r["passed"] is False
    assert any(c["name"] == "zero_split_leakage" and not c["passed"] for c in r["checks"])


def test_frozen_gate_rejects_nan_and_unnormalised():
    d = _good_data()
    d["validation"].embeddings[0, 0] = np.nan
    assert _gate(d)["passed"] is False

    d2 = _good_data()
    d2["train"].embeddings *= 3.0
    r2 = _gate(d2)
    assert r2["passed"] is False
    assert any(c["name"] == "l2_normalised" and not c["passed"] for c in r2["checks"])


def test_frozen_gate_rejects_bad_label_or_language_domain():
    d = _good_data()
    d["train"].labels[0] = 7
    assert _gate(d)["passed"] is False

    d2 = _good_data()
    d2["validation"].languages[0] = "fr"
    assert _gate(d2)["passed"] is False


def test_subset_preserves_alignment():
    """Zero-shot slices by language; misalignment here would silently
    pair the wrong labels with the wrong rows."""
    s = _fake_split("train", 100, 25, seed=7)
    en = s.subset(s.language_mask("en"))
    assert en.n == 75
    assert set(en.languages.tolist()) == {"en"}
    assert len(en.labels) == len(en.record_ids) == 75
    # original untouched (frozen inputs must never be mutated)
    assert s.n == 100


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
