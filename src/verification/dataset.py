"""Module 6 — verifier dataset: label spaces and (claim, evidence) pairing.

THE LABEL-SPACE RULE, enforced in code rather than left to discipline:

    FEVER  : SUPPORTS / REFUTES / NOT ENOUGH INFO   (textual entailment)
    X-FACT : false / true / partly true/misleading  (fact-checker veracity)

These are different tasks over different label semantics. "partly
true/misleading" has no FEVER counterpart; forcing it into SUPPORTS or
REFUTES would fabricate a judgement the annotator never made. So:

  * no mapping function between the two spaces exists in this file;
  * `LabelSpace` refuses to encode a label it does not own;
  * X-FACT labels are stored exactly as published, including the literal
    string "partly true/misleading";
  * the encoder is shared between tasks so FEVER -> Tamil transfer is
    measurable, but the classification HEAD is replaced, never reused.

Reported separately, always.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

FEVER_LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
X_FACT_LABELS = ["false", "true", "partly true/misleading"]


class LabelSpace:
    """A closed label vocabulary. Unknown labels raise instead of defaulting."""

    def __init__(self, name: str, labels: Sequence[str], task: str) -> None:
        self.name = name
        self.task = task
        self.labels = list(labels)
        self._to_id = {lab: i for i, lab in enumerate(self.labels)}
        # Tolerate case/whitespace noise in the source, never semantic drift.
        self._norm = {lab.strip().lower(): i for i, lab in enumerate(self.labels)}

    def __len__(self) -> int:
        return len(self.labels)

    def encode(self, label: Any) -> int:
        if label is None:
            raise KeyError(f"{self.name}: missing label")
        s = str(label)
        if s in self._to_id:
            return self._to_id[s]
        key = s.strip().lower()
        if key in self._norm:
            return self._norm[key]
        raise KeyError(
            f"{self.name}: unknown label {s!r}. Known: {self.labels}. "
            f"Labels are never remapped across datasets; if this label came from "
            f"another corpus, it belongs to a different label space.")

    def decode(self, idx: int) -> str:
        if not 0 <= int(idx) < len(self.labels):
            raise IndexError(f"{self.name}: label id {idx} outside 0..{len(self.labels)-1}")
        return self.labels[int(idx)]

    def can_encode(self, label: Any) -> bool:
        try:
            self.encode(label)
            return True
        except KeyError:
            return False


FEVER_SPACE = LabelSpace("fever", FEVER_LABELS, task="entailment")
X_FACT_SPACE = LabelSpace("x_fact", X_FACT_LABELS, task="veracity")
LABEL_SPACES = {"fever": FEVER_SPACE, "x_fact": X_FACT_SPACE}


def get_label_space(dataset: str) -> LabelSpace:
    if dataset not in LABEL_SPACES:
        raise KeyError(f"no label space for dataset {dataset!r}; known: {sorted(LABEL_SPACES)}")
    return LABEL_SPACES[dataset]


def assert_label_spaces_disjoint() -> None:
    """FEVER and X-FACT label strings must not overlap.

    If they ever did, a mis-routed record could be silently encoded against
    the wrong space and no error would surface.
    """
    overlap = set(FEVER_LABELS) & set(X_FACT_LABELS)
    if overlap:
        raise AssertionError(f"label spaces overlap: {sorted(overlap)}")


@dataclass
class VerifierExample:
    claim_id: str
    claim_text: str
    evidence_texts: list[str]
    label_id: int
    label_str: str
    dataset: str
    language: str
    split: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_text_pair(self, separator: str = " </s> ", max_evidence: int = 5) -> tuple[str, str]:
        """(claim, joined evidence) for a cross-encoder tokenizer.

        Evidence is joined in RANK ORDER: the reranker's best candidate comes
        first, so truncation at max_length drops the least relevant text.
        """
        return self.claim_text, separator.join(self.evidence_texts[:max_evidence])


def build_examples(
    claims: Sequence[dict[str, Any]],
    evidence_texts: dict[str, str],
    retrieved: dict[str, Sequence[str]] | None,
    dataset: str,
    split_of: dict[int, str],
    max_evidence: int = 5,
    use_gold_evidence: bool = False,
    claim_only: bool = False,
    evidence_lang: dict[str, str] | None = None,
    keep_languages: Sequence[str] | None = None,
) -> tuple[list[VerifierExample], dict[str, Any]]:
    """Assemble verifier examples, accounting for every input claim.

    `use_gold_evidence=True` is the oracle ablation -- an upper bound, never
    the primary path (`gold_evidence_in_primary_path: false` in Module 5's
    config). `claim_only=True` is the ablation that isolates whether evidence
    contributes at all.

    `keep_languages` restricts evidence to those languages (the English-only
    vs multilingual ablation). Filtering happens BEFORE the `max_evidence`
    slice, so the model sees the top-k of the *permitted* language rather than
    the top-k overall with holes punched in it. Requires `evidence_lang`.

    REPORTING HONESTY: `stats["evidence_source"]` is derived from what was
    actually ATTACHED, not from the caller's intent. An earlier version
    hard-coded "retrieved" whenever claim_only/oracle were both false, which
    recorded `"retrieved"` on a run where all 260,032 examples in fact had
    zero evidence. The manifest is the audit trail; it must not flatter the
    run. `n_with_evidence` is the number to check.
    """
    space = get_label_space(dataset)
    if keep_languages is not None and evidence_lang is None:
        raise ValueError(
            "keep_languages was given but evidence_lang was not; the language of "
            "each evidence item cannot be determined. Pass evidence_lang="
            "{evidence_id: language} built from corpus['evidence'].")
    keep = set(keep_languages) if keep_languages is not None else None
    out: list[VerifierExample] = []
    stats = {"n_input": len(claims), "n_built": 0, "n_dropped_unknown_label": 0,
             "n_dropped_no_split": 0, "n_with_no_evidence": 0, "n_with_evidence": 0,
             "n_evidence_attached": 0, "n_dropped_by_language_filter": 0,
             "by_label": {}, "by_split": {}, "by_evidence_language": {},
             "requested_mode": ("claim_only" if claim_only
                                else ("gold" if use_gold_evidence else "retrieved")),
             "keep_languages": sorted(keep) if keep else None}

    for i, c in enumerate(claims):
        split = split_of.get(i)
        if split is None:
            stats["n_dropped_no_split"] += 1
            continue
        try:
            lid = space.encode(c.get("label"))
        except KeyError:
            stats["n_dropped_unknown_label"] += 1
            continue

        cid = c.get("claim_id", str(i))
        if claim_only:
            ev_ids: Sequence[str] = []
        elif use_gold_evidence:
            ev_ids = c.get("evidence") or []
        else:
            ev_ids = (retrieved or {}).get(cid, [])

        ev_ids = [e for e in ev_ids if e in evidence_texts]
        if keep is not None:
            before = len(ev_ids)
            ev_ids = [e for e in ev_ids if (evidence_lang or {}).get(e) in keep]
            stats["n_dropped_by_language_filter"] += before - len(ev_ids)
        ev_ids = list(ev_ids)[:max_evidence]          # slice AFTER filtering
        texts = [evidence_texts[e] for e in ev_ids]

        if texts:
            stats["n_with_evidence"] += 1
            stats["n_evidence_attached"] += len(texts)
            for e in ev_ids:
                lg = (evidence_lang or {}).get(e, "unknown")
                stats["by_evidence_language"][lg] = stats["by_evidence_language"].get(lg, 0) + 1
        elif not claim_only:
            stats["n_with_no_evidence"] += 1

        lab = space.decode(lid)
        stats["by_label"][lab] = stats["by_label"].get(lab, 0) + 1
        stats["by_split"][split] = stats["by_split"].get(split, 0) + 1
        out.append(VerifierExample(
            claim_id=cid, claim_text=c.get("text", ""), evidence_texts=texts,
            label_id=lid, label_str=lab, dataset=dataset,
            language=str(c.get("language", "unknown")), split=split,
            metadata={"n_evidence": len(texts),
                      "official_split": c.get("metadata", {}).get("official_split")}))
        stats["n_built"] += 1

    stats["accounting_balances"] = (
        stats["n_input"] == stats["n_built"] + stats["n_dropped_unknown_label"]
        + stats["n_dropped_no_split"])

    # Derived from reality, never from intent. If nothing was attached, say so.
    if claim_only:
        stats["evidence_source"] = "claim_only"
    elif stats["n_with_evidence"] == 0:
        stats["evidence_source"] = "none"
    elif use_gold_evidence:
        stats["evidence_source"] = "gold"
    else:
        stats["evidence_source"] = "retrieved"
    stats["mean_evidence_per_example"] = (
        stats["n_evidence_attached"] / stats["n_built"] if stats["n_built"] else 0.0)
    stats["evidence_coverage"] = (
        stats["n_with_evidence"] / stats["n_built"] if stats["n_built"] else 0.0)
    return out, stats


def stratified_subset(examples: Sequence[VerifierExample], size: int | None,
                      seed: int = 42) -> list[VerifierExample]:
    """Label-stratified subsample, deterministic for a given seed.

    Used to cut FEVER's 228,090 training claims to a size free Colab can
    actually train on. Stratifying keeps the label distribution intact, so the
    reduction changes scale but not the task balance -- and it is reported as
    a scope limitation rather than presented as a full-corpus result.
    """
    import random

    if size is None or size >= len(examples):
        return list(examples)
    by_label: dict[int, list[VerifierExample]] = {}
    for e in examples:
        by_label.setdefault(e.label_id, []).append(e)

    rng = random.Random(seed)
    total = len(examples)
    out: list[VerifierExample] = []
    for lid in sorted(by_label):
        group = sorted(by_label[lid], key=lambda x: x.claim_id)   # stable input order
        take = max(1, round(size * len(group) / total))
        rng.shuffle(group)
        out.extend(group[:take])
    out.sort(key=lambda x: x.claim_id)
    if len(out) > size:
        rng.shuffle(out)
        out = sorted(out[:size], key=lambda x: x.claim_id)
    return out
