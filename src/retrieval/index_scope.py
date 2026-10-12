"""Module 6 — translate Module 5's claim-level index scope into evidence ids.

This is the single most dangerous file in Module 6. Module 5's
`build_index_scope()` answers "which CLAIMS may a given phase see?", but a
retrieval index stores EVIDENCE. The translation between the two is where a
leakage bug would hide, because nothing downstream would look wrong: the
index would simply be a little too rich, and every retrieval number would be
optimistic.

The rule implemented here:

    permitted evidence for phase P
        = union of the evidence cited by the claims permitted in phase P

Evidence cited ONLY by out-of-scope claims is excluded. Evidence cited by
both an in-scope and an out-of-scope claim is INCLUDED -- it is legitimately
reachable from the in-scope side, and excluding it would distort the corpus.
That asymmetry is deliberate and is asserted by tests.

Module 5 is read-only here: `build_index_scope()` is imported and used
as-is, never reimplemented, so the two can never drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from src.fact_verification.leakage_gate import build_index_scope


@dataclass
class EvidenceScope:
    """Which evidence a phase may index, and the audit trail for why."""

    phase: str
    allowed_splits: list[str]
    claim_indices: list[int]
    evidence_ids: set[str]
    n_claims_in_scope: int = 0
    n_claims_out_of_scope: int = 0
    n_evidence_in_scope: int = 0
    n_evidence_excluded: int = 0
    n_evidence_shared_with_excluded_claims: int = 0
    stats: dict[str, Any] = field(default_factory=dict)

    def contains(self, evidence_id: str) -> bool:
        return evidence_id in self.evidence_ids


def claim_evidence_ids(claim: dict[str, Any]) -> list[str]:
    """Evidence ids cited by one claim record, defensively.

    Module 5 writes `evidence` as a list of evidence-id strings. A claim with
    no gold evidence (FEVER's NOT ENOUGH INFO) yields an empty list, which is
    correct and not an error.
    """
    ev = claim.get("evidence")
    if not ev:
        return []
    if isinstance(ev, str):
        return [ev]
    return [str(e) for e in ev if e]


def build_evidence_scope(
    claims: list[dict[str, Any]],
    splits: dict[str, list[int]],
    phase: str,
    scope_config: dict[str, list[str]],
) -> EvidenceScope:
    """Evidence permitted for `phase`, derived from Module 5's claim scope.

    Raises on an unknown phase or an unknown split name -- delegated to
    `build_index_scope()`, so the error text stays consistent with Module 5.
    """
    claim_idx = build_index_scope(splits, phase, scope_config)
    in_scope = set(claim_idx)

    permitted: set[str] = set()
    for i in claim_idx:
        if i < 0 or i >= len(claims):
            raise IndexError(f"claim index {i} out of range for {len(claims)} claims")
        permitted.update(claim_evidence_ids(claims[i]))

    out_of_scope_only: set[str] = set()
    shared = 0
    for i, c in enumerate(claims):
        if i in in_scope:
            continue
        for eid in claim_evidence_ids(c):
            if eid in permitted:
                shared += 1
            else:
                out_of_scope_only.add(eid)

    scope = EvidenceScope(
        phase=phase,
        allowed_splits=list(scope_config.get(phase, [])),
        claim_indices=claim_idx,
        evidence_ids=permitted,
        n_claims_in_scope=len(claim_idx),
        n_claims_out_of_scope=len(claims) - len(claim_idx),
        n_evidence_in_scope=len(permitted),
        n_evidence_excluded=len(out_of_scope_only),
        n_evidence_shared_with_excluded_claims=shared,
    )
    scope.stats = {
        "phase": phase,
        "allowed_splits": scope.allowed_splits,
        "n_claims_in_scope": scope.n_claims_in_scope,
        "n_claims_out_of_scope": scope.n_claims_out_of_scope,
        "n_evidence_in_scope": scope.n_evidence_in_scope,
        "n_evidence_excluded_as_out_of_scope_only": scope.n_evidence_excluded,
        "n_citations_from_excluded_claims_to_in_scope_evidence": shared,
        "rule": ("evidence permitted = union of evidence cited by permitted claims; "
                 "evidence cited only by out-of-scope claims is excluded"),
    }
    return scope


def assert_index_within_scope(indexed_ids: Iterable[str], scope: EvidenceScope) -> None:
    """Hard gate: fail loudly if the built index exceeds its scope.

    Called after index construction, not before, so it audits what was
    actually indexed rather than what was intended.
    """
    extra = sorted(set(indexed_ids) - scope.evidence_ids)
    if extra:
        raise AssertionError(
            f"LEAKAGE: {len(extra)} evidence item(s) in the {scope.phase!r} index are "
            f"outside its permitted scope (allowed splits {scope.allowed_splits}). "
            f"First offenders: {extra[:5]}")


def verify_phase_isolation(
    claims: list[dict[str, Any]],
    splits: dict[str, list[int]],
    scope_config: dict[str, list[str]],
) -> dict[str, Any]:
    """Independent re-check that each phase excludes what it must exclude.

    Recomputed from the claims rather than from the scopes, so a bug in
    `build_evidence_scope()` cannot hide behind its own bookkeeping.
    """
    checks: list[dict[str, Any]] = []

    def split_exclusive_evidence(target: str, others: list[str]) -> set[str]:
        """Evidence cited by `target` and by no split in `others`."""
        tgt: set[str] = set()
        for i in splits.get(target, []):
            tgt.update(claim_evidence_ids(claims[i]))
        elsewhere: set[str] = set()
        for s in others:
            for i in splits.get(s, []):
                elsewhere.update(claim_evidence_ids(claims[i]))
        return tgt - elsewhere

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    present = [s for s in ("train", "validation", "test") if s in splits]

    if "train" in splits:
        train_scope = build_evidence_scope(claims, splits, "train", scope_config)
        for holdout in ("validation", "test"):
            if holdout not in splits:
                continue
            excl = split_exclusive_evidence(holdout, [s for s in present if s != holdout])
            bad = excl & train_scope.evidence_ids
            add(f"train_index_excludes_{holdout}_exclusive_evidence", not bad,
                f"{len(excl)} {holdout}-exclusive evidence item(s); {len(bad)} leaked into train"
                + (f" e.g. {sorted(bad)[:3]}" if bad else ""))

    if "validation" in splits and "test" in splits:
        val_scope = build_evidence_scope(claims, splits, "validation", scope_config)
        excl = split_exclusive_evidence("test", [s for s in present if s != "test"])
        bad = excl & val_scope.evidence_ids
        add("validation_index_excludes_test_exclusive_evidence", not bad,
            f"{len(excl)} test-exclusive evidence item(s); {len(bad)} leaked into validation"
            + (f" e.g. {sorted(bad)[:3]}" if bad else ""))

    if "test" in splits:
        test_scope = build_evidence_scope(claims, splits, "test", scope_config)
        all_ids: set[str] = set()
        for c in claims:
            all_ids.update(claim_evidence_ids(c))
        add("test_index_within_corpus", test_scope.evidence_ids <= all_ids,
            f"{len(test_scope.evidence_ids)} evidence item(s) indexed at test phase")

    return {
        "passed": all(c["passed"] for c in checks),
        "n_checks": len(checks),
        "n_passed": sum(1 for c in checks if c["passed"]),
        "checks": checks,
    }
