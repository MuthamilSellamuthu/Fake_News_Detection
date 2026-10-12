"""Module 5 — claim-level deduplication, grouping, and leakage-safe splitting.

Fact-verification corpora carry a hazard that ordinary classification
splits do not: **the same claim recurs across many articles**. A viral
claim may be fact-checked by five outlets in slightly different wording.
Splitting those rows independently puts near-identical claims on both
sides of the train/test boundary, and the reported score then measures
memorisation rather than verification.

The controls, applied strictly BEFORE any split:

  1. Exact deduplication on an aggressively normalised key.
  2. Near-duplicate detection (character n-gram Jaccard) to catch
     rewordings that exact matching misses.
  3. Transitive grouping via union-find, so A~B and B~C put A, B and C
     in one group even when A and C are not directly similar.
  4. Group-aware splitting: an entire group goes to exactly one split.
  5. Retrieval-index scoping, so a training-phase index can never
     contain validation or test evidence.

`verify_no_leakage()` is the gate: it re-checks the finished split from
scratch rather than trusting the splitter, because a splitter bug and a
splitter self-check would fail together.

Character n-grams (not word n-grams) are used deliberately: Tamil is
agglutinative and whitespace-tokenised word overlap badly understates
similarity between morphological variants of the same claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

import numpy as np

from src.fact_verification.claim_selection import Claim, hash_key


# --------------------------------------------------------------------
# Similarity
# --------------------------------------------------------------------

def char_ngrams(text: str, n: int = 5) -> set[str]:
    s = hash_key(text)
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


# --------------------------------------------------------------------
# Union-find for transitive grouping
# --------------------------------------------------------------------

class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# --------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------

@dataclass
class GroupingResult:
    group_ids: list[int]                 # one per claim, aligned to input
    n_groups: int
    n_exact_duplicate_pairs: int
    n_near_duplicate_pairs: int
    largest_group_size: int
    summary: dict[str, Any]


def candidate_pairs(
    grams: list[set[str]],
    languages: list[str],
    threshold: float,
) -> Iterator[tuple[int, int]]:
    """Yield every pair that could possibly reach Jaccard >= `threshold`.

    EXACT: this is a filter, never an approximation. Any pair it skips is
    provably below the threshold, so the resulting grouping is identical
    to comparing all pairs. `tests/test_fact_verification.py` asserts
    that equivalence against brute force.

    Why this exists. The previous implementation blocked on
    (language, length//25, first-3-chars) and compared all pairs inside
    each bucket. With one language and a handful of length buckets --
    exactly FEVER's shape -- bucket size grows linearly with the corpus,
    so the pass is quadratic. Measured: O(n^2.05) at n = 4k/8k/16k,
    projecting to roughly 5 hours for FEVER's 260,251 claims, which no
    Colab session survives. Skipping the check instead was not acceptable:
    an unchecked corpus cannot be reported as leakage-free.

    Two standard exact filters replace the blocking:

    1. Length filter. Jaccard(A,B) = |A n B| / |A u B| <= min/max of the
       set sizes, so |A| < threshold * |B| makes the threshold
       unreachable. Sorting by size lets us stop scanning early.
    2. Prefix filter (PPJoin). Order n-grams by global rarity. For
       Jaccard >= t the overlap must be at least ceil(t * |A|), so A and B
       must share at least one n-gram within A's first
       |A| - ceil(t*|A|) + 1 rarest n-grams. Indexing only that prefix
       turns the scan near-linear, because rare n-grams match few sets.

    Different languages are never compared, preserving the old behaviour.
    """
    n = len(grams)
    if n < 2:
        return

    # Global n-gram frequency: rare grams first make prefixes selective.
    freq: dict[str, int] = {}
    for g in grams:
        for tok in g:
            freq[tok] = freq.get(tok, 0) + 1

    order = sorted(range(n), key=lambda i: len(grams[i]))
    sorted_tokens = {i: sorted(grams[i], key=lambda t: (freq[t], t)) for i in order}

    index: dict[tuple[str, str], list[int]] = {}
    emitted: set[tuple[int, int]] = set()

    for i in order:
        gi = grams[i]
        size_i = len(gi)
        if size_i == 0:
            continue
        prefix_len = max(1, size_i - math.ceil(threshold * size_i) + 1)
        seen: set[int] = set()
        for tok in sorted_tokens[i][:prefix_len]:
            for j in index.get((languages[i], tok), ()):
                if j in seen:
                    continue
                seen.add(j)
                # Length filter: |smaller| >= t * |larger| is necessary.
                if len(grams[j]) < threshold * size_i:
                    continue
                pair = (j, i) if j < i else (i, j)
                if pair not in emitted:
                    emitted.add(pair)
                    yield pair
        for tok in sorted_tokens[i][:prefix_len]:
            index.setdefault((languages[i], tok), []).append(i)


def group_claims(
    claims: list[Claim],
    near_duplicate: bool = True,
    ngram: int = 5,
    threshold: float = 0.90,
    blocking: bool = True,
) -> GroupingResult:
    """Assign every claim a group id; identical/near-identical claims share one.

    Near-duplicate comparison is O(n^2) in the worst case. `blocking`
    restricts comparisons to claims sharing a coarse bucket
    (language + rounded length + first-3-gram), which makes the pass
    tractable on tens of thousands of claims while keeping recall high
    for genuine rewordings -- true near-duplicates are almost always the
    same language and similar length.
    """
    n = len(claims)
    uf = _UnionFind(n)
    keys = [hash_key(c.text) for c in claims]

    # ---- 1. exact ----
    exact_pairs = 0
    first_seen: dict[str, int] = {}
    for i, k in enumerate(keys):
        if k in first_seen:
            uf.union(first_seen[k], i)
            exact_pairs += 1
        else:
            first_seen[k] = i

    # ---- 2. near-duplicate ----
    near_pairs = 0
    if near_duplicate and n > 1:
        grams = [char_ngrams(c.text, ngram) for c in claims]
        for i, j in candidate_pairs(grams, [c.language for c in claims], threshold):
            if uf.find(i) == uf.find(j):
                continue
            if jaccard(grams[i], grams[j]) >= threshold:
                uf.union(i, j)
                near_pairs += 1

    roots = [uf.find(i) for i in range(n)]
    remap: dict[int, int] = {}
    group_ids: list[int] = []
    for r in roots:
        if r not in remap:
            remap[r] = len(remap)
        group_ids.append(remap[r])

    sizes: dict[int, int] = {}
    for g in group_ids:
        sizes[g] = sizes.get(g, 0) + 1

    return GroupingResult(
        group_ids=group_ids,
        n_groups=len(remap),
        n_exact_duplicate_pairs=exact_pairs,
        n_near_duplicate_pairs=near_pairs,
        largest_group_size=max(sizes.values()) if sizes else 0,
        summary={
            "n_claims": n,
            "n_groups": len(remap),
            "n_claims_in_multi_claim_groups": sum(v for v in sizes.values() if v > 1),
            "n_multi_claim_groups": sum(1 for v in sizes.values() if v > 1),
            "exact_duplicate_pairs": exact_pairs,
            "near_duplicate_pairs": near_pairs,
            "largest_group_size": max(sizes.values()) if sizes else 0,
            "near_duplicate_threshold": threshold,
            "ngram": ngram,
        },
    )


# --------------------------------------------------------------------
# Group-aware splitting
# --------------------------------------------------------------------

def group_aware_split(
    claims: list[Claim],
    group_ids: list[int],
    ratios: dict[str, float],
    stratify_by: Iterable[str] = ("dataset", "language"),
    seed: int = 42,
) -> dict[str, list[int]]:
    """Split by GROUP, never by claim.

    Stratification is applied at group level using the group's majority
    stratum. Exact ratios are unattainable when groups vary in size --
    the guarantee this function makes is that no group is ever split,
    which is the property that matters for leakage. Realised ratios are
    reported by `verify_no_leakage`.
    """
    rng = np.random.default_rng(seed)

    members: dict[int, list[int]] = {}
    for idx, g in enumerate(group_ids):
        members.setdefault(g, []).append(idx)

    def stratum_of(group: int) -> tuple:
        counts: dict[tuple, int] = {}
        for i in members[group]:
            c = claims[i]
            key = tuple(str(getattr(c, f, "")) for f in stratify_by)
            counts[key] = counts.get(key, 0) + 1
        return max(counts.items(), key=lambda kv: kv[1])[0]

    by_stratum: dict[tuple, list[int]] = {}
    for g in members:
        by_stratum.setdefault(stratum_of(g), []).append(g)

    splits: dict[str, list[int]] = {k: [] for k in ratios}
    for _, groups in sorted(by_stratum.items()):
        groups = list(rng.permutation(groups))
        # Largest-group-first with running quotas keeps realised ratios
        # close even when a few groups are very large.
        groups.sort(key=lambda g: len(members[g]), reverse=True)
        total = sum(len(members[g]) for g in groups)
        quota = {k: ratios[k] * total for k in ratios}
        filled = {k: 0 for k in ratios}
        for g in groups:
            target = max(quota, key=lambda k: quota[k] - filled[k])
            splits[target].extend(members[g])
            filled[target] += len(members[g])

    return {k: sorted(v) for k, v in splits.items()}


# --------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------

def verify_no_leakage(
    claims: list[Claim],
    splits: dict[str, list[int]],
    group_ids: list[int],
    near_duplicate_threshold: float = 0.90,
    ngram: int = 5,
    check_cross_split_similarity: bool = True,
    max_similarity_pairs_to_report: int = 10,
) -> dict[str, Any]:
    """Independent re-verification of a finished split.

    Recomputes from the claims themselves rather than trusting the
    splitter's bookkeeping: a bug in the splitter and a self-check
    inside the splitter would fail together.

    Checks:
      1. every claim assigned exactly once
      2. no group spans two splits
      3. no exact-duplicate claim text across splits
      4. no cross-split pair above the near-duplicate threshold
      5. realised split ratios reported
    """
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str, data: Any = None) -> None:
        e = {"name": name, "passed": bool(passed), "detail": detail}
        if data is not None:
            e["data"] = data
        checks.append(e)

    n = len(claims)
    assigned: dict[int, list[str]] = {}
    for split, idxs in splits.items():
        for i in idxs:
            assigned.setdefault(i, []).append(split)

    missing = [i for i in range(n) if i not in assigned]
    multiple = {i: s for i, s in assigned.items() if len(s) > 1}
    add("every_claim_assigned_exactly_once", not missing and not multiple,
        f"missing={len(missing)}, assigned_more_than_once={len(multiple)}",
        {"n_missing": len(missing), "n_multiple": len(multiple)})

    # ---- groups must not span splits ----
    group_to_splits: dict[int, set[str]] = {}
    for split, idxs in splits.items():
        for i in idxs:
            group_to_splits.setdefault(group_ids[i], set()).add(split)
    spanning = {g: sorted(s) for g, s in group_to_splits.items() if len(s) > 1}
    add("no_group_spans_multiple_splits", not spanning,
        f"{len(spanning)} group(s) span multiple splits" if spanning
        else f"all {len(group_to_splits)} groups confined to one split",
        {"spanning_groups": dict(list(spanning.items())[:10])})

    # ---- exact text overlap ----
    keys_by_split = {s: {hash_key(claims[i].text) for i in idxs} for s, idxs in splits.items()}
    exact_overlap: dict[str, int] = {}
    names = sorted(keys_by_split)
    for a_pos, a in enumerate(names):
        for b in names[a_pos + 1 :]:
            exact_overlap[f"{a}_vs_{b}"] = len(keys_by_split[a] & keys_by_split[b])
    add("no_exact_claim_overlap_across_splits", all(v == 0 for v in exact_overlap.values()),
        f"{exact_overlap}", exact_overlap)

    # ---- near-duplicate overlap ----
    if check_cross_split_similarity:
        # Uses the same EXACT prefix/length filter as group_claims. The
        # previous version blocked on (language, length//25) and compared
        # every cross-split pair inside a bucket; on a single-language
        # corpus that is quadratic (measured O(n^2.0), ~1.7 h for FEVER).
        # The filter changes cost only -- no pair above the threshold can
        # be missed, so a PASS still means what it says.
        grams_list = [char_ngrams(claims[i].text, ngram) for i in range(n)]
        split_of: dict[int, str] = {}
        for s, idxs in splits.items():
            for i in idxs:
                split_of[i] = s
        offenders: list[dict[str, Any]] = []
        for i, j in candidate_pairs(grams_list, [c.language for c in claims],
                                    near_duplicate_threshold):
            a, b = split_of.get(i), split_of.get(j)
            if a is None or b is None or a == b:
                continue
            if jaccard(grams_list[i], grams_list[j]) >= near_duplicate_threshold:
                offenders.append({
                    "split_a": a, "split_b": b,
                    "claim_a": claims[i].claim_id, "claim_b": claims[j].claim_id,
                    "text_a": claims[i].text[:80], "text_b": claims[j].text[:80],
                })
                if len(offenders) >= max_similarity_pairs_to_report:
                    break
        add("no_near_duplicate_claims_across_splits", not offenders,
            "no cross-split pair at or above the near-duplicate threshold" if not offenders
            else f"{len(offenders)}+ near-duplicate cross-split pair(s) found",
            {"examples": offenders})

    # ---- realised ratios ----
    total = sum(len(v) for v in splits.values())
    realised = {s: round(len(v) / total, 4) if total else 0.0 for s, v in splits.items()}
    counts = {s: len(v) for s, v in splits.items()}
    add("split_ratios_reported", True, f"{counts} -> {realised}",
        {"counts": counts, "realised_ratios": realised})

    passed = all(c["passed"] for c in checks)
    return {
        "passed": passed,
        "n_checks": len(checks),
        "n_passed": sum(1 for c in checks if c["passed"]),
        "n_failed": sum(1 for c in checks if not c["passed"]),
        "checks": checks,
    }


SPLIT_PRECEDENCE = ("train", "validation", "test")


def enforce_group_safe_splits(
    claims: list[Claim],
    splits: dict[str, list[int]],
    group_ids: list[int],
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """Make official splits group-safe by moving whole groups, never claims.

    Why this is needed. The build preserves each corpus's published split
    verbatim and computes near-duplicate groups afterwards, independently.
    Nothing reconciled the two, so a group could straddle train and test
    whenever the corpus itself places duplicate claims on both sides --
    which FEVER does for a small number of claims.

    The repair. Any group occupying more than one split is assigned in its
    entirety to TRAIN. Evaluation sets therefore only ever shrink when
    leakage is repaired: a claim the corpus designated for training is
    never promoted into validation or test, which would otherwise let
    training data influence a reported score. Majority-wins would preserve
    split proportions slightly better but can move train claims into test,
    so it is deliberately not used here.

    Determinism, with no RNG. The target is a constant, groups are
    processed in sorted id order, and member indices are sorted, so
    rebuilding from the same inputs yields byte-identical splits. No seed
    is consumed; the config's `leakage.seed` keeps its existing meaning.
    `SPLIT_PRECEDENCE` is used only as a fallback when no split is
    literally named "train".

    What it does NOT do. No claim is deleted, no text/label/id/metadata is
    altered, and `metadata.official_split` is deliberately left holding
    the corpus's own published value -- `claim_splits.json` is the
    authoritative assignment, and keeping both preserves provenance. Every
    move is recorded in the returned report.

    Because `group_claims()` unions exact duplicates AND every pair at or
    above the near-duplicate threshold into one group, group-safety
    implies zero cross-split exact overlap and zero cross-split
    near-duplicate pairs. This one repair satisfies all three failing
    checks.

    Returns `(new_splits, report)`. A corpus whose splits are already
    group-safe is returned unchanged, with `n_groups_repaired == 0`.
    """
    split_of: dict[int, str] = {}
    for s, idxs in splits.items():
        for i in idxs:
            split_of[i] = s

    group_members: dict[int, list[int]] = {}
    for i, g in enumerate(group_ids):
        group_members.setdefault(g, []).append(i)

    # Contested groups always go to train. Fallback only if no split is
    # literally named "train" (e.g. a corpus using different split names).
    target_split = "train" if "train" in splits else next(
        (s for s in SPLIT_PRECEDENCE if s in splits), sorted(splits)[0] if splits else "train")

    moves: list[dict[str, Any]] = []
    reassigned: dict[int, str] = {}
    n_spanning = 0

    for g in sorted(group_members):
        members = sorted(group_members[g])
        by_split: dict[str, list[int]] = {}
        for i in members:
            s = split_of.get(i)
            if s is not None:
                by_split.setdefault(s, []).append(i)
        if len(by_split) <= 1:
            continue
        n_spanning += 1
        target = target_split
        for s, idxs in by_split.items():
            if s == target:
                continue
            for i in idxs:
                reassigned[i] = target
                moves.append({
                    "claim_id": claims[i].claim_id,
                    "claim_index": i,
                    "claim_group_id": int(g),
                    "group_size": len(members),
                    "from_split": s,
                    "to_split": target,
                    "member_counts": {k: len(v) for k, v in sorted(by_split.items())},
                })

    counts_before = {s: len(v) for s, v in sorted(splits.items())}
    new_splits: dict[str, list[int]] = {s: [] for s in splits}
    for i, s in split_of.items():
        new_splits.setdefault(reassigned.get(i, s), []).append(i)
    for s in new_splits:
        new_splits[s].sort()
    counts_after = {s: len(v) for s, v in sorted(new_splits.items())}

    report = {
        "policy": "contested_groups_to_train",
        "policy_detail": (f"A group spanning multiple splits is assigned whole to "
                          f"'{target_split}'. Evaluation splits only shrink; no claim is ever "
                          f"promoted into validation or test. Deterministic, no RNG."),
        "target_split": target_split,
        "n_groups_spanning_before": n_spanning,
        "n_groups_repaired": n_spanning,
        "n_claims_moved": len(moves),
        "counts_before": counts_before,
        "counts_after": counts_after,
        "delta": {s: counts_after.get(s, 0) - counts_before.get(s, 0) for s in counts_before},
        "claims_deleted": 0,
        "moves": moves[:200],
        "n_moves_recorded": min(len(moves), 200),
        "note": ("No claim was deleted and no claim content changed. "
                 "metadata.official_split still holds the corpus's published split; "
                 "claim_splits.json is the authoritative assignment."),
    }
    return new_splits, report


def build_index_scope(splits: dict[str, list[int]], phase: str, scope_config: dict[str, list[str]]) -> list[int]:
    """Claim indices permitted in the retrieval index for a given phase.

    Training must not be able to retrieve validation or test evidence.
    Returning explicit indices makes the scope auditable rather than
    implicit in how the index happens to be built.
    """
    allowed = scope_config.get(phase)
    if allowed is None:
        raise ValueError(f"No index scope configured for phase {phase!r}; "
                         f"known phases: {sorted(scope_config)}")
    out: list[int] = []
    for split in allowed:
        if split not in splits:
            raise ValueError(f"Index scope for phase {phase!r} references unknown split {split!r}")
        out.extend(splits[split])
    return sorted(out)
