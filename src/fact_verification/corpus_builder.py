"""Module 5 Stage 3 — unified claim/evidence corpus construction.

Converts the downloaded X-FACT and FEVER JSONL files into one internal
schema so retrieval, reranking and verification all consume the same
record shape. This is the "Claim Representation" stage: it sits between
claim selection (Stage 1) and evidence retrieval, and produces the
`claims.jsonl` / `evidence.jsonl` / `claim_splits.json` artefacts the
config already declares.

Design commitments carried from earlier stages:

  * **Official splits are preserved.** X-FACT ships train/dev/test and
    FEVER ships train/validation/test. Those are the published
    evaluation protocol; reshuffling them would make results
    incomparable with prior work. The group-aware 70/15/15 splitter
    exists for corpora WITHOUT a usable official split and is not
    applied here.
  * **Datasets stay separable.** Every record carries `dataset`, so
    X-FACT and FEVER results can be reported apart and domain mismatch
    stays visible.
  * **Leakage is measured, not assumed.** The gate runs against the
    official splits. Failures there are a property of the published
    dataset and are reported as such — they are not silently repaired
    by re-splitting.
  * **Evidence identifiers are preserved** so FEVER's link back to
    Wikipedia survives, and the closed-pool caveat travels with the data.

Field discovery duplicates a small number of pure helpers that also
exist in `scripts/audit_module5_corpora.py`. The audit script is a
verified, passing artefact and was deliberately not edited; instead
`tests/test_fact_verification.py` asserts the two implementations agree
on the same inputs, so the copies cannot drift apart unnoticed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from src.fact_verification.claim_selection import Claim, normalize_claim, tamil_ratio

# Field aliases, kept identical to the audit script's lists.
CLAIM_FIELDS = ["claim", "claim_text", "text", "statement", "sentence"]
LABEL_FIELDS = ["label", "labels", "verdict", "gold_label", "class"]
LANG_FIELDS = ["language", "lang", "claim_language"]
SITE_FIELDS = ["site", "source", "domain", "fact_checker", "url"]
ID_FIELDS = ["id", "claim_id", "example_id", "uid"]
EVIDENCE_LIST_FIELDS = ["evidence", "evidences", "gold_evidence", "evidence_sentences"]
EVIDENCE_NUMBERED_PREFIX = "evidence_"

#: Canonical split names. X-FACT uses "dev", FEVER uses "validation";
#: both are mapped to "validation" so downstream code has one vocabulary,
#: while `official_split` retains the corpus's own name for traceability.
SPLIT_ALIASES = {"dev": "validation", "val": "validation", "valid": "validation",
                 "train": "train", "validation": "validation", "test": "test"}


def first_present(row: dict[str, Any], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in row and row[c] not in (None, ""):
            return c
    for c in candidates:
        if c in row:
            return c
    return None


def extract_evidence(row: dict[str, Any]) -> tuple[list[str], str]:
    """Evidence text from any of the observed record shapes."""
    for f in EVIDENCE_LIST_FIELDS:
        if f in row and row[f]:
            val = row[f]
            if isinstance(val, str):
                return ([val], f"string:{f}")
            if isinstance(val, list):
                out: list[str] = []
                for item in val:
                    if isinstance(item, str):
                        out.append(item)
                    elif isinstance(item, dict):
                        for k in ("text", "sentence", "evidence"):
                            if k in item and isinstance(item[k], str):
                                out.append(item[k])
                                break
                    elif isinstance(item, (list, tuple)):
                        strs = [x for x in item if isinstance(x, str)]
                        if strs:
                            out.append(strs[-1])
                return ([e for e in out if e and e.strip()], f"list:{f}")

    numbered = sorted(k for k in row if k.startswith(EVIDENCE_NUMBERED_PREFIX))
    if numbered:
        out = [str(row[k]) for k in numbered
               if row.get(k) not in (None, "", "nan") and str(row[k]).strip()]
        return (out, f"numbered:{len(numbered)}_columns")
    return ([], "none_found")


def extract_evidence_ids(row: dict[str, Any]) -> list[str]:
    """Original page/sentence identifiers, where the corpus supplies them."""
    ids: list[str] = []
    for f in EVIDENCE_LIST_FIELDS:
        val = row.get(f)
        if isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    page = item.get("page") or item.get("title") or item.get("doc_id")
                    sent = item.get("sentence_id", item.get("line", item.get("sent_id")))
                    if page is not None:
                        ids.append(f"{page}#{sent}" if sent is not None else str(page))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    non_str = [x for x in item if not isinstance(x, str)]
                    strs = [x for x in item if isinstance(x, str)]
                    if strs:
                        ids.append(f"{strs[0]}#{non_str[0]}" if non_str else strs[0])
    for k in ("evidence_id", "doc_id", "page", "wikipedia_url"):
        if row.get(k):
            ids.append(str(row[k]))
    return ids


def canonical_split(raw: str) -> str:
    return SPLIT_ALIASES.get(raw.strip().lower(), raw.strip().lower())


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evidence_hash(normalized_text: str) -> str:
    """Deterministic SHA-256 digest of already-normalized evidence text.

    Must be a real digest, not a truncated copy of the text. An earlier
    version built the id as `hash_key(text)[:64]`, but `hash_key()`
    returns the normalized *string*, so the "id" was just the first 64
    characters of the sentence. Two distinct evidence sentences sharing a
    64-character prefix — routine with news boilerplate leads such as
    "According to a statement released by the ministry on Monday, ..." —
    would collide and be silently merged into one pool entry, destroying
    a distinct piece of evidence and corrupting `n_claims_referencing`.

    The input is the normalized text, so the digest inherits whatever
    normalization the pipeline already applies; normalization behaviour
    itself is unchanged. Encoding is fixed to UTF-8 so Tamil text hashes
    identically across platforms.
    """
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


@dataclass
class EvidenceItem:
    """One evidence sentence in the shared pool.

    `evidence_id` is content-derived so identical text from different
    claims collapses to one pool entry — the retrieval index should not
    contain the same sentence many times. `source_ids` accumulates every
    original identifier that produced it, preserving provenance.
    """

    evidence_id: str
    text: str
    dataset: str
    language: str
    source_ids: list[str] = field(default_factory=list)
    n_claims_referencing: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"evidence_id": self.evidence_id, "text": self.text, "dataset": self.dataset,
                "language": self.language, "source_ids": sorted(set(self.source_ids))[:20],
                "n_claims_referencing": self.n_claims_referencing}


def detect_language(text: str, declared: str | None = None) -> str:
    """Declared language wins; script is the fallback, never an override."""
    if declared:
        d = str(declared).strip().lower()
        if d:
            return d
    if not text or not text.strip():
        return "unknown"
    return "ta" if tamil_ratio(text) > 0.3 else "en"


def build_corpus_from_files(
    files: Iterable[tuple[Path, str]],
    dataset: str,
    config: dict[str, Any],
) -> tuple[list[Claim], dict[str, EvidenceItem], dict[str, Any]]:
    """Build claims + a de-duplicated evidence pool for one corpus.

    `files` is an iterable of (path, official_split_name). Every input
    row produces exactly one claim record unless it has no usable claim
    text, and every drop is counted so row loss cannot pass unnoticed.
    """
    norm_cfg = config["normalization"]
    cs_cfg = config["claim_selection"]

    claims: list[Claim] = []
    evidence_pool: dict[str, EvidenceItem] = {}
    stats: dict[str, Any] = {
        "dataset": dataset, "n_input_rows": 0, "n_claims_built": 0,
        "n_dropped_no_claim_text": 0, "n_dropped_below_min_chars": 0,
        "by_official_split": {}, "by_canonical_split": {}, "by_language": {},
        "by_label": {}, "evidence_extraction_methods": {},
        "n_claims_with_no_evidence": 0, "n_evidence_mentions": 0,
        "field_mapping_per_file": {},
    }

    for path, official_split in files:
        rows = load_jsonl(path)
        stats["n_input_rows"] += len(rows)
        stats["by_official_split"][official_split] = len(rows)
        if not rows:
            continue

        sample = rows[0]
        cf = first_present(sample, CLAIM_FIELDS)
        lf = first_present(sample, LABEL_FIELDS)
        gf = first_present(sample, LANG_FIELDS)
        sf = first_present(sample, SITE_FIELDS)
        idf = first_present(sample, ID_FIELDS)
        stats["field_mapping_per_file"][path.name] = {
            "claim": cf, "label": lf, "language": gf, "site": sf, "id": idf,
            "columns": list(sample.keys()),
        }

        canon = canonical_split(official_split)
        for i, row in enumerate(rows):
            raw_claim = str(row.get(cf, "")) if cf else ""
            if not raw_claim.strip():
                stats["n_dropped_no_claim_text"] += 1
                continue

            text = normalize_claim(raw_claim, norm_cfg)
            if cs_cfg.get("truncate_long_titles", True) and len(text) > cs_cfg["max_claim_chars"]:
                text = text[: cs_cfg["max_claim_chars"]].rstrip()
            if cs_cfg.get("drop_claims_below_min", True) and len(text) < cs_cfg["min_claim_chars"]:
                stats["n_dropped_below_min_chars"] += 1
                continue

            language = detect_language(text, row.get(gf) if gf else None)
            native_id = str(row.get(idf)) if idf and row.get(idf) is not None else f"{official_split}_{i}"

            ev_texts, method = extract_evidence(row)
            stats["evidence_extraction_methods"][method] = \
                stats["evidence_extraction_methods"].get(method, 0) + 1
            ev_ids_native = extract_evidence_ids(row)

            evidence_refs: list[str] = []
            for e in ev_texts:
                e_norm = normalize_claim(e, norm_cfg)
                if not e_norm.strip():
                    continue
                eid = f"{dataset}:ev:{evidence_hash(e_norm)}"
                item = evidence_pool.get(eid)
                if item is None:
                    item = EvidenceItem(evidence_id=eid, text=e_norm, dataset=dataset,
                                        language=detect_language(e_norm))
                    evidence_pool[eid] = item
                item.n_claims_referencing += 1
                item.source_ids.extend(ev_ids_native)
                evidence_refs.append(eid)
                stats["n_evidence_mentions"] += 1

            if not evidence_refs:
                stats["n_claims_with_no_evidence"] += 1

            claims.append(Claim(
                claim_id=f"{dataset}:{official_split}:{native_id}",
                text=text, raw_text=raw_claim, language=language, dataset=dataset,
                source=str(row.get(sf))[:120] if sf and row.get(sf) else None,
                label=row.get(lf) if lf else None,
                record_id=native_id,
                url=row.get("url") if isinstance(row.get("url"), str) else None,
                evidence=evidence_refs,
                selection_rule="native_claim_field",
                metadata={
                    "official_split": official_split,
                    "canonical_split": canon,
                    "evidence_extraction_method": method,
                    "n_evidence": len(evidence_refs),
                    "native_evidence_ids": ev_ids_native[:10],
                    "tamil_ratio": round(tamil_ratio(text), 4),
                },
            ))

            stats["by_canonical_split"][canon] = stats["by_canonical_split"].get(canon, 0) + 1
            stats["by_language"][language] = stats["by_language"].get(language, 0) + 1
            lab = str(row.get(lf)) if lf and row.get(lf) is not None else "MISSING"
            stats["by_label"][lab] = stats["by_label"].get(lab, 0) + 1

    stats["n_claims_built"] = len(claims)
    stats["n_unique_evidence_items"] = len(evidence_pool)
    stats["row_accounting_balances"] = (
        stats["n_input_rows"]
        == stats["n_claims_built"] + stats["n_dropped_no_claim_text"] + stats["n_dropped_below_min_chars"]
    )
    return claims, evidence_pool, stats


def splits_from_claims(claims: list[Claim]) -> dict[str, list[int]]:
    """Index map from the OFFICIAL splits already carried on each claim.

    No reshuffling: this reads the published protocol rather than
    imposing a new one.
    """
    out: dict[str, list[int]] = {}
    for i, c in enumerate(claims):
        out.setdefault(str(c.metadata.get("canonical_split", "unknown")), []).append(i)
    return out


def write_corpus(
    claims: list[Claim],
    evidence_pool: dict[str, EvidenceItem],
    splits: dict[str, list[int]],
    group_ids: list[int],
    out_dir: Path,
    claims_filename: str,
    evidence_filename: str = "evidence.jsonl",
    splits_filename: str = "claim_splits.json",
) -> dict[str, str]:
    """Persist the corpus. Writes only under `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)

    claims_path = out_dir / claims_filename
    with claims_path.open("w", encoding="utf-8") as f:
        for i, c in enumerate(claims):
            d = c.to_dict()
            d["claim_group_id"] = int(group_ids[i])
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    evidence_path = out_dir / evidence_filename
    with evidence_path.open("w", encoding="utf-8") as f:
        for item in evidence_pool.values():
            f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

    splits_path = out_dir / splits_filename
    splits_path.write_text(json.dumps({
        "note": ("Indices into claims.jsonl (line order). Splits are the corpora's OWN "
                 "official splits; they were not reshuffled."),
        "splits": splits,
        "counts": {k: len(v) for k, v in splits.items()},
    }, indent=2), encoding="utf-8")

    return {"claims": str(claims_path), "evidence": str(evidence_path), "splits": str(splits_path)}
