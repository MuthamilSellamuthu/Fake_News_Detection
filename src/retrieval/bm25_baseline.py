"""Module 6 — BM25 lexical retrieval baseline.

Why a lexical baseline matters here: if BM25 matches or beats LaBSE, the
dense retriever is not earning its place in the pipeline, and reporting only
the dense numbers would overstate the contribution. This is the honest
control for the retrieval stage.

Tamil-safe tokenisation. `rank_bm25` and most BM25 tutorials split on
whitespace after `.lower()`, which is wrong for this corpus twice over:
Tamil has no case, and Tamil punctuation/ZWNJ handling differs from Latin.
Tokenisation here uses a Unicode word-character regex, lowercases Latin only
(mirroring Module 2 and `claim_selection.normalize_claim`), and never applies
NFKC -- which can collapse linguistically distinct Tamil characters.

Implemented directly rather than via `rank_bm25` so the project gains no new
dependency and the scoring is auditable. `tests/test_retrieval.py` checks the
scores against hand-computed BM25 values.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any, Iterable, Sequence

_TAMIL_LO, _TAMIL_HI = "஀", "௿"

# `\w` alone is WRONG for Tamil. Python's `\w` follows str.isalnum(), which is
# False for combining marks (categories Mn/Mc) -- and Tamil vowel signs and the
# virama are exactly that. Measured: r"\w+" tokenises
# "தாக்குதல்" as ['த','க','க','தல'], shattering the word at every matra and
# silently dropping U+0BBE, U+0BBF, U+0BC1, U+0BC2 and U+0BCD. BM25 over Tamil
# would then match on meaningless consonant fragments.
# The Tamil block and general combining diacritics are therefore added
# explicitly so a word survives as one token.
_WORD_RE = re.compile(r"[\w஀-௿̀-ͯ‌‍]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Unicode word tokens; Latin lowercased, Tamil left untouched."""
    if not text:
        return []
    s = unicodedata.normalize("NFC", text)      # NFC only, never NFKC
    out = []
    for tok in _WORD_RE.findall(s):
        # Lowercasing is a no-op for Tamil but would be wrong to skip for Latin.
        out.append("".join(ch if _TAMIL_LO <= ch <= _TAMIL_HI else ch.lower() for ch in tok))
    return out


class BM25:
    """Okapi BM25 with the standard IDF variant used by Robertson et al.

    IDF is floored at zero. Without the floor, a term appearing in more than
    half the corpus gets a negative IDF and a document can be *penalised* for
    containing a query term, which produces nonsensical rankings on small
    evidence pools like X-FACT's 6,968 items.
    """

    def __init__(self, corpus_tokens: Sequence[Sequence[str]], k1: float = 1.5,
                 b: float = 0.75) -> None:
        self.k1, self.b = float(k1), float(b)
        self.corpus_tokens = [list(d) for d in corpus_tokens]
        self.n_docs = len(self.corpus_tokens)
        self.doc_len = [len(d) for d in self.corpus_tokens]
        self.avg_len = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        self.tf: list[Counter] = [Counter(d) for d in self.corpus_tokens]
        df: Counter = Counter()
        for t in self.tf:
            df.update(t.keys())
        self.df = df
        self.idf = {
            term: max(0.0, math.log((self.n_docs - n + 0.5) / (n + 0.5) + 1.0))
            for term, n in df.items()
        }

    def score(self, query_tokens: Sequence[str], doc_index: int) -> float:
        if not self.n_docs:
            return 0.0
        tf, dl = self.tf[doc_index], self.doc_len[doc_index]
        denom_norm = self.k1 * (1 - self.b + self.b * (dl / self.avg_len if self.avg_len else 0.0))
        total = 0.0
        for term in query_tokens:
            f = tf.get(term, 0)
            if not f:
                continue
            total += self.idf.get(term, 0.0) * (f * (self.k1 + 1)) / (f + denom_norm)
        return total

    def scores(self, query_tokens: Sequence[str]) -> list[float]:
        return [self.score(query_tokens, i) for i in range(self.n_docs)]


class BM25Retriever:
    """BM25 over an evidence pool restricted to a phase's permitted ids."""

    def __init__(self, evidence_ids: Sequence[str], texts: Sequence[str],
                 k1: float = 1.5, b: float = 0.75) -> None:
        if len(evidence_ids) != len(texts):
            raise ValueError(f"{len(evidence_ids)} ids but {len(texts)} texts")
        self.evidence_ids = list(evidence_ids)
        self.bm25 = BM25([tokenize(t) for t in texts], k1=k1, b=b)

    def __len__(self) -> int:
        return len(self.evidence_ids)

    def retrieve(self, query: str, top_k: int) -> tuple[list[str], list[float]]:
        """Top-k ids, ties broken by index order so results are deterministic."""
        scores = self.bm25.scores(tokenize(query))
        k = min(int(top_k), len(self.evidence_ids))
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        return [self.evidence_ids[i] for i in order], [scores[i] for i in order]

    def retrieve_many(self, queries: Iterable[str], top_k: int
                      ) -> list[tuple[list[str], list[float]]]:
        return [self.retrieve(q, top_k) for q in queries]
