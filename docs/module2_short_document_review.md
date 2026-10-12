# Module 2 — Short-Document Removal Review (Version A vs. Version B)

Requested review, performed before approving Module 2, into whether the
uniform 5-word minimum was silently damaging the (already scarce) Tamil
portion of the dataset. Short answer: **yes, it was** — this review found it,
quantified it, and Version B (language-aware threshold) has been adopted.

---

## 1–2. Analysis of every row removed by the 5-word threshold

Recomputed directly against the real, cleaned Module 1 output (495 rows
removed total):

| | English | Tamil | Total |
|---|---:|---:|---:|
| **Rows removed** | 69 | 426 | 495 |
| **% of that language's pre-filter pool** | 0.1% (69/63,673) | **7.9%** (426/5,416) | 0.7% |
| Real (label 0) | 68 | 345 | 413 |
| Fake (label 1) | 1 | 81 | 82 |
| By source | welfake: 69 | aanirudh_headlines: 424, dailythanthi: 2 | |

The removal rate for Tamil (7.9%) was **~79x** the removal rate for English
(0.1%) — a strong signal the threshold wasn't language-neutral in effect,
even though it was language-neutral in code.

## 3. Are the removed Tamil rows legitimate headlines or noise?

Manually inspected all 426 removed Tamil rows (word-count breakdown: 1 row
at 1 word, 42 at 2 words, 134 at 3 words, 249 at 4 words). Overwhelmingly
**legitimate, coherent, real news headlines** — examples pulled directly
from the removed set:

- "பில்கேட்ஸின் தந்தை காலமானார்" (3 words) — "Bill Gates' father passed away"
- "ஆஸ்திரேலியாவை மீண்டும் வீழ்த்தியது இங்கிலாந்து" (4 words) — "England defeats Australia again"
- "மைக்ரோசாப்டிலிருந்து பில் கேட்ஸ் விலகல்" (4 words) — "Bill Gates steps down from Microsoft"
- "லேப்டாப் வியாபாரத்திலிருந்து வெளியேறும் டோஷிபா" (4 words) — "Toshiba exits the laptop business"

These are real, substantive, fact-checkable claims — exactly the kind of
content a fake-news classifier should train on. Only **1 of 426** rows was
genuinely uninformative on inspection: a single bare word, "பாப்கார்ன்"
("Popcorn"), with no claim to evaluate.

Compare this to the 69 removed **English** rows, which were manually
confirmed as genuine noise: 5 rows that cleaned down to nothing (empty
WELFake entries), and fragments like "horseshit", "ludicrous...", "agreed.",
"obamas blunder." — comment/reaction snippets, not article content. The
English threshold is doing its job correctly; the Tamil threshold was not.

**Root cause:** Tamil is agglutinative — a single whitespace-delimited Tamil
"word" routinely carries what would be 2-3 English words' worth of
grammatical/morphological information (case markers, postpositions, and
compounding are fused onto the stem rather than written as separate words).
Tamil news headlines are also conventionally terse. A raw word-count
threshold tuned against English article prose systematically
under-measures how much content a short Tamil string actually carries.

## 4–5. Recommended strategy and comparison

**Recommendation:** language-aware minimum word count — English stays at 5
(validated as correctly separating signal from noise), Tamil lowered to 2
(keeps every legitimate headline found in review, still drops the one
genuinely content-free entry).

Full pipeline re-run for both versions, real numbers:

| | Version A (uniform, min=5) | Version B (language-aware, en=5/ta=2) |
|---|---:|---:|
| Final rows | 68,584 | **69,008** |
| English | 63,600 | 63,600 (unchanged) |
| Tamil | 4,984 | **5,408** (+424) |
| Fake (1) | 37,623 | 37,704 |
| Real (0) | 30,961 | 31,304 |
| Train / Val / Test | 48,009 / 10,287 / 10,288 | 48,305 / 10,351 / 10,352 |
| Leakage check | Passed | Passed |

Of the 426 originally-removed Tamil rows, Version B recovers **424** — the
only 2 that remain excluded are the single "Popcorn" row (correctly, still
below the 2-word floor) and one row that turned out to be an exact
post-cleaning duplicate of another headline (caught by duplicate
verification, not the length filter — working as intended).

## 6. Which version is more appropriate for an IEEE/Springer multilingual paper

**Version B.** Three reasons, in order of how much they matter for this
specific paper:

1. **This is a low-resource-language paper; the low-resource language is
   what's being evaluated.** Tamil started with only 5,417 rows in the
   entire project (vs. 63,672 English) — already a 12:1 imbalance the paper
   has to defend. Discarding a further 7.9% of that scarce resource, using a
   rule that turns out to encode an English-specific assumption, directly
   undermines the paper's own stated contribution (multilingual, low-resource
   fake news detection). Reviewers evaluating a low-resource-language claim
   are exactly the audience most likely to notice and penalize this.
2. **The removed data has real classification signal, confirmed by
   inspection, not assumed.** These aren't ambiguous edge cases — "Bill
   Gates' father passed away" and "England defeats Australia again" are
   genuine, checkable claims. Dropping them for being "too short" when
   English rows of similar semantic density are kept is an inconsistency a
   careful reviewer could flag as an unjustified methodological choice.
3. **English is unaffected — this isn't a tradeoff.** Version B changes
   nothing about English's threshold or the 69 rows correctly removed
   there. There's no quality cost to English for the Tamil gain, which is
   why "language-aware" is a stronger choice than either "keep everything"
   (would readmit real noise into English) or "keep the uniform rule"
   (measurably discards good Tamil data).

The only argument for Version A would be simplicity/uniformity for its own
sake — not a strong argument against measured, inspected, reversible harm to
the exact data this paper is about.

## Outcome

Version B adopted and **approved by the user on 2026-08-06**. Module 1 was
not touched. Module 2's outputs were regenerated with the language-aware
threshold and verified directly against the files on disk (row counts,
language/label distributions, zero data leakage, Tamil Unicode NFC
correctness) — see `docs/module2_completion_report.md` for the final,
current numbers and full file list. `configs/module2_preprocessing.yaml`
documents this decision inline for anyone re-running the pipeline later.
Module 2 is now finalized; it will not be modified further unless a
critical bug is discovered.
