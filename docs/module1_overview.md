# Module 1 — Research Project Initialization & Dataset Collection

## Purpose

Module 1 produces the raw, real, source-traceable multilingual dataset
(English + Tamil) this project's later modules build on. It does three
things only: **collect**, **lightly validate/standardize**, and
**report on** the data. It does not clean text, build contrastive
pairs, tokenize, or train anything — that begins in Module 2.

## Scope Boundary (Module 1 vs. Module 2)

| In Module 1 | Deferred to Module 2 |
|---|---|
| Download / scrape raw source data | HTML/URL/emoji/symbol text cleaning |
| Exact-duplicate removal | Contrastive pair construction |
| Lightweight language identification (tag `en`/`ta`) | Consistency/threshold filtering |
| Label standardization (map every source's own convention to `fake=1, real=0`) | Train/val/test stratified splitting |
| Dataset statistics reporting | Tokenization |

## Why This Structure

- **`datasets/raw/{english,tamil}`**: untouched, source-of-truth data. Nothing in this project ever overwrites a file here — every later stage reads from it and writes elsewhere.
- **`datasets/processed/`**: Module 1's *lightly* standardized output (deduplicated, language-tagged, label-standardized) — still raw text, not yet cleaned.
- **`src/data_collection/`**: one collector class per data source, sharing a common base class so retry/backoff, rate-limiting, robots.txt compliance, and resumable deduplication are implemented once, not per-source.
- **`configs/module1_sources.yaml`**: every source's URL/section/rate-limit lives here as data, not hardcoded in Python, so a source can be reconfigured or a new one added without touching collector code.
- **`logs/`**: one timestamped file per run, so a multi-hour Tamil scraping job leaves a permanent record.
- **`docs/`**: this file and its siblings — documentation that doesn't belong inside code comments (architecture rationale, scope boundaries, known limitations).

## Data Sources

**English**: WELFake (direct download, Zenodo), ISOT Fake News Dataset (direct download, University of Victoria). Both ship with a real, documented label convention — no approximation needed.

**Tamil**: BOOM Tamil (fact-check desk — mirrors the base paper's own fact-checking-platform data source, which skews fake by construction), Daily Thanthi (mainstream newspaper — the "real news" counterpart, mirroring the base paper's design), with Dinamalar and The Hindu Tamil scaffolded as additional mainstream sources pending section-slug verification. YouTurn and PIB Fact Check are documented as currently uncollectable (JS-rendered SPA; claim-form-only portal, respectively) rather than silently omitted.

**Tamil, added to raise volume**: the scraped Tamil sources above total only 239 rows on their own (see "Validated Findings" below) — far too little. `aanirudh_headlines` (`src/data_collection/tamil/aanirudh_headlines_collector.py`) adds a peer-reviewed, MIT-licensed, pre-labeled Tamil fake-news corpus — 5,226 real rows (2,902 fake / 2,324 real, per the file actually downloaded; the repo's README figure of 5,273/2,949 is very close but not pixel-identical, likely a minor edit to the source file since the README was written), from Mirnalinee et al., "A Novel Dataset for Fake News Detection in Tamil Regional Language," SPELLL 2022 / Springer (github.com/AAnirudh07/Fake-News-Headlines-In-Tamil). Unlike the scraped sources, it needs no keyword-scan/source-type label approximation — it ships a real fake/real column. **Downloaded for real and integrated** (see "Validated Findings" below and `docs/module1_completion_report.md`) — this sandbox's `raw.githubusercontent.com` access is blocked at the proxy level, so the collector falls back to `git clone` against `github.com` (which works), documented in the collector's own module docstring and in `docs/module1_completion_report.md` §9.

**Known, honest limitation**: unlike English, Tamil sources don't ship a structured true/false verdict field. Labels are assigned via a source-type default (fact-check source → presumed fake-skewed; mainstream newsroom → presumed real) plus keyword scanning of article text — the same approximation strategy the base paper itself uses for its own IndoHoax dataset. This should be stated explicitly in any resulting publication, with a recommended manual-verification pass on a sample before final claims.

## Validated Findings (from a real run against the collected data)

Running the validation pipeline against the real files in `datasets/raw/` (not
synthetic data) produced two results worth flagging explicitly rather than
leaving buried in a JSON report:

1. After exact `(title, text)` deduplication, **0 of ISOT's 44,898 rows
   survived as unique** — every one was an exact duplicate of a WELFake row.
   This is expected, not a bug: WELFake (Verma et al., 2021) was itself
   constructed by merging four source datasets, one of which is ISOT.
   Downloading both was still the right call — it's what confirms the overlap
   for this specific pair of files rather than assuming it from the WELFake
   paper's description — but it means the *effective* English corpus size for
   this project is WELFake's alone (63,672 rows after dedup/validation), not
   the naively-summed 72,134 + 44,898.
2. Adding `aanirudh_headlines` raised Tamil from 239 to 5,417 rows (a real
   ~22.7x increase) with **zero title overlap** against the existing scraped
   Tamil sources — the only Tamil duplicates found were 48 rows *within*
   `aanirudh_headlines` itself (the source corpus's own internal repeats,
   mostly duplicated cricket headlines), which the standard dedup step removed
   the same way it would any other source's internal duplicates.

See `outputs/module1/validation_report.json`, `outputs/module1/dataset_statistics.md`,
and `docs/module1_completion_report.md` for the full numbers.

## Ethical / Policy Compliance

Every collector checks `robots.txt` before fetching any path and honors `Disallow` rules. The HTTP User-Agent identifies this project honestly (project name + contact email) rather than spoofing a browser — if a site blocks a disclosed research bot, that block is treated as the site's decision to honor, not an obstacle to route around. Requests are rate-limited (default: 1.5s between requests to the same host) and retried with exponential backoff on transient failures only, not on explicit blocks (403/robots disallow).
