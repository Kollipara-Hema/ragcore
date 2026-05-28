# data/

Locally-materialized corpora. Subdirectories are corpus-specific and gitignored
(only this README is tracked); fresh checkouts must run a fetch script to
populate them before the backend can serve those corpora.

## Why this directory is gitignored

The corpus payload is binary (PDF / HTML / CSV), produces no useful diffs, and
the Apple environmental report alone is ~22 MB. Tracking it would bloat the
repo without giving anyone meaningful version control over the contents. The
authoritative copies live outside the repo (the staging directory documented
below); the fetch script copies them in deterministically.

This differs from the FiQA benchmark dataset at `evaluation/datasets/`, which
is tracked as committed JSON because it's small (~228 KB) and benchmark-
critical (CI and the eval scripts both depend on its exact contents).

## Deployment posture

The Apple multi-corpus demo is built local-first. The backend ingests these
corpora into local Chroma collections; the UI queries them via a local
backend. Deploying the Apple corpora to Render (Setup B) is a planned
follow-up that requires solving corpus delivery to Render's disk and
validating free-tier memory headroom with multiple Chroma collections plus
the embedding model loaded simultaneously. Until then, Render production
serves the FiQA default corpus.

## apple_demo/

The 5 files below populate `data/apple_demo/`. They come from the Apple
corpus assembly directory; FY2025 is chosen for cross-source alignment
(10-K, earnings release, environmental report) and matches the financial
metrics extracted from the SEC companyfacts JSON for the same fiscal year.

| Filename                                                | Size    | Provenance                                                     |
|---------------------------------------------------------|---------|----------------------------------------------------------------|
| `apple__sec__form_10k__2025.pdf`                        | 2.5 MB  | SEC EDGAR — Apple FY2025 Form 10-K, printed from EDGAR HTML    |
| `apple__sec__q4_earnings_release__2025.html`            | 194 KB  | SEC EDGAR — Apple FY2025 Q4 earnings release (8-K exhibit)     |
| `apple__corporate__environmental_progress_report__2025.pdf` | 21.9 MB | apple.com/environment — 2025 Environmental Progress Report     |
| `apple__sec__financial_metrics_quarterly__2026.csv`     | 1.4 KB  | Extracted from SEC companyfacts XBRL JSON (quarterly figures)  |
| `apple__sec__financial_metrics_annual__2026.csv`        | 0.5 KB  | Extracted from SEC companyfacts XBRL JSON (annual figures)     |

Explicitly excluded from the corpus:

- `apple__sec__companyfacts_raw__2026.json` — raw XBRL source for the CSVs;
  the structured CSVs are what the corpus serves.
- `apple__sec__form_10k__2025.html` — duplicate of the 10-K PDF in another
  format; the PDF is the canonical version.
- `apple__corporate__environmental_progress_report__2026.pdf` — 2026 edition;
  the demo uses 2025 for FY2025 alignment with the 10-K.

## Populating apple_demo/

```bash
python scripts/fetch_apple_corpus.py
```

Defaults to `--source /Users/hemakollipara/Desktop/files_ragcore/` and copies
into `data/apple_demo/`. Override either with `--source <path>` or
`--dest <path>`. The script verifies every source file exists before copying
anything (no partial state on failure) and verifies size after each copy.
