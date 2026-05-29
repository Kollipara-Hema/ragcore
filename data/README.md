# data/

Local-first corpus assembly area. As of 2026-05-28 both `apple_demo/`
(the source documents) and `chroma_collections/` (the built per-corpus
Chroma persist dirs) are tracked in git and ship in the Docker image; the
fetch script remains useful for re-materializing sources from an external
staging directory on a fresh box.

## Re-materializing sources from an external staging dir

The 5 source files originate from an external staging directory; the
`scripts/fetch_apple_corpus.py` helper copies them deterministically into
`apple_demo/`. The script is still the supported way to rebuild the source
set on a fresh box or when the staging dir is the source of truth — even
though `apple_demo/` is now tracked in the repo, the staging dir remains
authoritative for provenance.

## Deployment posture

The Apple corpora deploy to Render via Setup B: collections shipped in-repo,
bundled into the Docker image, and seeded onto the persistent disk on first
boot. See [README §Deployment](../README.md#deployment) for the mechanism.

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

## chroma_collections/

`data/chroma_collections/` holds per-corpus Chroma persist directories — one
subdirectory per entry in `config/corpora.py`, each containing a
`chroma.sqlite3`, a `bm25_state.pkl` sidecar for hybrid retrieval, and a
UUID-named HNSW segment subdir. It is tracked in git and shipped in the
Docker image; on a fresh persistent disk in production the lifespan seeder
copies each corpus directory onto the disk on first boot, and subsequent
deploys do not overwrite already-seeded dirs — see
[README §Deployment](../README.md#deployment) for the mechanism and how to
push an updated collection.

Regenerating locally:

```bash
python scripts/ingest_apple_corpus.py
```

The script reads `CORPORA_CONFIG`, runs the chunker + embedder + indexer
once per corpus, and writes into `data/chroma_collections/<corpus_name>/`.
Expected wall-clock: 7–12 minutes (embedding dominates). Re-running
overwrites in place — Chroma upserts by chunk_id, so a clean redo benefits
from wiping the directory first.

Apple corpora are served from these collections; FiQA (the "default" corpus
on FAISS) lives outside this directory and is unaffected.

## Multi-corpus naming and the hierarchical caveat

The corpus name `apple_10k_hierarchical` reflects the chunker class
(`HierarchicalChunker`) used at ingestion time, not retrieve-child-return-
parent semantics. The pipeline drops parent chunks before embedding and no
retrieval code expands matched children back to their parents, so this
corpus effectively behaves as a fixed-size child corpus over the 10-K.
Wiring real parent-child retrieval is a planned follow-up.
