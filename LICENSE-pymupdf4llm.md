# pymupdf4llm License Notice

RAGCore uses [pymupdf4llm](https://github.com/pymupdf/PyMuPDF) for PDF parsing. pymupdf4llm is built on PyMuPDF (Artifex), which is dual-licensed:

- **AGPL-3.0**: free for open-source projects whose own license is compatible with AGPL-3.0
- **Commercial license**: required for proprietary or non-AGPL network services that embed PyMuPDF; available from Artifex Software

## What this means for RAGCore

RAGCore is licensed MIT. The pymupdf4llm dependency does not relicense RAGCore — MIT and AGPL coexist at the dependency level. However, anyone deploying RAGCore as a network service is bound by AGPL-3.0's § 13 obligation to provide source code to network users, including any modifications.

## Why pymupdf4llm was chosen

An empirical bake-off (2026-05-27) compared three PDF parsers — pdfplumber with extract_tables(), pymupdf4llm, and unstructured — on two real documents (Apple 10-K, Apple Environmental Progress Report). pymupdf4llm was the only candidate that handled both:
- the 10-K's Consolidated Statements of Operations with column structure preserved
- the Environmental Progress Report's image-embedded data without footnote text being spliced into data labels

The license trade-off was accepted for this project because (a) RAGCore is a public open-source portfolio project, not a proprietary product, and (b) the quality difference on the demo corpus was material, not marginal.

## Alternatives

If AGPL is unacceptable for a deployment, two paths:

- Replace pymupdf4llm with `pdfplumber` + `extract_tables()` and accept degraded parsing on graphic-heavy documents
- Purchase a commercial PyMuPDF license from Artifex Software
