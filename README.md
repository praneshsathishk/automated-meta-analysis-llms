# Automated Meta-Analysis using LLMs

This repo implements a full pipeline for automating meta-analysis using large language models. Each step of the process is modular and customizable.

## 💡 Pipeline Overview

1. **Keyword Generation** — Uses LLM to create PubMed search terms
2. **PubMed Search** — Queries PubMed via Entrez API
3. **Abstract Screening** — LLM filters relevant studies
4. **Full-text Retrieval** — Downloads open-access full-text via PMCID
5. **Full-text Chunking** — Splits articles into chunks
6. **Vectorization** — Converts chunks to embeddings
7. **Full-text Screening** — LLM filters relevant chunks
8. **Data Extraction** — LLM extracts study variables
9. **Synthesis** — Generates meta-analysis summary

## 🔧 Setup

```bash
git clone https://github.com/your-username/automated-meta-analysis-template.git
cd automated-meta-analysis-template
pip install -r requirements.txt
