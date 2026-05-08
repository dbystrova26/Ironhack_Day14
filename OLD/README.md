# LAB: Relevance Scoring and Rerankers for Trustworthy AI & EU AI Act

Advanced RAG system that queries the EU AI Act and Anthropic Claude 3
Model Card using three retrieval methods — baseline, LLM reranking,
and cross-encoder reranking — to demonstrate improved retrieval quality.

## How to run

```bash
# 1. Install dependencies
pip install langchain langchain-anthropic langchain-community
pip install langchain-huggingface langchain-text-splitters
pip install chromadb pypdf sentence-transformers requests python-dotenv

# 2. Add your API key to .env
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" > .env

# 3. Run the script
python relevance_scoring_rerankers.py
```

## File map

| File | Description |
|------|-------------|
| `relevance_scoring_rerankers.py` | Main script — all pipeline logic |
| `lab_summary.md` | One-paragraph narrative (generated on run) |
| `README.md` | This file |
| `eu_ai_act.pdf` | Downloaded on first run, EU AI Act Official Journal 2024 |
| `anthropic_model_card.pdf` | Downloaded on first run, Claude 3 Model Card |
| `chroma_db/` | Persisted vector store — not re-embedded on second run |

Do not commit: `.env`, `eu_ai_act.pdf`, `anthropic_model_card.pdf`, `chroma_db/`

## What the script does

1. Downloads both PDFs automatically
2. Chunks them with metadata (source, topic, page)
3. Embeds chunks using a free local HuggingFace model
4. Runs 5 evaluation queries × 3 methods = 15 pipeline runs
5. Writes lab_summary.md with comparison results