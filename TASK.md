# LAB: Relevance Scoring and Rerankers

## Goal
Build an advanced RAG system that queries two AI documents
and demonstrates the difference between retrieval with and
without reranking.

## Documents
- EU AI Act PDF:
  https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689
- Anthropic Claude 3 Model Card:
  https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf

## Steps to implement
1. Download both PDFs, chunk with metadata (source, topic, page)
2. Embed chunks with text-embedding-3-small, store in Chroma
3. Baseline retrieval — pure cosine similarity
4. LLM relevance scoring — GPT rates each chunk 0.0 to 1.0
5. Cross-encoder reranker — sentence-transformers, runs locally
6. Metadata filtering — restrict search by source
7. Complete RAG pipeline combining all steps
8. Evaluation — compare answers with and without reranking

## Tech stack
Python 3.11, langchain, langchain-openai, langchain-community,
chromadb, pypdf, sentence-transformers, openai, python-dotenv

## File to create
relevance_scoring_rerankers.py (not a notebook — plain Python script)

## Output files
- relevance_scoring_rerankers.py
- lab_summary.md
- requirements.txt
- README.md