import os
import sys
import time

import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder


def load_environment() -> None:
    """Load environment variables and validate required configuration.

    Inputs:
        None.
    Outputs:
        None.
    What it does:
        Loads values from the local environment file and prepares API keys or
        configuration needed by the lab.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    load_dotenv()
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY is missing. Add it to your .env file.")
    print("✓ Environment loaded")


def download_pdf(url: str, output_path: str) -> str:
    """Download a PDF from a URL to a local path.

    Inputs:
        url: The remote PDF URL to download.
        output_path: The local file path where the PDF should be saved.
    Outputs:
        The local path to the downloaded PDF as a string.
    What it does:
        Retrieves a PDF from the given URL and stores it at the requested path.
    """
    if os.path.exists(output_path):
        size_kb = os.path.getsize(output_path) / 1024
        print(f"Skipping download, file already exists: {output_path} ({size_kb:.1f} KB)")
        return output_path

    response = requests.get(url, timeout=60)
    if response.status_code != 200:
        raise Exception(f"Failed to download PDF from {url}. HTTP status: {response.status_code}")

    with open(output_path, "wb") as file:
        file.write(response.content)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"Downloaded {output_path} ({size_kb:.1f} KB)")
    return output_path


def download_documents() -> dict[str, str]:
    """Download lab source PDFs and return {source_name: local_path}.

    Inputs:
        None.
    Outputs:
        A dictionary mapping source names to local document file paths.
    What it does:
        Downloads the EU AI Act and Anthropic Claude 3 Model Card PDFs.
        Skips download if the file already exists locally.
    """
    documents = {
        "eu_ai_act": (
            "eu_ai_act.pdf",
            "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689",
        ),
        "anthropic": (
            "anthropic_model_card.pdf",
            "https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf",
        ),
    }

    document_paths = {}
    for source, (filename, url) in documents.items():
        document_paths[source] = download_pdf(url, filename)

    return document_paths


def load_pdf_pages(pdf_path: str, source: str, topic: str) -> list[Document]:
    """Load PDF pages as LangChain documents with metadata.

    Inputs:
        pdf_path: The local path to a PDF file.
        source: The source label to store in document metadata.
        topic: The topic label to store in document metadata.
    Outputs:
        A list of LangChain Document objects, one per loaded PDF page.
    What it does:
        Extracts page text from a PDF and attaches source, topic, and page metadata.
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    for page in pages:
        page.metadata["source"] = source
        page.metadata["topic"] = topic

    print(f"Loaded {len(pages)} pages from {source}")
    return pages


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split loaded documents into smaller chunks.

    Inputs:
        documents: The page-level LangChain documents to split.
    Outputs:
        A list of chunk-level LangChain Document objects.
    What it does:
        Splits document text into retrievable chunks while preserving metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(documents)
    chunk_total = len(chunks)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
        chunk.metadata["chunk_total"] = chunk_total

    print(f"Created {chunk_total} chunks")
    return chunks


def build_vector_store(chunks: list[Document], persist_directory: str) -> Chroma:
    """Embed chunks and store them in Chroma.

    Inputs:
        chunks: The chunk-level documents to embed.
        persist_directory: The directory where the Chroma store should persist.
    Outputs:
        A Chroma vector store containing the embedded chunks.
    What it does:
        Creates embeddings for document chunks and saves them in a Chroma collection.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="trustworthy_ai_rag",
        persist_directory=persist_directory,
    )
    vector_count = vector_store._collection.count()
    print(f"Stored {vector_count} vectors")
    return vector_store


def get_baseline_retrieval(
    vector_store: Chroma,
    query: str,
    k: int = 8,
    source_filter: str | None = None,
) -> list[Document]:
    """Retrieve relevant chunks using baseline cosine similarity.

    Inputs:
        vector_store: The Chroma vector store to search.
        query: The user question or search query.
        k: The number of chunks to retrieve.
        source_filter: An optional source metadata value to restrict retrieval.
    Outputs:
        A list of retrieved LangChain Document objects.
    What it does:
        Runs similarity search against Chroma, optionally limited by source metadata.
    """
    if source_filter is None:
        chunks = vector_store.similarity_search(query, k=k)
        source_message = "all sources"
    else:
        retriever = vector_store.as_retriever(
            search_kwargs={"k": k, "filter": {"source": source_filter}}
        )
        chunks = retriever.invoke(query)
        source_message = source_filter

    print(f"Retrieved {len(chunks)} chunks from {source_message}")
    return chunks


def score_chunk_relevance(query: str, chunk: Document) -> float:
    """Score one chunk's relevance to a query using an LLM.

    Inputs:
        query: The user question or search query.
        chunk: The document chunk to score.
    Outputs:
        A float relevance score from 0.0 to 1.0.
    What it does:
        Asks a language model to rate how relevant a chunk is to the query.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""
Return ONLY a float from 0.0 to 1.0.

Score how relevant this chunk is for answering the query.

Query:
{query}

Chunk:
{chunk.page_content}
"""
    response = llm.invoke(prompt)

    try:
        score = float(response.content.strip())
    except ValueError:
        return 0.5

    return max(0.0, min(1.0, score))


def llm_rerank_chunks(query: str, chunks: list[Document]) -> list[tuple[Document, float]]:
    """Rerank chunks using LLM relevance scores.

    Inputs:
        query: The user question or search query.
        chunks: The retrieved chunks to score and rerank.
    Outputs:
        A list of document and score tuples sorted by relevance.
    What it does:
        Scores each chunk with an LLM and orders the chunks by score.
    """
    results = []
    for chunk in chunks:
        score = score_chunk_relevance(query, chunk)
        results.append((chunk, score))
        time.sleep(0.2)

    results.sort(key=lambda item: item[1], reverse=True)
    return results


def load_cross_encoder(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> CrossEncoder:
    """Load a local cross-encoder reranking model.

    Inputs:
        model_name: The sentence-transformers cross-encoder model identifier.
    Outputs:
        A loaded CrossEncoder model.
    What it does:
        Initializes the local reranker used to score query and chunk pairs.
    """
    reranker = CrossEncoder(model_name)
    print(f"Loaded cross-encoder model: {model_name}")
    return reranker


def cross_encoder_rerank_chunks(
    query: str,
    chunks: list[Document],
    reranker: CrossEncoder,
) -> list[tuple[Document, float]]:
    """Rerank chunks using a local cross-encoder model.

    Inputs:
        query: The user question or search query.
        chunks: The retrieved chunks to rerank.
        reranker: The loaded CrossEncoder model.
    Outputs:
        A list of document and score tuples sorted by relevance.
    What it does:
        Scores query and chunk pairs locally and orders chunks by score.
    """
    pairs = [(query, chunk.page_content) for chunk in chunks]
    scores = reranker.predict(pairs)
    results = list(zip(chunks, [float(score) for score in scores]))
    results.sort(key=lambda item: item[1], reverse=True)
    return results


def generate_answer(query: str, chunks: list[Document]) -> str:
    """Generate a RAG answer from selected context chunks.

    Inputs:
        query: The user question.
        chunks: The context chunks to use for answer generation.
    Outputs:
        A generated answer string.
    What it does:
        Uses the provided chunks as context to answer the query.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    context_parts = []

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "unknown")
        context_parts.append(
            f"Source: {source}, page: {page}\n\n{chunk.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)
    system_instruction = """Answer using ONLY the provided context.
If context is insufficient say so clearly.
Cite the source and page for key facts."""
    prompt = f"""
Context:
{context}

Question:
{query}
"""
    response = llm.invoke(
        [
            ("system", system_instruction),
            ("human", prompt),
        ]
    )
    return response.content


def run_rag_pipeline(
    query: str,
    vector_store: Chroma,
    method: str,
    source_filter: str | None = None,
) -> dict:
    """Run the retrieval and answer-generation pipeline.

    Inputs:
        query: The user question.
        vector_store: The Chroma vector store to search.
        method: The retrieval or reranking method to use.
        source_filter: An optional source metadata value to restrict retrieval.
    Outputs:
        A dictionary containing the answer, retrieved chunks, and method details.
    What it does:
        Combines baseline retrieval, optional reranking, and answer generation.
    """
    initial_chunks = get_baseline_retrieval(
        vector_store,
        query,
        k=10,
        source_filter=source_filter,
    )

    if method == "baseline":
        selected_chunks = initial_chunks[:3]
    elif method == "llm_rerank":
        reranked_chunks = llm_rerank_chunks(query, initial_chunks)
        selected_chunks = [chunk for chunk, _score in reranked_chunks[:3]]
    elif method == "cross_encoder":
        reranker = load_cross_encoder()
        reranked_chunks = cross_encoder_rerank_chunks(query, initial_chunks, reranker)
        selected_chunks = [chunk for chunk, _score in reranked_chunks[:3]]
    else:
        raise ValueError("method must be one of: baseline, llm_rerank, cross_encoder")

    answer = generate_answer(query, selected_chunks)
    sources = [
        {
            "source": chunk.metadata.get("source"),
            "topic": chunk.metadata.get("topic"),
            "page": chunk.metadata.get("page"),
            "chunk_id": chunk.metadata.get("chunk_id"),
        }
        for chunk in selected_chunks
    ]

    return {
        "query": query,
        "method": method,
        "answer": answer,
        "sources": sources,
        "chunks_used": len(selected_chunks),
    }


def evaluate_queries(vector_store: Chroma, queries: list[str]) -> list[dict]:
    """Evaluate multiple queries across retrieval methods.

    Inputs:
        vector_store: The Chroma vector store to search.
        queries: The list of evaluation questions to run.
    Outputs:
        A list of dictionaries containing comparison results.
    What it does:
        Runs the lab evaluation queries and compares answers with and without reranking.
    """
    methods = ["baseline", "llm_rerank", "cross_encoder"]
    results = []

    for query in queries:
        for method in methods:
            result = run_rag_pipeline(query, vector_store, method=method)
            results.append(result)

    return results


def write_lab_summary(results: list[dict], output_path: str) -> None:
    """Write the lab evaluation summary to a Markdown file.

    Inputs:
        results: The evaluation results to summarize.
        output_path: The Markdown file path to write.
    Outputs:
        None.
    What it does:
        Creates a readable lab summary comparing retrieval and reranking behavior.
    """
    grouped_results = {}
    for result in results:
        grouped_results.setdefault(result["query"], []).append(result)

    best_by_query = {}
    for query, query_results in grouped_results.items():
        best_result = max(
            query_results,
            key=lambda result: (
                "insufficient" not in result["answer"].lower(),
                len(result["answer"]),
            ),
        )
        best_by_query[query] = best_result["method"]

    table_rows = [
        "| Query | Best method |",
        "| --- | --- |",
    ]
    for query, method in best_by_query.items():
        table_rows.append(f"| {query} | {method} |")

    summary = f"""# Lab Summary: Relevance Scoring and Rerankers

Reranking helped most when the baseline retrieval returned broadly related chunks but did not put the most directly answerable evidence first. In this lab, that is most likely for queries where both documents discuss similar themes such as transparency, safety, risk, harmful outputs, or bias.

Use the cross-encoder reranker when you want a fast local reranking step after retrieval and want to avoid extra LLM calls. Use the LLM reranker when interpretive relevance judgment matters more than speed or cost, especially when the wording of the query needs deeper semantic judgment.

## Best Answer By Query

{chr(10).join(table_rows)}
"""

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(summary)

    print(f"Wrote lab summary to {output_path}")


def main() -> None:
    """Run the relevance scoring and reranker lab script.

    Inputs:
        None.
    Outputs:
        None.
    What it does:
        Coordinates document preparation, retrieval, reranking, evaluation, and summary output.
    """
    load_environment()
    pdf_paths = download_documents()
    topics = {
        "eu_ai_act": "ai_regulation",
        "anthropic": "ai_safety",
    }

    all_pages = []
    for source, path in pdf_paths.items():
        if not path.lower().endswith(".pdf"):
            print(f"Skipping non-PDF document for PDF loading: {path}")
            continue
        pages = load_pdf_pages(path, source=source, topic=topics[source])
        all_pages.extend(pages)

    chunks = chunk_documents(all_pages)
    vector_store = build_vector_store(chunks, persist_directory="./chroma_db")
    queries = [
        "What AI systems are prohibited under the EU AI Act?",
        "How does Anthropic test for harmful outputs?",
        "What transparency obligations apply to AI providers?",
        "What is the definition of a high-risk AI system?",
        "How should AI systems handle bias?",
    ]
    results = evaluate_queries(vector_store, queries=queries)
    write_lab_summary(results, "lab_summary.md")
    print("Lab complete.")


if __name__ == "__main__":
    main()

