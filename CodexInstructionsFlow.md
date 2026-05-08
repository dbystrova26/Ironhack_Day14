Step 1
Read TASK.md.

Do not write any code yet.

Give me:
1. A list of functions we will need with their signatures
2. The order to implement them
3. The three biggest risks or gotchas for this specific lab
4. What I should test after each step


Step 2
Create requirements.txt for this lab.
Use these exact packages and versions known to work with Python 3.11:

langchain==0.3.25
langchain-openai==0.2.14
langchain-community==0.3.20
openai==1.58.1
python-dotenv==1.0.1
chromadb==0.5.23
pypdf==4.3.1
sentence-transformers==3.3.1

No other packages needed.

pip install -r requirements.txt

pip install chromadb --prefer-binary


Step 3
Create relevance_scoring_rerankers.py with empty function stubs only.

Use exactly these function signatures from the plan:

def load_environment() -> None
def download_pdf(url: str, output_path: str) -> str
def download_documents() -> dict[str, str]
def load_pdf_pages(pdf_path: str, source: str, topic: str) -> list[Document]
def chunk_documents(documents: list[Document]) -> list[Document]
def build_vector_store(chunks: list[Document], persist_directory: str) -> Chroma
def get_baseline_retrieval(vector_store: Chroma, query: str, k: int = 8, source_filter: str | None = None) -> list[Document]
def score_chunk_relevance(query: str, chunk: Document) -> float
def llm_rerank_chunks(query: str, chunks: list[Document]) -> list[tuple[Document, float]]
def load_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder
def cross_encoder_rerank_chunks(query: str, chunks: list[Document], reranker: CrossEncoder) -> list[tuple[Document, float]]
def generate_answer(query: str, chunks: list[Document]) -> str
def run_rag_pipeline(query: str, vector_store: Chroma, method: str, source_filter: str | None = None) -> dict
def evaluate_queries(vector_store: Chroma, queries: list[str]) -> list[dict]
def write_lab_summary(results: list[dict], output_path: str) -> None
def main() -> None

Each function body should contain only:
- a docstring describing inputs, outputs, and what it does
- pass

Add all necessary imports at the top.
Add if __name__ == "__main__": main() at the bottom.


Import conventions for all future code in this repo:
- Document comes from langchain_core.documents, not langchain.schema
- RecursiveCharacterTextSplitter comes from langchain_text_splitters, not langchain.text_splitter
- ChatOpenAI comes from langchain_openai
- OpenAIEmbeddings comes from langchain_openai
- Chroma comes from langchain_community.vectorstores
- PyPDFLoader comes from langchain_community.document_loaders


Test conventions:
- Test each function in isolation before running main()
- Use only 3 pages and 10 chunks for embedding tests to minimise API cost
- Always activate langchain-lab conda environment before running
- Never run main() until all individual function tests pass

