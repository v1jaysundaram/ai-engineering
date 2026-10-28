"""Build the Chroma vector store from the knowledge base PDF."""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from pathlib import Path
import re

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "knowledge_base.pdf"
PERSIST_DIR = PROJECT_ROOT / "chroma-db"
COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "text-embedding-3-small" # "text-embedding-3-large"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 0


def load_documents():
    """Load the PDF and normalise the extracted text."""
    docs = PyPDFLoader(file_path=str(PDF_PATH)).load()

    for doc in docs:
        # PDF extraction scatters tabs and newlines mid-sentence, so drop all of
        # them and rebuild the only breaks that carry meaning: the headings.
        text = re.sub(r"\s+", " ", doc.page_content)
        text = re.sub(r"\s*(Section \d+)", r"\n\1", text)        # Section 2 - Cards
        text = re.sub(r"\s*(\d+\.\d+ [A-Z])", r"\n\1", text)     # 2.1 Debit Cards
        doc.page_content = text.strip()

    return docs


def split_documents(docs):
    """Split on section / subsection headings, falling back to sentences."""
    splitter = RecursiveCharacterTextSplitter(
        separators=[r"\nSection \d+", r"\n\d+\.\d+ ", "\n", ". ", " "],
        is_separator_regex=True,
        keep_separator="start",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def get_vector_store():
    """Open the persisted Chroma collection (creates it if missing)."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=str(PERSIST_DIR),
    )


def build_vector_store():
    """Load, chunk and index the knowledge base. Re-running upserts by id."""
    docs = load_documents()
    chunks = split_documents(docs)

    vector_store = get_vector_store()
    vector_store.add_documents(
        documents=chunks,
        ids=[f"chunk-{i:03d}" for i in range(len(chunks))],
    )
    return vector_store


if __name__ == "__main__":
    vector_store = build_vector_store()
    print("indexed chunks:", vector_store._collection.count())

    results = vector_store.similarity_search_with_score(
        query="How do I report a lost card?", k=2
    )
    for doc, score in results:
        print(doc.page_content)
