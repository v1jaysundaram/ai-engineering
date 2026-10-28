"""Retriever over the persisted Chroma knowledge base."""

from build_vector_store import get_vector_store

SEARCH_TYPE = "similarity"
TOP_K = 4


def get_retriever(search_type=SEARCH_TYPE, k=TOP_K):
    """Return a retriever over the existing Chroma collection."""
    vector_store = get_vector_store()
    return vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )


if __name__ == "__main__":
    retriever = get_retriever()

    docs = retriever.invoke("How do I report a lost card?")
    for doc in docs:
        print(doc.page_content)
