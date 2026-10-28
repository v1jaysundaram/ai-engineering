"""End to end RAG: reranked retrieval feeding the generator."""

from generator import generate
from reranker import FETCH_K, TOP_K, get_reranking_retriever


class RagPipeline:
    """Retrieve, then answer. One instance loads the store and reranker once."""

    def __init__(self, fetch_k=FETCH_K, k=TOP_K):
        self.retriever = get_reranking_retriever(fetch_k=fetch_k, k=k)

    def invoke(self, query):
        docs = self.retriever.invoke(query)

        # The generator wants list[str], and so do the eval metrics, so unpack once.
        context = [doc.page_content for doc in docs]

        return {
            "query": query,
            "context": context,
            "answer": generate(query, context),
        }


if __name__ == "__main__":
    rag = RagPipeline()

    result = rag.invoke("How do I report a lost card?")
    print("QUERY:", result["query"])
    print("ANSWER:", result["answer"])
    print("\nCONTEXT:")
    for i, chunk in enumerate(result["context"]):
        print(f"  [{i}] {chunk[:120]}...")
