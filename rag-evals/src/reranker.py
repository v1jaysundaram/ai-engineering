"""Reranking retriever: over-fetch with the bi-encoder, reorder with a cross-encoder."""

from sentence_transformers import CrossEncoder

from build_vector_store import get_vector_store

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FETCH_K = 10
TOP_K = 3


class RerankingRetriever:
    """Drop-in replacement for the base retriever: same .invoke(query) -> [Document]."""

    def __init__(self, model=CROSS_ENCODER_MODEL, fetch_k=FETCH_K, k=TOP_K):
        self.vector_store = get_vector_store()
        self.cross_encoder = CrossEncoder(model)
        self.fetch_k = fetch_k  # how many the bi-encoder brings back (over-retrieve)
        self.k = k              # how many survive the rerank

    def invoke(self, query):
        # The bi-encoder embedded each chunk without ever seeing the query, so its
        # ordering is cheap but blunt. Over-fetch, then let the cross-encoder read
        # each (query, chunk) pair together and re-sort what came back.
        candidates = self.vector_store.similarity_search(query, k=self.fetch_k)
        if not candidates:
            return []

        scores = self.cross_encoder.predict([(query, doc.page_content) for doc in candidates])

        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        return [doc for doc, _ in ranked[: self.k]]


def get_reranking_retriever(model=CROSS_ENCODER_MODEL, fetch_k=FETCH_K, k=TOP_K):
    return RerankingRetriever(model=model, fetch_k=fetch_k, k=k)


if __name__ == "__main__":
    retriever = get_reranking_retriever()

    docs = retriever.invoke("How do I report a lost card?")
    for doc in docs:
        print(doc.page_content)
