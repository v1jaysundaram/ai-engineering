"""Evaluate the retriever against the golden set on contextual recall and precision."""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate.configs import CacheConfig
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase

# src/ holds the retriever
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retriever import get_retriever
from reranker import get_reranking_retriever

# The judge model calls OpenAI, so the key must be loaded before any metric runs.
load_dotenv()

GOLDENS_PATH = PROJECT_ROOT / "goldens" / "rag_goldens.json"
JUDGE_MODEL = "gpt-4.1-mini"
THRESHOLD = 0.7
TOP_K = 3


def load_goldens():
    with open(GOLDENS_PATH, encoding="utf-8") as f:
        return json.load(f)


def build_test_cases(goldens, k=TOP_K):
    """Run each golden question through the live retriever."""
    #retriever = get_retriever(k=k)
    retriever = get_reranking_retriever(fetch_k=10, k=k)

    test_cases = []
    for golden in goldens:
        retrieved = retriever.invoke(golden["input"])
        test_cases.append(
            LLMTestCase(
                input=golden["input"],
                actual_output="( not evaluated in this run)",
                expected_output=golden["expected_output"],
                retrieval_context=[doc.page_content for doc in retrieved],
            )
        )
    return test_cases


if __name__ == "__main__":
    goldens = load_goldens()
    print(f"goldens: {len(goldens)}")

    test_cases = build_test_cases(goldens)

    metrics = [
        ContextualRecallMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ]

    # Disk cache needs a shared file lock deepeval can't take on Windows without
    # pywin32, which leaves its cached test run as None and crashes the writer.
    evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "retriever": "base_k3",  # "reranked_k3" when you swap it in
            "embedding_model": "text-embedding-3-small",
            "chunk_size": 600,
            "chunk_overlap": 0,
            "top_k": TOP_K,
            "judge_model": JUDGE_MODEL,
            "golden_set": GOLDENS_PATH.name,
        },
        cache_config=CacheConfig(write_cache=False, use_cache=False),
    )
