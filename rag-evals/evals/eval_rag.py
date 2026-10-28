"""Evaluate the full RAG pipeline: reranked retrieval plus generation, scored end to end."""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

# src/ holds the pipeline
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_pipeline import RagPipeline

# The judge model calls OpenAI, so the key must be loaded before any metric runs.
load_dotenv()

GOLDENS_PATH = PROJECT_ROOT / "goldens" / "rag_goldens.json"
JUDGE_MODEL = "gpt-4.1-mini"
THRESHOLD = 0.7
FETCH_K = 5
TOP_K = 3


def load_goldens():
    with open(GOLDENS_PATH, encoding="utf-8") as f:
        return json.load(f)


def build_test_cases(goldens, fetch_k=FETCH_K, k=TOP_K):
    """Run each golden question through the whole pipeline."""
    rag = RagPipeline(fetch_k=fetch_k, k=k)

    test_cases = []
    for golden in goldens:
        result = rag.invoke(golden["input"])

        test_cases.append(
            LLMTestCase(
                input=result["query"],
                actual_output=result["answer"],
                expected_output=golden["expected_output"],
                retrieval_context=result["context"],
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
        ContextualRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),

        FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ]

    # Disk cache needs a shared file lock deepeval can't take on Windows without
    # pywin32, which leaves its cached test run as None and crashes the writer.
    evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "pipeline": "reranked_fetch10_k3",
            "generator_model": "gpt-4.1-mini",
            "embedding_model": "text-embedding-3-small",
            "cross_encoder": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "chunk_size": 600,
            "chunk_overlap": 0,
            "fetch_k": FETCH_K,
            "top_k": TOP_K,
            "judge_model": JUDGE_MODEL,
            "golden_set": GOLDENS_PATH.name,
        },
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        # 5 metrics x 26 cases at the default concurrency of 20 blows past the
        # 200k tokens/min limit on gpt-4.1-mini. Fewer in flight, briefly spaced.
        async_config=AsyncConfig(max_concurrent=4, throttle_value=1),
    )
