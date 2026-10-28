"""Evaluate the generator against the golden set on faithfulness and answer relevancy.

Standalone: context comes from the goldens, not the retriever, so a bad score here
is the generator's fault and nothing else's.
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate.configs import CacheConfig
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

# src/ holds the generator
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from generator import generate

# The judge model calls OpenAI, so the key must be loaded before any metric runs.
load_dotenv()

GOLDENS_PATH = PROJECT_ROOT / "goldens" / "rag_goldens.json"
JUDGE_MODEL = "gpt-4.1-mini"
THRESHOLD = 0.7


def load_goldens():
    with open(GOLDENS_PATH, encoding="utf-8") as f:
        return json.load(f)


def build_test_cases(goldens):
    """Answer each golden question from its own hand-authored context."""
    test_cases = []
    for golden in goldens:
        context = golden["context"]

        test_cases.append(
            LLMTestCase(
                input=golden["input"],
                actual_output=generate(golden["input"], context),
                expected_output=golden["expected_output"],
                retrieval_context=context,
            )
        )
    return test_cases


if __name__ == "__main__":
    goldens = load_goldens()
    print(f"goldens: {len(goldens)}")

    test_cases = build_test_cases(goldens)

    metrics = [
        FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ]

    # Disk cache needs a shared file lock deepeval can't take on Windows without
    # pywin32, which leaves its cached test run as None and crashes the writer.
    evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters={
            "context": "golden",  # generator evaluated standalone
            "generator_model": "gpt-4.1-mini",
            "judge_model": JUDGE_MODEL,
            "golden_set": GOLDENS_PATH.name,
        },
        cache_config=CacheConfig(write_cache=False, use_cache=False),
    )
