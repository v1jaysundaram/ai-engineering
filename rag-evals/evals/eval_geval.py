"""Custom G-Eval metrics over the RAG pipeline: completeness, correctness, tonality."""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import LLMTestCase, SingleTurnParams

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_pipeline import RagPipeline

load_dotenv()

GOLDENS_PATH = PROJECT_ROOT / "goldens" / "rag_goldens.json"
JUDGE_MODEL = "gpt-4.1-mini"
THRESHOLD = 0.7

PARAMS = [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT]


completeness = GEval(
    name="Completeness",
    criteria="Determine whether the actual output covers everything in the expected output that the user needs in order to act on their question.",
    evaluation_params=PARAMS,
    rubric=[
        Rubric(score_range=(0, 2), expected_outcome="Missing most of what the user needs to act."),
        Rubric(score_range=(3, 6), expected_outcome="Covers the main point but omits a condition the user must meet."),
        Rubric(score_range=(7, 9), expected_outcome="Covers every condition, omits only optional background detail."),
        Rubric(score_range=(10, 10), expected_outcome="Fully complete; nothing the user needs is missing."),
    ],
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
)

correctness = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    evaluation_params=PARAMS,
    rubric=[
        Rubric(score_range=(0, 2), expected_outcome="Contradicts the expected output."),
        Rubric(score_range=(3, 6), expected_outcome="Partly right, or restates a general rule instead of answering."),
        Rubric(score_range=(7, 9), expected_outcome="Correct conclusion with a minor imprecision."),
        Rubric(score_range=(10, 10), expected_outcome="Fully correct."),
    ],
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
)

tonality = GEval(
    name="Tonality",
    evaluation_steps=[
        "Check whether 'actual output' is warm and kind in tone, and never blunt, cold, or dismissive",
        "Check whether 'actual output' ends by expressing the hope that it answered the user's question",
        "Heavily penalize a missing closing line of that kind, even when the rest of the answer is kind",
        "Do not judge whether the answer is factually right; tone and the closing line are all that matter here",
    ],
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
    rubric=[
        Rubric(score_range=(0, 2), expected_outcome="Cold or dismissive, and no closing line."),
        Rubric(score_range=(3, 6), expected_outcome="Neutral and polite, but no closing line expressing hope."),
        Rubric(score_range=(7, 9), expected_outcome="Kind, with a closing line that is weak or only implied."),
        Rubric(score_range=(10, 10), expected_outcome="Kind throughout and clearly ends hoping it answered the question."),
    ],
    threshold=THRESHOLD,
    model=JUDGE_MODEL,
)


def build_test_cases():
    """Run each golden question through the pipeline."""
    with open(GOLDENS_PATH, encoding="utf-8") as f:
        goldens = json.load(f)

    rag = RagPipeline()
    test_cases = []
    for golden in goldens:
        result = rag.invoke(golden["input"])
        test_cases.append(
            LLMTestCase(
                input=result["query"],
                actual_output=result["answer"],
                expected_output=golden["expected_output"],
            )
        )
    return test_cases


if __name__ == "__main__":
    test_cases = build_test_cases()
    print(f"test cases: {len(test_cases)}")

    evaluate(
        test_cases=test_cases,
        metrics=[completeness, correctness, tonality],
        # Disk cache needs a file lock deepeval can't take on Windows without pywin32.
        cache_config=CacheConfig(write_cache=False, use_cache=False),
        # Keeps the judge under the gpt-4.1-mini tokens/min limit.
        async_config=AsyncConfig(max_concurrent=4, throttle_value=1),
    )
