from deepeval.test_case import SingleTurnParams
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

custom_metric = GEval(
    name="Completeness",
    #criteria="Determine whether the actual output is complete based on the expected output.",

    evaluation_steps=[
        "Identify every condition in 'expected output' that the user must satisfy to get what they asked for in 'input'",
        "Check whether each of those conditions appears in 'actual output'",
        "Heavily penalize any omitted condition or requirement the user would need to act correctly",
        "Do NOT penalize the omission of process instructions, refund timelines, or background detail that is not a condition the user must meet",
        "Shorter phrasing, paraphrasing, and a different order are perfectly acceptable",
    ],
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
    # rubric=[
    #     Rubric(score_range=(0, 2), expected_outcome="Missing most of what the user needs to act."),
    #     Rubric(score_range=(3, 6), expected_outcome="Covers the main point but omits a condition the user must meet."),
    #     Rubric(score_range=(7, 9), expected_outcome="Covers every condition, omits only optional background detail."),
    #     Rubric(score_range=(10, 10), expected_outcome="Fully complete; nothing the user needs is missing."),
    # ],
)

test_case = LLMTestCase(
    input="I bought a laptop 20 days ago. It's unopened in the original box. Can I return it?",
    actual_output=(
        "Yes — you're at day 20 of the 30-day window and you still have the original "
        "packaging and accessories, so you're eligible to return it."
    ),
    expected_output=(
        "Yes. You're within the 30-day return window for electronics, and since you still "
        "have the original packaging and all accessories you meet the condition for return. "
        "Refunds are issued to your original payment method within 5-7 business days, and you "
        "can start the return from the Orders page in your account."
    ),
)

custom_metric.measure(test_case)
print(custom_metric.score, custom_metric.reason)
