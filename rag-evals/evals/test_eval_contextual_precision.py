from dotenv import load_dotenv

from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

load_dotenv()

# Replace this with the actual output from your LLM application
actual_output = "We offer a 30-day full refund at no extra cost."

# Replace this with the expected output from your RAG generator
expected_output = "You are eligible for a 30 day full refund at no extra cost. If you don't have the tag, then refund is not possible"

# Replace this with the actual retrieved context from your RAG pipeline
retrieval_context = ["All customers are eligible for a 30 day full refund at no extra cost.",
                     "Refund is not possible without a tag",
                     "My youtube channel is Vijai Sundaram",
]

metric = ContextualPrecisionMetric(
    threshold=0.7,
    model="gpt-4.1-mini",
    include_reason=True
)
test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output=actual_output,
    expected_output=expected_output,
    retrieval_context=retrieval_context
)

# To run metric as a standalone
metric.measure(test_case)
print(metric.score, metric.reason)

#evaluate(test_cases=[test_case], metrics=[metric])