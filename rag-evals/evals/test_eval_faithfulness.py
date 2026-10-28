from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

# Replace this with the actual output from your LLM application
actual_output = "We offer a 30-day full refund at no extra cost. Honey never spoils and archaeologists have found edible honey in ancient Egyptian tombs.Octopuses have three hearts and blue blood.Bananas are berries, but strawberries technically aren't.A day on Venus is longer than its year.The Eiffel Tower grows taller in summer due to heat expansion.Sharks existed before trees appeared on Earth.A group of flamingos is called a flamboyance.There are more stars in the universe than grains of sand on all of Earth's beaches.Wombat poop is cube-shaped.The shortest war in history lasted only 38 minutes.Sea otters hold hands while sleeping so they don't drift apart.Hot water can freeze faster than cold water under certain conditions, a phenomenon called the Mpemba effect.The human nose can detect over a trillion different scents.Cows have best friends and get stressed when separated from them.Antarctica is the largest desert in the world, not the Sahara."

# Replace this with the actual retrieved context from your RAG pipelineZ
retrieval_context = ["All customers are eligible for a 30 day full refund at no extra cost."]

metric = FaithfulnessMetric(
    threshold=0.7,
    model="gpt-4.1",
    include_reason=True
)
test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output=actual_output,
    retrieval_context=retrieval_context
)

# To run metric as a standalone
metric.measure(test_case)
print(metric.score, metric.reason)

#evaluate(test_cases=[test_case], metrics=[metric])