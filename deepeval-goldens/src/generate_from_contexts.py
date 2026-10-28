import json
import sys
from pathlib import Path

from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import (
    FiltrationConfig,
    EvolutionConfig,
    Evolution,
    StylingConfig,
)

# deepeval-goldens/; rag-evals/src (a sibling project) holds the vector store helpers
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_EVALS_SRC = PROJECT_ROOT.parent / "rag-evals" / "src"
sys.path.insert(0, str(RAG_EVALS_SRC))

from build_vector_store import get_vector_store

CHUNKS_PER_CONTEXT = 2   # each chunk is paired with its nearest neighbour
GOLDENS_PER_CONTEXT = 2

vector_store = get_vector_store()
chunks = vector_store._collection.get(include=["documents"])["documents"]

# every chunk seeds one context, so nothing in the knowledge base goes untested
contexts = [
    [doc.page_content for doc in vector_store.similarity_search(chunk, k=CHUNKS_PER_CONTEXT)]
    for chunk in chunks
]

filtration_config = FiltrationConfig(
    critic_model="gpt-4.1",
    synthetic_input_quality_threshold=0.5,
)

evolution_config = EvolutionConfig(
    evolutions={
        Evolution.CONCRETIZING: 0.4,
        Evolution.CONSTRAINED: 0.4,
        Evolution.MULTICONTEXT: 0.2,
    },
    num_evolutions=1,
)

styling_config = StylingConfig(
    input_format="Short, plainly-worded questions in English, as a real customer would type them into a chat window. Ask about one specific thing at a time. Use everyday language, not banking jargon, and write words out in full rather than abbreviating.",
    expected_output_format="A short, direct answer in plain English, usually one to three sentences. Lead with the answer itself, then add only the detail the customer needs to act on it. No headings, no bullet lists, no bold text.",
    task="Answering customer questions about a fintech company's banking products and policies, using its help knowledge base",
    scenario="Everyday banking customers of varying technical and financial literacy, asking a support chatbot about accounts, cards, loans, and general policies",
)

synthesizer = Synthesizer(
    filtration_config=filtration_config,
    evolution_config=evolution_config,
    styling_config=styling_config,
)

goldens = synthesizer.generate_goldens_from_contexts(
    contexts=contexts,
    include_expected_output=True,
    max_goldens_per_context=GOLDENS_PER_CONTEXT,
)

print(f"goldens generated: {len(goldens)}")

OUT_PATH = PROJECT_ROOT / "goldens" / "rag_golden_dataset-contexts.json"

dataset = [
    {
        "id": f"g{i:03d}",
        "input": golden.input,
        "expected_output": golden.expected_output,
        "context": golden.context,
    }
    for i, golden in enumerate(goldens, 1)
]

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"saved {len(dataset)} goldens -> {OUT_PATH}")
