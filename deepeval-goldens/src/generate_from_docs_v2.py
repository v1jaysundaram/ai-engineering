from pathlib import Path

from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import (
    FiltrationConfig,
    EvolutionConfig,
    Evolution,
    StylingConfig
)

# deepeval-goldens/ (paths resolve from this file, so scripts run from any folder)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

filtration_config = FiltrationConfig(
  critic_model="gpt-4.1",
  synthetic_input_quality_threshold=0.6
)

evolution_config = EvolutionConfig(
    evolutions={
        Evolution.CONCRETIZING: 0.4,
        Evolution.CONSTRAINED: 0.4,
        Evolution.MULTICONTEXT: 0.2,
    },
    num_evolutions=1
)

styling_config = StylingConfig(
    input_format="Short, plainly-worded questions in English, as a real customer would type them into a chat window. Ask about one specific thing at a time. Use everyday language, not banking jargon, and write words out in full rather than abbreviating.",
    expected_output_format="A short, direct answer in plain English, usually one to three sentences. Lead with the answer itself, then add only the detail the customer needs to act on it. No headings, no bullet lists, no bold text.",
    task="Answering customer questions about a fintech company's banking products and policies, using its help knowledge base",
    scenario="Everyday banking customers of varying technical and financial literacy, asking a support chatbot about accounts, cards, loans, and general policies",
)

synthesizer = Synthesizer(filtration_config=filtration_config,
                          evolution_config=evolution_config,
                          styling_config=styling_config)


goldens = synthesizer.generate_goldens_from_docs(
    document_paths=[str(PROJECT_ROOT / 'data' / 'knowledge_base.pdf')],
    include_expected_output=True
)

synthesizer.save_as(
    file_type='json',
    directory=str(PROJECT_ROOT / "goldens"),
    file_name='rag_golden_dataset-v2'
)