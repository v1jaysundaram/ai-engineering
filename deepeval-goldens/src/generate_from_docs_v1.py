from pathlib import Path

from deepeval.synthesizer import Synthesizer

# deepeval-goldens/ (paths resolve from this file, so scripts run from any folder)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

synthesizer = Synthesizer(
    model="gpt-5.4",
    cost_tracking=True,
)

goldens = synthesizer.generate_goldens_from_docs(
    document_paths=[str(PROJECT_ROOT / 'data' / 'knowledge_base.pdf')],
    include_expected_output=True
)

synthesizer.save_as(
    file_type='json',
    directory=str(PROJECT_ROOT / "goldens"),
    file_name='rag_golden_dataset-v1'
)

print(goldens[0])
print(f"\nCost: ${synthesizer.synthesis_cost:.6f}")