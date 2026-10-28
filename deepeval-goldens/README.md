# DeepEval Goldens

Generate synthetic golden datasets (question + expected answer + context) for evaluating the [`rag-evals`](../rag-evals/) pipeline, using DeepEval's `Synthesizer`.

| Script | Approach |
|---|---|
| `src/generate_from_docs_v1.py` | Straight from the knowledge base PDF, default settings |
| `src/generate_from_docs_v2.py` | From the PDF with filtration, evolutions and a customer-chat styling config |
| `src/generate_from_contexts.py` | From the chunks already in the `rag-evals` Chroma store, so every golden maps to real retrievable context |

Outputs are written to `goldens/`.

## Run

```bash
uv pip install -r requirements.txt
uv pip install -r ../rag-evals/requirements.txt   # needed by generate_from_contexts.py

python src/generate_from_docs_v2.py
python src/generate_from_contexts.py   # build the rag-evals vector store first
```
