# AI Engineering

Hands-on AI engineering projects: code from my video series. Each folder is a self-contained project with its own README and `requirements.txt`.

| Project | What it covers |
|---|---|
| [`rag-evals/`](rag-evals/) | A customer-support RAG pipeline (Chroma + cross-encoder reranker + GPT-4.1-mini) and a full eval suite: retriever, generator, end-to-end, G-Eval, safety, and operational (latency, cost, reliability) |
| [`deepeval-goldens/`](deepeval-goldens/) | Generating synthetic golden datasets with DeepEval's Synthesizer, from documents and from contexts |
| [`jev/`](jev/) | Experiments with the TypeSafe SDK: structured classification and a speed comparison against a plain LLM call |

More projects will be added over time.

## Setup

```bash
git clone https://github.com/v1jaysundaram/ai-engineering.git
cd ai-engineering

uv venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

cp .env.example .env             # then add your API keys
uv pip install -r rag-evals/requirements.txt   # or whichever project you want to run
```

All projects read their API keys from the single `.env` at the repo root.
