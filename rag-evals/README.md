# RAG Evals

A customer-support RAG chatbot for **Sunrise Bank**, a fictional bank, plus a suite of evals that measure it from every angle: retrieval quality, answer quality, safety, and operational health.

## The pipeline

```
customer question
  -> OpenAI embedding (text-embedding-3-small)
  -> Chroma similarity search (fetch_k = 10)
  -> cross-encoder rerank (ms-marco-MiniLM-L-6-v2, keep top_k = 3)
  -> gpt-4.1-mini answer
```

| File | Role |
|---|---|
| `src/build_vector_store.py` | Loads `data/knowledge_base.pdf`, chunks it on section headings, indexes it into Chroma |
| `src/retriever.py` | Plain similarity retriever |
| `src/reranker.py` | Over-fetch + cross-encoder rerank retriever |
| `src/generator.py` | Prompt + gpt-4.1-mini answer generation |
| `src/rag_pipeline.py` | `RagPipeline`: reranked retrieval feeding the generator |

## The evals

| File | Question it answers | Metrics |
|---|---|---|
| `evals/eval_retriever.py` | Does retrieval find the right chunks? | Contextual recall, precision |
| `evals/eval_generator.py` | Given perfect context, is the answer good? | Faithfulness, answer relevancy |
| `evals/eval_rag.py` | Is the whole pipeline good end to end? | Faithfulness, answer relevancy, contextual precision / recall / relevancy |
| `evals/eval_geval.py` | Custom criteria with an LLM judge | G-Eval: completeness, correctness, tonality |
| `evals/eval_safety.py` | Is it safe? | Scope adherence, system prompt / corpus / PII leakage, toxicity |
| `evals/eval_ops.py` | Can it run in production? | Latency (p50/p95/p99, TTFT), cost per query, reliability |
| `evals/test_*.py` | Single-metric versions of the above, one file per metric | |

Golden datasets live in `goldens/`. The operational evals (`eval_ops.py`) need no goldens and no LLM judge.

## Run

From this folder, with the repo-root `.env` filled in:

```bash
uv pip install -r requirements.txt

python src/build_vector_store.py   # once: builds chroma-db/ (gitignored)
python src/rag_pipeline.py         # smoke test: one question end to end

python evals/eval_rag.py
python evals/eval_safety.py
python evals/eval_ops.py
```

Running the evals calls the OpenAI API, so it costs money: the LLM judge for quality and safety evals, and around 80 generation calls for `eval_ops.py`.
