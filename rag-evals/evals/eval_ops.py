"""Operational evals, merged: LATENCY, COST, RELIABILITY for the Sunrise Bank support RAG.

The quality evals (eval_rag, eval_geval) and safety evals (eval_safety) ask "are the
answers good and safe?". These three ask the OPERATIONAL question: can the Sunrise Bank
support bot answer customers quickly, cheaply and without falling over?

None of them needs goldens or an LLM judge. They are plain timing, token and error
measurements taken on the real pipeline:

    query -> Chroma similarity search (FETCH_K) -> cross-encoder rerank (TOP_K)
          -> gpt-4.1-mini generation

  LATENCY     -- latency is noisy, so we collect many samples and report percentiles
                 (p95 / p99 tail, not the mean) against an SLO. We measure end to end
                 AND time-to-first-token (what a streaming chat widget feels like), split
                 retrieval vs generation, throw away warmup runs (cold start: Chroma open,
                 cross-encoder first pass) and stay single-user (load testing is separate).

  COST        -- derived, not measured: cost = tokens x price. Temperature is 0, so token
                 counts barely move between runs, which makes this an honest OFFLINE answer
                 to "what does each customer question cost us?". The only paid calls are the
                 OpenAI query embedding and the gpt-4.1-mini generation; the cross-encoder
                 rerank runs locally and is free.

  RELIABILITY -- can we serve requests without failing? Success / error / retry rates with
                 an exponential-backoff retry wrapper, plus "soft failures": the call did not
                 crash but the answer is unusable (empty answer, no chunks retrieved, or the
                 required "Hope this answers your question" sign-off is missing).

Each eval has a run_*() that PRINTS its report and RETURNS a flat dict of metrics.
run_ops() runs all three on one pipeline and returns one operational snapshot, the
slice that regression testing diffs against a baseline.

Run from the rag-evals folder:  python evals/eval_ops.py
"""

import math
import sys
import time
from pathlib import Path

import tiktoken
from dotenv import load_dotenv

# src/ holds the pipeline
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from build_vector_store import EMBEDDING_MODEL
from generator import GENERATOR_MODEL, chain as generator_chain, llm, prompt
from rag_pipeline import RagPipeline
from reranker import FETCH_K, TOP_K

# OpenAI key for the embedding + generation calls.
load_dotenv()

# COST reads token usage off the AIMessage, so it composes prompt | llm directly and stops
# BEFORE the StrOutputParser that generator.chain uses (the parser throws usage away and
# returns a bare string). Same prompt, same model, same real context.
measured_chain = prompt | llm


# ---------------------------------------------------------------------------
# Shared config: the customer questions every eval replays
# ---------------------------------------------------------------------------

# Real Sunrise Bank support questions, segmented by how much work they cause. Long,
# multi-part questions pull in more context and produce longer answers, so they cost
# more and take longer. Keeping the mix explicit stops the numbers being flattered by
# easy questions only. Adjust the mix to match the traffic you actually expect.
QUESTIONS = {
    # one fact, one chunk, short answer
    "simple": [
        "Is there a monthly account fee for Sunrise Bank's Everyday Account?",
        "What minimum balance is required to keep Sunrise Bank's Plus Account open?",
    ],
    # one topic, a short procedure or two linked facts
    "medium": [
        "My card is lost. How do I freeze it in the app and get a free replacement?",
        "How can I make a complaint to Sunrise Bank, and how long does it take to get a response?",
    ],
    # many facts across sections, long answer
    "complex": [
        "What fees apply to Sunrise Bank's Everyday and Plus accounts, including monthly fees, "
        "international card fees, overdraft fees, and replacement card fees?",
        "What are your phone and live chat hours, including Saturday limits, Sunday and United "
        "Kingdom bank holiday closures, and any twenty four seven app support?",
    ],
}

# Flat list of (segment, question) pairs so every loop below can report per segment.
QUESTION_SET = [(seg, q) for seg, qs in QUESTIONS.items() for q in qs]

# The generator prompt orders every answer to end with this sign-off; reliability uses it
# as a cheap "did the generator follow its contract" check.
REQUIRED_SIGNOFF = "Hope this answers your question"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def percentile(values, p):
    """Linear-interpolation percentile (same result as numpy.percentile), written out
    so the tail maths is visible."""
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def col_avg(rows, key):
    """Average of one column across a list of dict rows."""
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def retrieve_context(rag, question):
    """Run the pipeline's reranking retriever and unpack to list[str], exactly as
    RagPipeline.invoke does, so the evals measure the same work production does."""
    docs = rag.retriever.invoke(question)
    return [doc.page_content for doc in docs]


# ===========================================================================
# LATENCY
# ===========================================================================

# ---------------------------------------------------------------------------
# Latency: config
# ---------------------------------------------------------------------------

LAT_REPEATS = 5           # measured runs per question -> samples = len(QUESTION_SET) * LAT_REPEATS
LAT_WARMUP_RUNS = 2       # throwaway calls first: Chroma open + cross-encoder load skew run 1
LAT_MEASURE_TTFT = True   # stream generation and clock time-to-first-token
LAT_STAGE_LEVEL = True    # split retrieval vs generation (always on when TTFT is on)

# SLOs for a customer support chat widget.
SLO_P95_MS = 4000         # full answer: 95% of customers have the whole answer within 4s
SLO_TTFT_P95_MS = 1500    # perceived: 95% see the first words within 1.5s


# ---------------------------------------------------------------------------
# Latency: pipeline adapters (the ONE place to edit if the pipeline API changes)
# ---------------------------------------------------------------------------

def lat_end_to_end(rag, question):
    """Plain RagPipeline.invoke: one number for the whole request, no breakdown."""
    return rag.invoke(question)["answer"]


def lat_stages(rag, question):
    """Non-streaming, timed per stage. 'retrieval' bundles the OpenAI query embedding,
    the Chroma search for FETCH_K candidates and the local cross-encoder rerank to TOP_K."""
    t0 = time.perf_counter()
    context = retrieve_context(rag, question)
    t1 = time.perf_counter()
    answer = generator_chain.invoke({"question": question, "context": "\n\n".join(context)})
    t2 = time.perf_counter()
    return answer, {"retrieval": (t1 - t0) * 1000, "generation": (t2 - t1) * 1000}


def lat_stages_streaming(rag, question):
    """Same retrieval, but STREAM the gpt-4.1-mini answer and clock the first token.

      ttft       = question submitted (t0) -> first visible token (what the customer feels)
      generation = generation start (t1)   -> last token

    TTFT includes retrieval on purpose: the customer waits through embedding + search +
    rerank before a single word can appear."""
    t0 = time.perf_counter()
    context = retrieve_context(rag, question)
    t1 = time.perf_counter()

    first_token_t = None
    pieces = []
    for piece in generator_chain.stream({"question": question, "context": "\n\n".join(context)}):
        if first_token_t is None and piece:
            first_token_t = time.perf_counter()   # first non-empty chunk
        pieces.append(piece)
    t2 = time.perf_counter()

    answer = "".join(pieces)
    ttft_ms = (first_token_t - t0) * 1000 if first_token_t else float("nan")
    return answer, {
        "retrieval": (t1 - t0) * 1000,
        "generation": (t2 - t1) * 1000,
        "ttft": ttft_ms,
    }


# ---------------------------------------------------------------------------
# Latency: benchmark
# ---------------------------------------------------------------------------

def lat_benchmark(rag):
    """Warm up, then time every question LAT_REPEATS times, keeping each sample
    tagged with its segment so simple vs complex can be compared."""
    # Warmup: run and DISCARD, so cold start does not pollute the stats.
    print(f"[latency] warming up ({LAT_WARMUP_RUNS} runs, discarded)...")
    for i in range(LAT_WARMUP_RUNS):
        lat_end_to_end(rag, QUESTION_SET[i % len(QUESTION_SET)][1])

    results = {"total": [], "retrieval": [], "generation": [], "ttft": [],
               "answer_len": [], "segment": []}

    print(f"[latency] measuring {len(QUESTION_SET)} questions x {LAT_REPEATS} repeats...")
    for segment, question in QUESTION_SET:
        for _ in range(LAT_REPEATS):
            start = time.perf_counter()
            if LAT_MEASURE_TTFT:
                answer, stage = lat_stages_streaming(rag, question)
                results["ttft"].append(stage["ttft"])
            elif LAT_STAGE_LEVEL:
                answer, stage = lat_stages(rag, question)
            else:
                answer, stage = lat_end_to_end(rag, question), None
            elapsed_ms = (time.perf_counter() - start) * 1000

            if stage:
                results["retrieval"].append(stage["retrieval"])
                results["generation"].append(stage["generation"])
            results["total"].append(elapsed_ms)
            results["answer_len"].append(len(answer or ""))
            results["segment"].append(segment)

    return results


# ---------------------------------------------------------------------------
# Latency: summarise + report
# ---------------------------------------------------------------------------

def lat_summarize(samples):
    """Distribution summary of one list of millisecond samples."""
    clean = [s for s in samples if not math.isnan(s)]
    if not clean:
        return {k: float("nan") for k in ("mean", "p50", "p95", "p99", "min", "max")} | {"n": 0}
    return {
        "n": len(clean),
        "mean": sum(clean) / len(clean),
        "p50": percentile(clean, 50),
        "p95": percentile(clean, 95),
        "p99": percentile(clean, 99),
        "min": min(clean),
        "max": max(clean),
    }


def lat_print_row(label, s):
    print(f"{label:<16} | n={s['n']:<3} "
          f"mean={s['mean']:7.1f}  p50={s['p50']:7.1f}  "
          f"p95={s['p95']:7.1f}  p99={s['p99']:7.1f}  "
          f"min={s['min']:7.1f}  max={s['max']:7.1f}")


def lat_slo_line(label, p95, budget):
    verdict = "PASS" if p95 <= budget else "FAIL"
    print(f"SLO: {label:<24} p95 <= {budget:>5} ms  ->  p95 = {p95:7.0f} ms   [{verdict}]")


def lat_report(results):
    print("\n" + "=" * 90)
    print(f"LATENCY (milliseconds)   generator={GENERATOR_MODEL}  fetch_k={FETCH_K}  top_k={TOP_K}")
    print("=" * 90)

    # Whole-pipeline view: end to end, perceived (TTFT) and the two stages.
    total = lat_summarize(results["total"])
    lat_print_row("end-to-end", total)
    if results["ttft"]:
        lat_print_row("ttft", lat_summarize(results["ttft"]))
    if results["retrieval"]:
        lat_print_row("retrieval", lat_summarize(results["retrieval"]))
        lat_print_row("generation", lat_summarize(results["generation"]))

    # Per-segment view: complex questions should be visibly slower; if they are not,
    # the segments are not doing their job.
    print("-" * 90)
    for segment in QUESTIONS:
        seg_total = [t for t, s in zip(results["total"], results["segment"]) if s == segment]
        lat_print_row(f"e2e [{segment}]", lat_summarize(seg_total))

    # Output length drives generation time, so note it when comparing configs.
    avg_len = sum(results["answer_len"]) / len(results["answer_len"])
    print("-" * 90)
    print(f"avg answer length: {avg_len:.0f} chars "
          f"(latency scales with output length -- keep in mind when comparing configs)")

    print("=" * 90)
    lat_slo_line("full answer", total["p95"], SLO_P95_MS)
    if results["ttft"]:
        lat_slo_line("first token (perceived)", lat_summarize(results["ttft"])["p95"], SLO_TTFT_P95_MS)
    print("=" * 90)


# ---------------------------------------------------------------------------
# Latency: run
# ---------------------------------------------------------------------------

def run_latency(rag, verbose=True):
    """Measure latency; print the report and return a flat metrics dict."""
    results = lat_benchmark(rag)
    if verbose:
        lat_report(results)

    total = lat_summarize(results["total"])
    metrics = {
        "e2e_mean_ms": total["mean"],
        "e2e_p50_ms": total["p50"],
        "e2e_p95_ms": total["p95"],
        "e2e_p99_ms": total["p99"],
        "avg_answer_len": sum(results["answer_len"]) / len(results["answer_len"]),
        "slo_e2e_pass": total["p95"] <= SLO_P95_MS,
    }
    if results["ttft"]:
        ttft = lat_summarize(results["ttft"])
        metrics["ttft_p95_ms"] = ttft["p95"]
        metrics["slo_ttft_pass"] = ttft["p95"] <= SLO_TTFT_P95_MS
    if results["retrieval"]:
        metrics["retrieval_p95_ms"] = lat_summarize(results["retrieval"])["p95"]
        metrics["generation_p95_ms"] = lat_summarize(results["generation"])["p95"]
    for segment in QUESTIONS:
        seg_total = [t for t, s in zip(results["total"], results["segment"]) if s == segment]
        metrics[f"e2e_p95_ms_{segment}"] = lat_summarize(seg_total)["p95"]
    return metrics


# ===========================================================================
# COST
# ===========================================================================

# ---------------------------------------------------------------------------
# Cost: config
# ---------------------------------------------------------------------------

COST_REPEATS = 3          # token counts are stable at temperature 0, so fewer repeats than latency

# Pricing in USD per 1M tokens. Prices change -- keep them here as constants, never buried
# in the code, and re-check https://openai.com/api/pricing before trusting a budget.
# gpt-4.1-mini (the generator):
PRICE_INPUT_PER_1M = 0.40          # cache-miss input
PRICE_CACHED_INPUT_PER_1M = 0.10   # cached (repeated prefix) input
PRICE_OUTPUT_PER_1M = 1.60         # output (4x input -> long answers dominate)
# text-embedding-3-small (embeds each customer query before the Chroma search):
PRICE_EMBEDDING_PER_1M = 0.02
# The cross-encoder rerank runs locally (sentence-transformers), so it costs $0 in API fees.

# Business projection knobs (set these to Sunrise Bank's reality).
QUERIES_PER_DAY = 2000             # expected support-bot questions per day
USD_TO_GBP = 0.75                  # Sunrise Bank reports in GBP; set to the current rate

# Budget (the cost "SLO"): offline pass/fail line per customer question.
COST_BUDGET_PER_QUERY_USD = 0.002

# text-embedding-3-small uses the cl100k_base tokenizer; counting locally saves an API call.
_embedding_encoder = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Cost: token measurement + cost maths
# ---------------------------------------------------------------------------

def cost_measure_tokens(rag, question):
    """Retrieve REAL context (so input tokens reflect the TOP_K reranked chunks the
    generator actually receives), run one generation and read token usage off the
    AIMessage."""
    context = retrieve_context(rag, question)
    msg = measured_chain.invoke({"question": question, "context": "\n\n".join(context)})
    usage = msg.usage_metadata or {}

    # Cached prefix tokens, if OpenAI reports any. OpenAI only caches prompts of 1024+
    # tokens, and our prompt starts short and puts the varying context early, so expect
    # this to read 0 here. It matters if the prompt grows a long fixed system section.
    details = usage.get("input_token_details") or {}

    return {
        "embed": len(_embedding_encoder.encode(question)),
        "input": usage.get("input_tokens", 0),
        "output": usage.get("output_tokens", 0),
        "cached": details.get("cache_read", 0) or 0,
    }


def cost_usd(tok):
    """cost = query embedding + uncached input @ full rate + cached input @ cached rate
    + output @ output rate."""
    uncached_input = max(tok["input"] - tok["cached"], 0)
    c_embed = tok["embed"] / 1_000_000 * PRICE_EMBEDDING_PER_1M
    c_in = uncached_input / 1_000_000 * PRICE_INPUT_PER_1M
    c_cached = tok["cached"] / 1_000_000 * PRICE_CACHED_INPUT_PER_1M
    c_out = tok["output"] / 1_000_000 * PRICE_OUTPUT_PER_1M
    return {"embed": c_embed, "input": c_in, "cached": c_cached, "output": c_out,
            "total": c_embed + c_in + c_cached + c_out}


# ---------------------------------------------------------------------------
# Cost: benchmark + report
# ---------------------------------------------------------------------------

def cost_benchmark(rag):
    """One row per (question, repeat): token counts plus the dollar breakdown."""
    rows = []
    print(f"[cost] measuring token usage ({len(QUESTION_SET)} questions x {COST_REPEATS} repeats)...")
    for segment, question in QUESTION_SET:
        for _ in range(COST_REPEATS):
            tok = cost_measure_tokens(rag, question)
            cost = cost_usd(tok)
            rows.append({"segment": segment, **tok, **{f"cost_{k}": v for k, v in cost.items()}})
    return rows


def cost_report(rows):
    n = len(rows)
    avg_in = col_avg(rows, "input")
    avg_out = col_avg(rows, "output")
    avg_cached = col_avg(rows, "cached")
    avg_cost = col_avg(rows, "cost_total")
    min_cost = min(r["cost_total"] for r in rows)
    max_cost = max(r["cost_total"] for r in rows)

    # Where the money goes: generator input vs generator output vs embedding.
    avg_cost_in = col_avg(rows, "cost_input") + col_avg(rows, "cost_cached")
    avg_cost_out = col_avg(rows, "cost_output")
    avg_cost_embed = col_avg(rows, "cost_embed")
    pct = lambda part: 100 * part / avg_cost if avg_cost else 0

    print("\n" + "=" * 78)
    print(f"COST  ({GENERATOR_MODEL} @ ${PRICE_INPUT_PER_1M}/${PRICE_OUTPUT_PER_1M} per 1M in/out, "
          f"{EMBEDDING_MODEL} @ ${PRICE_EMBEDDING_PER_1M})")
    print("=" * 78)
    print(f"samples                : {n}")
    print(f"avg input tokens       : {avg_in:8.0f}   ({avg_cached:.0f} cached)   "
          f"<- prompt + {TOP_K} reranked chunks + question")
    print(f"avg output tokens      : {avg_out:8.0f}")
    print("-" * 78)
    print(f"avg cost / query       : ${avg_cost:.6f}   (GBP {avg_cost * USD_TO_GBP:.6f})")
    print(f"   min / max           : ${min_cost:.6f} / ${max_cost:.6f}   "
          f"<- tight range = cost is stable, unlike latency")
    print(f"   split               : {pct(avg_cost_in):.0f}% gen input / {pct(avg_cost_out):.0f}% gen output"
          f" / {pct(avg_cost_embed):.1f}% embedding")

    # Per-segment cost: complex multi-part questions should cost the most.
    print("-" * 78)
    for segment in QUESTIONS:
        seg_rows = [r for r in rows if r["segment"] == segment]
        print(f"   [{segment:<7}]           : ${col_avg(seg_rows, 'cost_total'):.6f}   "
              f"(in {col_avg(seg_rows, 'input'):.0f} / out {col_avg(seg_rows, 'output'):.0f} tokens)")

    # Unit economics: what turning the bot on for QUERIES_PER_DAY customers costs.
    daily = avg_cost * QUERIES_PER_DAY
    monthly = daily * 30
    print("-" * 78)
    print(f"projection @ {QUERIES_PER_DAY}/day :")
    print(f"   per day             : ${daily:8.2f}   (GBP {daily * USD_TO_GBP:8.2f})")
    print(f"   per month           : ${monthly:8.2f}   (GBP {monthly * USD_TO_GBP:8.2f})")
    print("=" * 78)

    verdict = "PASS" if avg_cost <= COST_BUDGET_PER_QUERY_USD else "FAIL"
    print(f"BUDGET: cost/query <= ${COST_BUDGET_PER_QUERY_USD:.6f}  ->  ${avg_cost:.6f}   [{verdict}]")
    print("=" * 78)
    print(f"note: raising TOP_K (now {TOP_K}) adds a chunk of input tokens to EVERY query --")
    print("      check this report whenever retrieval settings change.")


# ---------------------------------------------------------------------------
# Cost: run
# ---------------------------------------------------------------------------

def run_cost(rag, verbose=True):
    """Measure cost; print the report and return a flat metrics dict."""
    rows = cost_benchmark(rag)
    if verbose:
        cost_report(rows)

    avg_cost = col_avg(rows, "cost_total")
    avg_cost_out = col_avg(rows, "cost_output")
    metrics = {
        "cost_per_query_usd": avg_cost,
        "cost_per_query_gbp": avg_cost * USD_TO_GBP,
        "avg_input_tokens": col_avg(rows, "input"),
        "avg_output_tokens": col_avg(rows, "output"),
        "avg_cached_tokens": col_avg(rows, "cached"),
        "output_cost_share_pct": 100 * avg_cost_out / avg_cost if avg_cost else 0.0,
        "monthly_usd": avg_cost * QUERIES_PER_DAY * 30,
        "budget_pass": avg_cost <= COST_BUDGET_PER_QUERY_USD,
    }
    for segment in QUESTIONS:
        seg_rows = [r for r in rows if r["segment"] == segment]
        metrics[f"cost_per_query_usd_{segment}"] = col_avg(seg_rows, "cost_total")
    return metrics


# ===========================================================================
# RELIABILITY
# ===========================================================================

# ---------------------------------------------------------------------------
# Reliability: config
# ---------------------------------------------------------------------------

REL_REPEATS = 5           # requests per question
MAX_RETRIES = 2           # retries after the first attempt (so up to 3 attempts per request)
BACKOFF_BASE_S = 0.5      # sleeps 0.5s, then 1s, ... between attempts

# Thresholds for the pass/fail verdict.
SLO_SUCCESS_RATE_PCT = 99.0      # hard: requests that returned an answer at all
SLO_SOFT_FAILURE_RATE_PCT = 5.0  # soft: answers that came back but are unusable


# ---------------------------------------------------------------------------
# Reliability: tracker + retry wrapper
# ---------------------------------------------------------------------------

class Reliability:
    """Counters for one reliability run."""

    def __init__(self):
        self.calls = 0                 # customer requests sent
        self.successes = 0             # requests that eventually returned a result
        self.failures = 0              # requests that failed even after all retries
        self.retry_attempts = 0        # total extra attempts made (one request can add several)
        self.retried_requests = 0      # requests that needed at least one retry
        self.soft_failures = {"empty_answer": 0, "no_context": 0, "missing_signoff": 0}
        self.errors = []               # exception messages, for the report


def call_with_retries(fn, rel):
    """Call fn(); on an exception wait with exponential backoff and try again, up to
    MAX_RETRIES times. Returns the result, or None if every attempt failed."""
    rel.calls += 1
    for attempt in range(MAX_RETRIES + 1):
        try:
            result = fn()
            rel.successes += 1
            if attempt > 0:
                rel.retried_requests += 1
            return result
        except Exception as e:
            if attempt < MAX_RETRIES:
                rel.retry_attempts += 1
                time.sleep(BACKOFF_BASE_S * (2 ** attempt))   # exponential backoff
            else:
                rel.failures += 1
                rel.errors.append(f"{type(e).__name__}: {e}")
                print(f"[reliability] FAILED after {MAX_RETRIES} retries: {e}")
                return None


def check_soft_failures(result, rel):
    """The request did not crash, but would the customer get a usable answer?"""
    answer = (result.get("answer") or "").strip()
    if not answer:
        rel.soft_failures["empty_answer"] += 1
    if not result.get("context"):
        # Chroma/rerank returned nothing, so the answer is not grounded in the knowledge base.
        rel.soft_failures["no_context"] += 1
    if answer and REQUIRED_SIGNOFF.lower() not in answer.lower():
        # The generator prompt requires this sign-off; missing it means the model drifted
        # from its instructions (or the answer was truncated).
        rel.soft_failures["missing_signoff"] += 1


# ---------------------------------------------------------------------------
# Reliability: benchmark + report
# ---------------------------------------------------------------------------

def rel_benchmark(rag):
    """Send every question REL_REPEATS times through the full RagPipeline.invoke,
    the same entrypoint the chat app would call."""
    rel = Reliability()
    print(f"[reliability] measuring ({len(QUESTION_SET)} questions x {REL_REPEATS} repeats)...")
    for _, question in QUESTION_SET:
        for _ in range(REL_REPEATS):
            result = call_with_retries(lambda q=question: rag.invoke(q), rel)
            if result is not None:
                check_soft_failures(result, rel)
    return rel


def rel_rates(rel):
    """All rates as percentages of requests sent."""
    calls = rel.calls or 1
    soft_total = sum(rel.soft_failures.values())
    return {
        "success_rate": 100 * rel.successes / calls,
        "error_rate": 100 * rel.failures / calls,
        "retry_rate": 100 * rel.retried_requests / calls,     # % of requests needing a retry
        "soft_failure_rate": 100 * soft_total / calls,
    }


def rel_report(rel):
    rates = rel_rates(rel)

    print("\n" + "=" * 64)
    print("RELIABILITY")
    print("=" * 64)
    print(f"total requests      : {rel.calls}")
    print(f"successful          : {rel.successes}")
    print(f"failed (hard)       : {rel.failures}")
    print(f"needed a retry      : {rel.retried_requests}   ({rel.retry_attempts} retry attempts)")
    print("-" * 64)
    print(f"success rate        : {rates['success_rate']:.2f}%")
    print(f"error rate          : {rates['error_rate']:.2f}%")
    print(f"retry rate          : {rates['retry_rate']:.2f}%")
    print("-" * 64)
    print("soft failures (answered, but unusable):")
    for name, count in rel.soft_failures.items():
        print(f"   {name:<17}: {count}")
    print(f"soft failure rate   : {rates['soft_failure_rate']:.2f}%")
    if rel.errors:
        print("-" * 64)
        print("errors:")
        for err in rel.errors:
            print(f"   {err}")
    print("=" * 64)

    hard = "PASS" if rates["success_rate"] >= SLO_SUCCESS_RATE_PCT else "FAIL"
    soft = "PASS" if rates["soft_failure_rate"] <= SLO_SOFT_FAILURE_RATE_PCT else "FAIL"
    print(f"SLO: success rate      >= {SLO_SUCCESS_RATE_PCT:5.1f}%  ->  {rates['success_rate']:6.2f}%   [{hard}]")
    print(f"SLO: soft failure rate <= {SLO_SOFT_FAILURE_RATE_PCT:5.1f}%  ->  {rates['soft_failure_rate']:6.2f}%   [{soft}]")
    print("=" * 64)
    print("note: single-user on one machine this should read ~100%; it only gets")
    print("      interesting under concurrency, rate limits and flaky networks.")


# ---------------------------------------------------------------------------
# Reliability: run
# ---------------------------------------------------------------------------

def run_reliability(rag, verbose=True):
    """Measure reliability; print the report and return a flat metrics dict."""
    rel = rel_benchmark(rag)
    if verbose:
        rel_report(rel)

    rates = rel_rates(rel)
    return {
        "total_requests": rel.calls,
        **rates,
        **{f"soft_{name}": count for name, count in rel.soft_failures.items()},
        "slo_success_pass": rates["success_rate"] >= SLO_SUCCESS_RATE_PCT,
        "slo_soft_failure_pass": rates["soft_failure_rate"] <= SLO_SOFT_FAILURE_RATE_PCT,
    }


# ===========================================================================
# ENTRYPOINT: run all three, return the operational snapshot
# ===========================================================================

def run_ops(rag=None, verbose=True):
    """One pipeline, three evals, one merged snapshot. The dotted keys become metric ids:
    this dict is the operational slice of what regression testing diffs against a baseline."""
    rag = rag or RagPipeline()

    latency = run_latency(rag, verbose=verbose)
    cost = run_cost(rag, verbose=verbose)
    reliability = run_reliability(rag, verbose=verbose)

    snapshot = {
        "config.generator_model": GENERATOR_MODEL,
        "config.embedding_model": EMBEDDING_MODEL,
        "config.fetch_k": rag.retriever.fetch_k,
        "config.top_k": rag.retriever.k,
    }
    snapshot.update({f"latency.{k}": v for k, v in latency.items()})
    snapshot.update({f"cost.{k}": v for k, v in cost.items()})
    snapshot.update({f"reliability.{k}": v for k, v in reliability.items()})
    return snapshot


def main():
    snapshot = run_ops(verbose=True)

    print("\n" + "=" * 70)
    print("OPERATIONAL SNAPSHOT  (feeds regression testing)")
    print("=" * 70)
    for key, value in snapshot.items():
        shown = f"{value:.6f}" if isinstance(value, float) else str(value)
        print(f"  {key:<40} {shown}")
    print("=" * 70)


if __name__ == "__main__":
    main()
