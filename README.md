# Bengali LLM Hallucination Detection



---

## The Problem

Large language models frequently generate responses that sound fluent and confident but are factually wrong, a phenomenon known as **hallucination**. This is especially severe for Bengali, the 6th most spoken language in the world, where training data is sparse, retrieval databases are limited, and most evaluation benchmarks simply don't exist.

The task: given a Bengali question, an LLM-generated response, and (for some rows) a grounding context passage, classify whether the response is **faithful** (`label = 1`) or **hallucinated** (`label = 0`).

The dataset has two tracks:
- **Track A** — context passage is provided. The response should be verifiable against it.
- **Track B** — closed-book. No context. The response must be judged on factual correctness alone.

Evaluation metric: **Macro F1** across both classes.

---

## Approach

Rather than throwing everything at an LLM judge, this solution uses a **cascaded multi-layer pipeline** that exhausts cheap, high-confidence signals first and only escalates to the LLM for genuinely ambiguous rows. This keeps the pipeline fast, cost-effective, and interpretable.

### Pipeline Overview

```
(prompt, response, context?)
         │
         ▼
┌─────────────────────────────────┐
│  L1 — Sample Answer-Key Match   │  char TF-IDF vs. labelled sample set (≥0.90)
│  Exact pair / gold-agree /      │  → copy known label directly
│  wrong-agree                    │
└──────────────┬──────────────────┘
               │ undecided
               ▼
┌─────────────────────────────────┐
│  L2 — Squad-BN Retrieval        │  TF-IDF + BGE-M3 dense hybrid vs. squad_bn
│  TF-IDF ≥ 0.88 / Dense ≥ 0.92  │  → answer-type gate before accepting gold
└──────────────┬──────────────────┘
               │ undecided / disagree
               ▼
┌─────────────────────────────────┐
│  L3 — Context Span Verification │  Track A only
│  Relation-span extraction       │  → verify date/event against the correct sentence
└──────────────┬──────────────────┘
               │ undecided
               ▼
┌─────────────────────────────────┐
│  QB v2 — Exam Bank Lookup       │  TF-IDF + BGE-M3 hybrid over 4 Bengali QA sets
│  Decide ≥ 0.92 / Hint ≥ 0.80   │  + cross-exam of all disagree rows
└──────────────┬──────────────────┘
               │ still undecided
               ▼
┌─────────────────────────────────┐
│  LLM Judge — Qwen2.5-32B-AWQ   │  Route-specific prompts + Bengali Wikipedia RAG
│  n=3 self-consistency voting    │  math / language / date / factual
└──────────────┬──────────────────┘
               │ tie / LLM unavailable
               ▼
         (route, track) prior
```


---

### Route-Specific LLM Prompts

A single generic judge prompt fails across diverse question types. Four dedicated prompts are used:

| Prompt | Purpose |
|--------|---------|
| `SYS_DATE_JUDGE` | Forces the judge to identify the *specific event's* date, not any date in the passage |
| `SYS_LANG_JUDGE` | Bengali linguist persona — handles antonyms, idioms, sandhi, samasa with semantic equivalence |
| `SYS_MATH` | Full step-by-step solver: sign handling, day-of-week modulo, fractions, interest, profit/loss |
| `SYS_FACTUAL_JUDGE` | Instructs the judge that "no bank match ≠ hallucination" — handles well-known Bengali facts that retrieval systems miss |

For language questions, an **answer-first** mode is also used: the model independently generates the correct answer, then compares to the candidate — rather than being asked "is this right?" which can bias the judge.

---

## Retrieval Databases Used

### Exam Bank (Question Bank)
Four Bengali QA datasets downloaded and indexed at runtime:

| Dataset | Description |
|---------|-------------|
| [BEnQA](https://github.com/sheikhshafayat/BEnQA) | Bengali MCQ exam questions — Bengali columns only |
| [BanglaMedQA](https://huggingface.co/datasets/ajwad-abrar/BanglaMedQA) | Bengali medical QA |
| [BanglaRQA](https://huggingface.co/datasets/sartajekram/BanglaRQA) | Bengali reading comprehension |
| [BanglaQuAD](https://github.com/rashad101/BanglaQuAD-LREC-COLING-24) | Bengali SQuAD-style QA |

### SQuAD-BN
`csebuetnlp/squad_bn` — up to 150,000 Bengali reading comprehension QA pairs used for Layer 2 retrieval.

### RAG Corpus
Bengali Wikipedia (`wikimedia/wikipedia 20231101.bn`) — up to 250,000 chunked passages (160 words/chunk, 30-word overlap), indexed with BM25 using BanglaBERT subword tokenisation for better morphological recall.

---

## Models & Libraries

| Component | Model / Library |
|-----------|----------------|
| LLM Judge | `Qwen/Qwen2.5-32B-Instruct-AWQ` (primary) → 14B → 7B fallback |
| Inference | vLLM with tensor parallelism (TP=2 for 32B) |
| HF fallback | `Qwen2.5-14B-Instruct` in 4-bit via bitsandbytes |
| Dense Encoder | `BAAI/bge-m3` via sentence-transformers |
| Sparse Retrieval | TF-IDF character n-grams (3–5), sklearn |
| BM25 | `rank-bm25` with BanglaBERT tokenisation |
| Math solver | `sympy` for algebraic expressions |
| Environment | Kaggle GPU T4 ×2, Internet ON |

---

## Notebook Structure


| Cell | Description |
|------|-------------|
| 1 | Environment setup — installs all packages and prints a version report |
| 2 | Exam-bank preload — downloads and deduplicates BEnQA, BanglaMedQA, BanglaRQA, BanglaQuAD |
| 3 | Core pipeline — text normalisation, routing, and deterministic Layers 1 / 2 / 3 |
| 4 | LLM engine, RAG corpus, question-bank index, and all judge prompt templates |
| 5 | Dense embedder — BGE-M3 with disk caching |
| 6 | Answer-type gate re-import for downstream cells |
| 7 | Layer 2 v2 — hybrid TF-IDF + dense squad-bn matching with calibration sweep |
| 8 | QB v2 — hybrid exam-bank lookup with calibration |
| 9 | LLM judge workload — per-route dispatch with self-consistency voting |
| 10 | Merge, QB cross-examination, and final `submission.csv` |
| 11 | Snapshot — zips all artefacts for download |

---

## How to Run

1. Go to [Kaggle Notebooks](https://www.kaggle.com/competitions/bengali-hallucination/code) and create a new notebook.
2. Attach the competition dataset
3. Enable **GPU T4 ×2** and turn **Internet ON**.
4. Upload `llm-hallucination-olikbochon.ipynb` and click **Save & Run All**.

> Total runtime is approximately 5–6 hours on T4 ×2, dominated by LLM inference on the undecided rows.

---

## Competition

**[অলীকবচন | Bengali LLM Hallucination Detection Challenge](https://www.kaggle.com/competitions/bengali-hallucination/overview)**  
Detect hallucinations in Bengali language-model outputs. Binary classification evaluated on Macro F1.


 **Final Score:** `0.850` Macro F1 on the public leaderboard
