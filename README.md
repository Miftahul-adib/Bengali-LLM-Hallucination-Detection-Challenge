# অলীকবচন — Bengali LLM Hallucination Detection

<p align="center">
  <img src="https://img.shields.io/badge/Score-0.850%20Macro%20F1-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Platform-Kaggle-blue?style=for-the-badge&logo=kaggle"/>
  <img src="https://img.shields.io/badge/Language-Bengali-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Model-Qwen2.5--32B-purple?style=for-the-badge"/>
</p>

> **Competition:** [অলীকবচন | Bengali LLM Hallucination Detection Challenge](https://www.kaggle.com/competitions/bengali-hallucination/overview)  
> **Final Score:** `0.850` Macro F1 on the public leaderboard

---

## The Problem

Large language models frequently generate responses that sound fluent and confident but are factually wrong — a phenomenon known as **hallucination**. This is especially severe for Bengali, the 6th most spoken language in the world, where training data is sparse, retrieval databases are limited, and most evaluation benchmarks simply don't exist.

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

## Key Fixes & Design Decisions

### Fix 1 — Relation-Span Context Verification (93 errors fixed)

**Problem:** The old Layer 3 would scan the entire context passage for the response. A passage mentioning both a *founding year* and a *renovation year* would incorrectly accept either year as valid for a "when was it founded?" question.

**Solution:** Map each question type to Bengali trigger words (জন্ম, মৃত্যু, প্রতিষ্ঠা, প্রকাশ, আবিষ্কার...), extract only the sentences containing the relevant trigger, and check the response against that sub-span only.

---

### Fix 2 — Answer-Type Gate (4+ errors fixed)

Before any retrieved gold answer is used to label a row, it must pass a semantic type check:

| Question keyword | Requirement |
|-----------------|-------------|
| কত সালে / কবে (year) | Response must contain a 4-digit year |
| কতটি / কয়টি (count) | Response must contain a digit |
| কে (who) | Response must not be a bare number |
| কোথায় (where) | Response must not start with a 4-digit year |
| বয়স / বছর বয়সে (age) | Response must have a number but no 4-digit year |

---

### Fix 3 — Strict Short-Answer Comparison (~40 errors fixed)

The old Jaccard-based `resp_agree` with threshold 0.5 was too lenient for short answers. `বৃহস্পতিবার` (Thursday) and `শুক্রবার` (Friday) would match. `১৯১৩` and `১৯৮৩` would match.

**Solution:** For answers ≤ 3 tokens, require near-exact match. Also explicitly check numeric sign (`ধনাত্মক ½` ≠ `ঋণাত্মক ½`) and fraction ordering (`1/2` ≠ `2/1`).

---

### Fix 4 — Relaxed QB Threshold for Language Route (52 errors fixed)

Bengali grammar questions (antonyms, idiom meanings, prefix classes) are often paraphrased differently in the question bank. A single strict similarity threshold rejected many valid matches.

**Solution:** Use a lower threshold (0.82 vs 0.88) for questions routed to the `language` track.

---

### Fix 5 — Expanded Math Routing (10 errors fixed)

The math regex was missing several common Bengali word-problem patterns. Added: day-of-week calculations, age calculations, verbal percentage expressions (শতাংশ হ্রাস/বৃদ্ধি), compound interest (চক্রবৃদ্ধি সুদ), profit/loss verbal forms, LCM/GCD, algebraic series.

---

### Fix 6 — L2 Disagree → LLM Judge (27 errors fixed)

Previously, when a retrieved Squad-BN answer *disagreed* with the candidate response, the row was automatically labelled `0`. This was wrong — a candidate can be correct even when it doesn't match the specific retrieved answer.

**Solution:** Disagree rows are re-routed to the LLM judge with the retrieved answer as a hint, not as a verdict.

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

The full solution is in [`llm-hallucination-olikbochon.ipynb`](./llm-hallucination-olikbochon.ipynb).

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
