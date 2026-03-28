# DiscoUQ Deep Dive

Paper: "DiscoUQ: Structured Disagreement Analysis for Uncertainty Quantification in LLM Agent Ensembles"
Author: Bo Jiang (Anthropic) | March 2026 | arxiv:2603.20975

## What They Do

When 5 LLM agents disagree, DiscoUQ analyzes the *structure* of the disagreement — not just the vote margin — to produce calibrated confidence scores per-question.

**Key insight:** Two 3-2 splits can have radically different reliability depending on *why* agents disagree. Vote counting (and kappa) can't see this.

## Their Method

1. Deploy K=5 agents with distinct role prompts (Analytical Reasoner, Devil's Advocate, Knowledge-Focused, Intuitive Responder, Systematic Verifier)
2. Model: Qwen3.5-27B via vLLM, temperature 0.7
3. For non-unanimous questions, extract 17 features (9 linguistic + 8 geometric)
4. Feed into lightweight classifier → calibrated confidence

### Features

**Linguistic (9):** Evidence overlap, minority new information, minority argument strength, majority confidence language, reasoning complexity, divergence depth (one-hot: early/middle/late), vote confidence

**Geometric (8):** Overall dispersion, majority cohesion, cluster distance, minority outlier degree, majority centrality, minority cohesion, PCA variance ratio, vote confidence

Embeddings: BGE-large-en-v1.5 (1024 dim)

### Three Variants
- **M1: DiscoUQ-LLM** — Logistic regression on 9 linguistic features. One extra LLM call per non-unanimous question. **Best performer.**
- **M2: DiscoUQ-Embed** — Logistic regression on 8 geometric features. Zero extra LLM calls.
- **M3: DiscoUQ-Learn** — MLP on all 17 features combined.

## Results

| Method | Avg AUROC | Avg ECE |
|---|---|---|
| Vote Count (baseline) | .770 | .084 |
| Vote Entropy | .769 | .206 |
| LLM Aggregator | .791 | .098 |
| **DiscoUQ-LLM (M1)** | **.802** | **.036** |
| DiscoUQ-Embed (M2) | .773 | .040 |
| DiscoUQ-Learn (M3) | .790 | .042 |

Benchmarks: StrategyQA (687q), MMLU (746q), TruthfulQA (817q), ARC-Challenge (500q)

Cross-benchmark generalization: near-zero degradation (Δ = -0.001 AUROC).

## How It Relates to agent-kappa

**They solve a DIFFERENT problem.** This is important.

| | agent-kappa | DiscoUQ |
|---|---|---|
| **Level** | Dataset-level | Per-question |
| **Question** | "Is my team well-composed?" | "Should I trust this specific answer?" |
| **When** | Before deployment (calibration) | During inference (runtime) |
| **Cost** | Cheap (one calibration run) | Expensive (extra LLM call per question) |
| **Metric** | κ_correct (single number) | Calibrated confidence (per-instance) |
| **Use case** | Team design / cost optimization | Selective abstention / UQ |

**They are complementary, not competitive:**
- Use agent-kappa to design your team (pick diverse agents, avoid redundancy)
- Use DiscoUQ at runtime to decide per-question confidence

**What DiscoUQ proves that helps us:**
1. Structured disagreement analysis IS valuable (validates the research direction)
2. Vote counting alone is insufficient (motivates richer metrics like κ_correct)
3. 5 diverse agents with role prompts is a viable setup
4. Cross-benchmark generalization works (features transfer across domains)

## Code

Not released yet. Paper states "Code and data will be made available upon publication."

## Limitations
- Only closed-form QA (yes/no, multiple choice)
- Single primary model (Qwen3.5-27B), limited cross-model testing
- Requires labeled data for classifier training
- Fixed at 5 agents
- M1 adds ~1,624 extra LLM calls (cost overhead)
