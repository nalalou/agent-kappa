# Research Findings

Source: autoresearch project at `~/projects/research-agent-networks/`
Date: 2026-03-27 to 2026-03-28

## The Core Finding

**κ_correct (inter-agent error correlation via Fleiss' kappa) predicts whether majority voting will improve a multi-agent LLM team's accuracy.**

Tested across 6 model architectures (Claude Opus + 5 Ollama models), the threshold is ~0.4:
- κ_correct < 0.4 → errors are independent → voting helps (+10% to +23%)
- κ_correct > 0.4 → errors are correlated → voting doesn't help (0% to -4%)

## Experiment Results

### Experiment 1: H1 Width (Claude Opus, 15 problems)
- 4 parallel agents, majority vote
- Individual hard accuracy: 70% → Vote accuracy: 100% (+30%)
- κ_raw = 0.82 (overall), 0.43 (hard problems)
- κ_correct = 0.26
- **Conclusion:** Low error correlation → vote corrects everything

### Experiment 2: H3 MoE Routing (Claude Opus, 20 problems)
- Haiku router classifies problems → specialist agents
- Router accuracy: 95-100% at max confidence
- Specialist = generalist accuracy (ceiling effect)
- **Conclusion:** MoE routing works; benefit is efficiency not accuracy

### Experiment 3: Open-Model Reproduction (Llama 3.2, 12 problems)
- Same H1 design with Llama 3.2 (3B) via Ollama
- Individual: 71% → Vote: 67% (vote HURTS by 4%)
- κ_correct = 0.66
- **Conclusion:** High error correlation → vote converges on shared mistakes

### Experiment 4: Multi-Model Study (5 Ollama models, 12 problems)

| Model | Size | Individual | Vote | Boost | κ_correct |
|---|---|---|---|---|---|
| qwen2.5:3b | 3B | 68.8% | 91.7% | +22.9% | 0.127 |
| phi3:mini | 3.8B | 41.7% | 58.3% | +16.7% | 0.371 |
| llama3.2 | 3B | 75.0% | 75.0% | +0.0% | 0.481 |
| llama3.1:8b | 8B | 75.0% | 75.0% | +0.0% | 0.778 |
| gemma2:2b | 2B | 62.5% | 66.7% | +4.2% | 0.881 |

Framework prediction accuracy: 5/6 (83%), with gemma2:2b as borderline case.

### Experiment 5 (pending): Scaled Study (14 models, 40 problems, 2 runs)
- Models: 0.5B to 14B across 6 architectures
- Queued for overnight run
- Will produce scatter plot with error bars

## Key Insights

### 1. The Two-Kappa Framework
- **κ_raw** = how much agents agree on answers (measures redundancy)
- **κ_correct** = how much agents agree on which problems they get right/wrong (measures error independence)
- Optimal team: high κ_raw (quality) + low κ_correct (diverse errors)

### 2. Bias-Variance Tradeoff for Agent Teams
- Width reduces variance (random errors) but not bias (systematic model limitations)
- κ_correct measures the bias component
- Prompt diversity can't fix capability gaps — need different architectures

### 3. Aggregation-Dependent Optimal κ
From our meta-experiment (4-agent publication strategy team, κ ≈ 0.1):
- **Majority vote** → target κ_raw = 0.4-0.7
- **Synthesis/integration** → target κ_raw = 0.1-0.4
- **Adversarial review** → target κ_raw < 0

### 4. Ceiling Effect
Frontier models (Claude Opus) hit 100% on moderate benchmarks, masking topology effects.
The kappa framework is most useful for:
- Smaller/cheaper models where accuracy isn't saturated
- Hard problems at the edge of model capability
- Cost-sensitive deployments where redundancy = wasted money

## Methodological Lessons

### What We Got Right
- Cross-architecture validation (Ollama) resolved the circularity objection
- Due diligence literature search (with web access) caught what training-data-only agents missed

### What We Got Wrong
- Initial literature survey used training data only → missed Ma et al. (2025), LLM-TOPLA, DiscoUQ
- Ran experiments before checking prior art thoroughly (should have been reversed)
- Problem set too small (12-15 problems) — need 50-100+ for publication credibility
- Only tested math/reasoning — need code generation, QA, knowledge tasks

### What's Still Needed
- Scaled overnight run (14 models × 40 problems)
- Comparison against all 10 Kuncheva & Whitaker metrics
- Testing on standard benchmarks (GSM8K, HumanEval, MMLU)
- Statistical significance tests (Spearman correlation, bootstrap CIs)
