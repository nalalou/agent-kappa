# Validation Plan

What we need to do before going public, based on our prior art research.

## Must-Do (Non-Negotiable)

### 1. Compare against all 10 Kuncheva & Whitaker metrics
Show that κ_correct predicts vote gain better than (or as well as):
- Q-statistic
- Correlation coefficient
- Disagreement measure
- Double-fault measure
- Entropy
- Kohavi-Wolpert variance
- Generalized diversity
- Coincident failure diversity
- Difficulty index
- Plain kappa (κ_raw, not error-specific)

**How:** Compute all 10 on our multi-model data. Rank by Spearman correlation with vote boost.

### 2. Test on real benchmarks (not just our toy problems)
Minimum viable set:
- **GSM8K** (math reasoning, ~1300 problems)
- **MMLU** (knowledge/reasoning, subset of ~500)
- **HumanEval** (code generation, 164 problems)
- **ARC-Challenge** (science reasoning, ~1100 problems)

Use Ollama models. 4 agents × 4 benchmarks × 5+ models.

### 3. Statistical rigor
- Spearman rank correlation between κ_correct and vote boost
- 95% bootstrap confidence intervals on all correlations
- Report effect sizes (Cohen's d or r)
- Multiple runs for variance estimation

### 4. Test both diversity sources
- Same model, different prompts (prompt ensemble)
- Same model, different temperatures
- Different models, same prompt
- Different models, different prompts

κ_correct should predict vote gain in ALL four configurations.

## Should-Do (Strengthens Credibility)

### 5. Reproduce LLM-TOPLA comparison
- Their focal diversity metric vs our κ_correct on the same benchmarks
- If κ_correct is competitive with a simpler metric, that's a win

### 6. Integration examples
- AutoGen: notebook showing κ_correct on a GroupChat
- CrewAI: diagnostic for a crew configuration
- LangGraph: post-run diversity analysis

### 7. Kappa-error diagram
- Reproduce Margineantu & Dietterich's visualization for LLM agents
- Plot pairwise κ vs averaged error for each model pair

## Nice-to-Have

### 8. Ablation: how many calibration problems do you need?
- Test κ_correct stability with 10, 20, 50, 100, 200 calibration problems
- Find the minimum viable calibration set size

### 9. Time series: does κ_correct change during a conversation?
- In multi-round agent interactions, does agreement increase over time (convergence)?
- Is early κ_correct predictive of final ensemble quality?

### 10. Cost-weighted analysis
- Factor in per-token costs
- Show the cost-accuracy Pareto frontier: at what κ_correct is it cheaper to use 1 better model vs 4 diverse worse ones?
