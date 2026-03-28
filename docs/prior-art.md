# Prior Art & Landscape Analysis

Last updated: 2026-03-28

This document maps everything we found that's related to agent-kappa. We conducted this analysis specifically to avoid claiming false novelty. If you find something we missed, please open an issue.

---

## 1. Classical Ensemble Diversity Metrics (1997-2003)

The idea of measuring diversity in classifier ensembles is **well-established**.

### Kuncheva & Whitaker (2003) — "Measures of Diversity in Classifier Ensembles"
- **The canonical reference.** Catalogued 10 diversity metrics for classifier ensembles.
- Includes pairwise kappa statistic as one of four averaged pairwise measures.
- **Critical finding:** All 10 measures showed weak/inconsistent correlation with actual ensemble accuracy improvement.
- Thousands of citations. If you're working on ensemble diversity, you cite this paper.
- Paper: https://link.springer.com/article/10.1023/A:1022859003006

### Margineantu & Dietterich (1997) — Kappa-Error Diagrams
- Introduced kappa-error diagrams: plot pairwise kappa (x) vs averaged error (y) for ensemble pairs.
- Used for ensemble pruning ("kappa pruning").
- Standard visualization tool in the ensemble literature.

### The 10 Classical Diversity Metrics
From Kuncheva & Whitaker. Any serious work in this space should compare against these:

| Metric | Type | What it measures |
|---|---|---|
| Q-statistic (Yule's Q) | Pairwise | Association between classifier outputs |
| Correlation coefficient | Pairwise | Pearson correlation of correct/incorrect |
| Disagreement measure | Pairwise | Frequency of exactly one correct |
| Double-fault measure | Pairwise | P(both wrong simultaneously) |
| **Kappa statistic** | Non-pairwise | Inter-rater agreement beyond chance |
| Entropy of votes | Non-pairwise | Spread of votes across classes |
| Kohavi-Wolpert variance | Non-pairwise | Variance of correct/incorrect across ensemble |
| Generalized diversity | Non-pairwise | Probability of disagreement |
| Coincident failure diversity | Non-pairwise | Diversity conditioned on at least one failure |
| Difficulty index | Non-pairwise | Variance of per-item difficulty |

### DESlib (Python package)
- scikit-learn-contrib library for Dynamic Ensemble Selection.
- `deslib.util.diversity` module implements Q-statistic, kappa, disagreement, correlation, double-fault, Kohavi-Wolpert variance.
- Works on binary classifier predictions (sklearn format), NOT LLM outputs.
- GitHub: https://github.com/scikit-learn-contrib/DESlib

### EnsembleDiversityTests (kbogas)
- Standalone Python repo implementing Q-stat, correlation, Cohen's kappa, KW variance.
- Simpler than DESlib, easier to fork.
- GitHub: https://github.com/kbogas/EnsembleDiversityTests

**Our position:** We adapt these classical metrics to LLM agent teams — a domain where no pip-installable tool exists. We do NOT claim to have invented ensemble diversity measurement.

---

## 2. The NN ↔ Agent Analogy (2024-2025)

The core observation that "agent teams are structurally like neural networks" has been made.

### Ma et al. (2025) — "Agentic Neural Networks" (LMU Munich / Schmidhuber orbit)
- **The most direct prior art for the structural isomorphism.**
- Maps: agents → nodes, layers → cooperative teams, forward pass → task decomposition, backward pass → textual backpropagation.
- Implements momentum-based prompt updates: `f_l^(t+1) = f_l^t - eta * G_local,l^t`
- Results: 87.8% HumanEval, 95% DABench.
- Paper: https://arxiv.org/abs/2506.09046

### Chen et al. (2025) — Textual Equilibrium Propagation
- Proves deep agent pipelines suffer the SAME pathologies as deep NNs (vanishing/exploding textual gradients).
- Fix mirrors equilibrium propagation from neuroscience.
- Paper: https://arxiv.org/abs/2601.21064

### Together AI (2024) — Mixture of Agents
- Layered feedforward agent architecture. Section 2.3 explicitly titled "Analogy to Mixture-of-Experts."
- Discovered "collaborativeness" — LLMs generate better responses given other models' outputs.
- Paper: https://arxiv.org/abs/2406.04692

### TextGrad, Trace, LLM-AutoDiff
- Family of frameworks extending automatic differentiation to agent pipelines.
- TextGrad: natural language backpropagation.
- Trace (NeurIPS 2024): computational workflow graphs with execution trace propagation.
- LLM-AutoDiff: textual gradients for cyclic architectures with skip connections.
- Papers: https://arxiv.org/abs/2406.16218, https://arxiv.org/abs/2501.16673

### GPTSwarm (Zhuge et al., 2024, ICML Oral)
- Agents as computational graphs, node/edge optimization via REINFORCE.
- From Schmidhuber's lab. Uses "society of mind" framing, not NN metaphor.
- Paper: https://arxiv.org/abs/2402.16823

### ADAS — Automated Design of Agentic Systems (Hu et al., 2024, ICLR 2025)
- Explicit NAS (Neural Architecture Search) analogy for agent systems.
- Meta-agent searches the space of possible agent architectures.
- Paper: https://arxiv.org/abs/2408.08435

**Our position:** We do NOT claim the NN ↔ agent analogy is novel. Our contribution is the practical diagnostic tool (κ_correct), not the theoretical framework.

---

## 3. LLM Ensemble Diversity (2024-2026)

Work specifically on measuring/optimizing diversity in LLM teams.

### LLM-TOPLA (EMNLP 2024) — "Efficient LLM Ensemble by Maximising Diversity"
- **Closest to our diversity-for-team-composition thesis.**
- Introduces "focal diversity metric" that captures error-pattern diversity between models.
- Uses genetic algorithm to prune down to top-k sub-ensembles.
- Outperforms Mixtral by 2.2% on MMLU, "More Agents" by 2.1% on GSM8k.
- Paper: https://arxiv.org/abs/2410.03953

### DiscoUQ (March 2026) — "Structured Disagreement Analysis for Uncertainty Quantification"
- Extracts structured inter-agent disagreement features (linguistic + geometric).
- Used for uncertainty quantification, not team composition.
- AUROC 0.802 on 5-agent ensembles across StrategyQA, MMLU, TruthfulQA, ARC-Challenge.
- Paper: https://arxiv.org/abs/2603.20975

### Multi-LLM Thematic Analysis (Feb 2026)
- **Literally uses Cohen's kappa on multi-LLM outputs.**
- Measures intra-model consistency (same model, multiple runs), NOT inter-agent diversity.
- Gemini κ=0.907, GPT-4o κ=0.853, Claude κ=0.842.
- Paper: https://arxiv.org/abs/2512.20352

### "More Agents Is All You Need" / Agent Forest (TMLR 2024)
- Scales agents with majority vote. Does NOT measure diversity.
- Uses BLEU for code, occurrence frequency for MC. No kappa, no formal agreement metric.
- Paper: https://arxiv.org/abs/2402.05120
- Code: https://github.com/MoreAgentsIsAllYouNeed/AgentForest

### ReConcile (ACL 2024) — Multi-model consensus via confidence-weighted voting
- Diverse LLMs with multi-round discussion. Diversity is a given (different models), not measured.
- Paper: https://arxiv.org/abs/2309.13007

### Dipper (EMNLP 2025) — Diversity in Prompts for LLM Ensembles
- Prompt diversity with single model can match multi-model ensembles.
- Paper: https://arxiv.org/abs/2412.15238

### "Diverse LLMs or Diverse Question Interpretations?" (2025)
- Question interpretation diversity with single LLM can be competitive.
- Paper: https://arxiv.org/abs/2507.21168

### "Beyond Majority Voting" (2025)
- Higher-order information beyond simple voting for LLM aggregation.
- Paper: https://arxiv.org/abs/2510.01499

**Our position:** LLM-TOPLA proved diversity metrics help. We use established metrics (kappa) rather than a custom one, test cross-architecture, and package it as a usable tool. DiscoUQ solves a different problem (UQ vs team composition).

---

## 4. Agent Scaling & Evaluation

### Qian et al. (2024, ICLR 2025) — Collaborative Scaling Laws
- Agent performance follows logistic growth, NOT power law (unlike neural scaling laws).
- "Collaborative emergence" occurs earlier and at smaller scale.
- Saturation around 2^5 agents. Irregular topologies > regular.
- Paper: https://arxiv.org/abs/2406.07155

### Kim et al. (2025, Google) — "Towards a Science of Scaling Agent Systems"
- Multi-agent effectiveness ranges from +81% to -70% depending on task.
- Regression model (R²=0.513) predicts optimal architecture from task properties.
- Paper: https://arxiv.org/abs/2512.08296

### "Why Do Multi-Agent LLM Systems Fail?" (NeurIPS 2025)
- MAST taxonomy: 14 failure modes across 1600+ traces.
- Used Cohen's kappa (0.88) for inter-annotator agreement on their taxonomy.
- Key: coordination failures, not capability failures, break MAS.
- Paper: https://arxiv.org/abs/2503.13657

### MARBLE / MultiAgentBench (ACL 2025)
- Evaluates collaboration + competition across 6 scenarios.
- Supports star/chain/tree/graph topologies. No diversity metrics (gap we could fill).
- Code: https://github.com/ulab-uiuc/MARBLE

---

## 5. Existing Tools & Packages

| Tool | What it does | LLM-compatible? | Diversity metrics? |
|---|---|---|---|
| DESlib | Dynamic ensemble selection for sklearn | No | Yes (Q, κ, disagreement, etc.) |
| EnsembleDiversityTests | Standalone diversity metrics | No | Yes (subset) |
| `agreement` (pip) | General inter-rater agreement | Partially | κ, Krippendorff's α |
| UQLM (CVS Health) | LLM hallucination detection via UQ | Yes | Consistency-based |
| ConsensusLLM | Multi-LLM consensus with weighting | Yes | Threshold-based only |
| AutoGen / CrewAI / LangGraph | Agent frameworks | Yes | **None** |

**The gap we fill:** No pip package adapts classical ensemble diversity metrics for LLM agent outputs. The major agent frameworks have zero diversity measurement built in.

---

## 6. What agent-kappa Actually Contributes

Given all the above, here's what we're NOT claiming and what we ARE contributing:

### NOT claiming:
- ❌ We invented ensemble diversity metrics (Kuncheva & Whitaker, 2003)
- ❌ We discovered the NN ↔ agent analogy (Ma et al., 2025)
- ❌ We were first to use diversity for LLM team composition (LLM-TOPLA, 2024)
- ❌ We were first to use kappa on LLM outputs (Multi-LLM Thematic Analysis, 2026)

### ARE contributing:
- ✅ First pip-installable tool for LLM agent team diversity diagnosis
- ✅ κ_correct (error correlation) as a specific predictor of majority vote gain
- ✅ Cross-architecture validation across 6+ model families via Ollama
- ✅ Practical threshold (κ_correct ≈ 0.4) validated on local open-weight models
- ✅ Insight that optimal κ depends on aggregation method (voting vs synthesis)
- ✅ Filling the tool gap in AutoGen/CrewAI/LangGraph ecosystem

---

## 7. Required Citations

Any blog post, paper, or documentation MUST cite:

1. **Kuncheva & Whitaker (2003)** — canonical ensemble diversity reference
2. **Margineantu & Dietterich (1997)** — kappa-error diagrams
3. **Ma et al. (2025)** — Agentic Neural Networks (structural isomorphism)
4. **LLM-TOPLA (2024)** — diversity for LLM ensemble composition
5. **Together AI (2024)** — Mixture of Agents
6. **Li et al. (2024)** — More Agents Is All You Need

Should also cite if relevant:
- DiscoUQ (2026) — structured disagreement for UQ
- Qian et al. (2024) — collaborative scaling laws
- TextGrad / Trace — optimization analogy
