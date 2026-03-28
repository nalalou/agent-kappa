# agent-kappa

**Are your agents diverse, or are you paying for clones?**

One number tells you if majority voting will help your multi-agent team — or waste compute. Run it on your own models, on your own machine, in 10 minutes.

```bash
pip install agent-kappa
agent-kappa benchmark --model llama3.2 --pretty
```

***

## The Idea

You have 4 agents. They vote on answers. But if they all fail on the same problems, voting just agrees with the shared mistake.

`κ_correct` measures that. It's the error correlation between your agents — adapted from [ensemble diversity metrics](https://link.springer.com/article/10.1023/A:1022859003006) (Kuncheva & Whitaker, 2003) for LLM agent teams.

- **Low κ_correct** → agents fail on different problems → voting helps
- **High κ_correct** → agents fail on the same problems → voting won't help

In our tests across 6 architectures, the split fell around 0.4. Run it on your own models to see where it lands for your use case.

***

## What It Looks Like

```
──────────────────────── agent-kappa benchmark ────────────────────────
ℹ Model: llama3.2 | Agents: 4 | Problems: 15
✓ Benchmark complete
Progress ██████████████████████████████████████████████████████████ 100%

=======================================================
AGENT TEAM DIAGNOSIS
=======================================================

  κ_correct:     0.127   ← primary metric
  κ_raw:         0.479
  Q-statistic:   0.532
  Disagreement:  0.208
  Double-fault:  0.042

  Individual accuracy (avg): 68.8%
    Agent 0: 75.0%
    Agent 1: 62.5%
    Agent 2: 75.0%
    Agent 3: 62.5%
  Majority vote accuracy:    91.7%
  Vote boost:                +22.9%

  VERDICT: DIVERSE — voting will help
```

The `--pretty` flag auto-downloads [gloss](https://github.com/nalalou/gloss) on first run for live progress bars. Works without it too.

***

## Use It in Code

```python
from agent_kappa import team_diagnosis

diagnosis = team_diagnosis(
    agent_outputs=[agent1_answers, agent2_answers, agent3_answers, agent4_answers],
    ground_truth=correct_answers,
)

print(diagnosis.kappa_correct)   # 0.127
print(diagnosis.verdict)         # "DIVERSE — voting will help"
print(diagnosis.vote_boost)      # 0.229
```

Or just the metrics:

```python
from agent_kappa import kappa_correct, all_diversity_metrics

kc = kappa_correct(agent_outputs, ground_truth)
# 0.127

metrics = all_diversity_metrics(agent_outputs, ground_truth)
# {'kappa_correct': 0.127, 'kappa_raw': 0.479, 'q_statistic': 0.532, ...}
```

***

## Tested Across 6 Architectures

All local, all reproducible with [Ollama](https://ollama.com).

| Model | κ_correct | Vote Boost |
|---|---|---|
| qwen2.5:3b | 0.13 | +22.9% |
| Claude Opus | 0.26 | +10.0% |
| phi3:mini | 0.37 | +16.7% |
| llama3.2 | 0.48 | +0.0% |
| llama3.1:8b | 0.78 | +0.0% |
| gemma2:2b | 0.88 | +4.2% |

Low κ_correct → vote helps. High κ_correct → vote doesn't. In our tests the split fell around 0.4, but this is from 6 data points — run the benchmark yourself.

***

## Prior Art

We didn't invent ensemble diversity metrics — [Kuncheva & Whitaker (2003)](https://link.springer.com/article/10.1023/A:1022859003006) catalogued 10 of them. [LLM-TOPLA (EMNLP 2024)](https://arxiv.org/abs/2410.03953) already applied diversity metrics to LLM ensembles. [Ma et al. (2025)](https://arxiv.org/abs/2506.09046) mapped neural network architecture to agent teams.

What didn't exist: a pip package that puts these metrics in the hands of people building with AutoGen, CrewAI, and LangGraph. None of those frameworks measure diversity. This does.

Full landscape: [docs/prior-art.md](docs/prior-art.md)

***

## License

MIT
