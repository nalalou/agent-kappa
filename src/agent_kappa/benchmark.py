"""
Built-in benchmark for testing agent-kappa on Ollama models.

Usage:
    python -m agent_kappa benchmark --model llama3.2
    python -m agent_kappa benchmark --model llama3.2 | gloss watch
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from agent_kappa.diagnosis import team_diagnosis
from agent_kappa.metrics import all_diversity_metrics

# --- Benchmark problems (subset — full set in benchmarks/problems.py) ---

BUILTIN_PROBLEMS = [
    {"id": "easy-01", "q": "What is 247 multiplied by 83?", "answer": "20501", "diff": "easy"},
    {"id": "easy-02", "q": "What is 15% of 360?", "answer": "54", "diff": "easy"},
    {"id": "easy-03", "q": "What is the sum of the interior angles of a regular hexagon, in degrees?", "answer": "720", "diff": "easy"},
    {"id": "easy-04", "q": "What is the greatest common divisor of 84 and 126?", "answer": "42", "diff": "easy"},
    {"id": "easy-05", "q": "What is 2 raised to the power of 10?", "answer": "1024", "diff": "easy"},
    {"id": "med-01", "q": "How many ways can you choose 4 items from a set of 10? That is, what is C(10,4)?", "answer": "210", "diff": "medium"},
    {"id": "med-02", "q": "What is the sum of the arithmetic series 3, 7, 11, 15, ..., 99?", "answer": "1275", "diff": "medium"},
    {"id": "med-03", "q": "What is the least common multiple of 12, 18, and 45?", "answer": "180", "diff": "medium"},
    {"id": "med-04", "q": "A right triangle has legs of length 7 and 24. What is the length of the hypotenuse?", "answer": "25", "diff": "medium"},
    {"id": "med-05", "q": "What is the remainder when 2^100 is divided by 7?", "answer": "2", "diff": "medium"},
    {"id": "hard-01", "q": "How many surjective (onto) functions are there from a set of 5 elements to a set of 3 elements?", "answer": "150", "diff": "hard"},
    {"id": "hard-02", "q": "How many derangements (permutations with no fixed points) are there of 5 elements?", "answer": "44", "diff": "hard"},
    {"id": "hard-03", "q": "What is the modular inverse of 13 modulo 97? That is, find x where 0 < x < 97 and 13x = 1 (mod 97).", "answer": "15", "diff": "hard"},
    {"id": "hard-04", "q": "How many compositions (ordered sums of positive integers) are there of the number 10?", "answer": "512", "diff": "hard"},
    {"id": "hard-05", "q": "What is the 6th Catalan number C_6?", "answer": "132", "diff": "hard"},
]

AGENT_PROMPTS = [
    "Solve this math problem step by step, then give ONLY the final number on the last line.\n\n{}",
    "Think carefully about this problem. Show brief work, then put ONLY the final numeric answer on the very last line.\n\n{}",
    "You are a careful math solver. Work through this, then write ANSWER: followed by just the number.\n\n{}",
    "Solve: {}\n\nThink step by step. End with just the number.",
]


def _gloss(directive: str):
    """Print a gloss directive (works plain if not piped through gloss)."""
    print(directive, flush=True)


def _ask_ollama(model: str, question: str, agent_id: int) -> str:
    prompt = AGENT_PROMPTS[agent_id % len(AGENT_PROMPTS)].format(question)
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True, text=True, timeout=180,
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except FileNotFoundError:
        print("Error: ollama not found. Install from https://ollama.com", file=sys.stderr)
        sys.exit(1)


def _extract_number(response: str) -> str:
    lines = response.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        for prefix in ["ANSWER:", "Answer:", "answer:", "The answer is", "= ", "**"]:
            if prefix in line:
                line = line.split(prefix)[-1].strip()
        line = line.rstrip(".*$, ").replace(",", "").replace("$", "").replace("**", "").replace("`", "")
        try:
            num = float(line)
            if not math.isfinite(num):
                continue
            return str(int(num)) if num == int(num) else str(round(num, 2))
        except (ValueError, OverflowError):
            continue
    for line in reversed(lines[-3:]):
        numbers = re.findall(r"-?\d+\.?\d*", line.replace(",", ""))
        if numbers:
            try:
                num = float(numbers[-1])
                if not math.isfinite(num):
                    continue
                return str(int(num)) if num == int(num) else str(round(num, 2))
            except (ValueError, OverflowError):
                continue
    return "PARSE_FAIL"


def list_models():
    """List available Ollama models."""
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    print(result.stdout)


def run_benchmark(
    model: str = "llama3.2",
    num_agents: int = 4,
    num_problems: int | None = None,
):
    """Run the benchmark and print diagnosis."""
    problems = BUILTIN_PROBLEMS[:num_problems] if num_problems else BUILTIN_PROBLEMS
    ground_truth = [p["answer"] for p in problems]

    _gloss(f"::divider agent-kappa benchmark")
    _gloss(f"::info Model: {model} | Agents: {num_agents} | Problems: {len(problems)}")
    _gloss(f"::status id=bench running Running benchmark...")
    _gloss(f"::bar id=progress 0 Progress")

    agent_outputs: list[list[str]] = [[] for _ in range(num_agents)]
    total_calls = len(problems) * num_agents
    calls_done = 0

    for pi, p in enumerate(problems):
        marks = ""
        for a in range(num_agents):
            raw = _ask_ollama(model, p["q"], a)
            extracted = _extract_number(raw)
            agent_outputs[a].append(extracted)
            correct = extracted == p["answer"]
            marks += "+" if correct else "-"
            calls_done += 1

        # Majority vote for this problem
        answers_i = [agent_outputs[a][pi] for a in range(num_agents)]
        vote = Counter(answers_i).most_common(1)[0][0]
        vote_correct = vote == p["answer"]
        vote_mark = "+" if vote_correct else "-"

        pct = int(calls_done / total_calls * 100)
        _gloss(f"::bar id=progress {pct} Progress")
        print(f"  [{p['id']}] agents:{marks} vote:{vote_mark}", flush=True)

    _gloss(f"::bar id=progress 100 Progress")
    _gloss(f"::status id=bench done Benchmark complete")
    print()

    # Run diagnosis
    diagnosis = team_diagnosis(agent_outputs, ground_truth)
    print(diagnosis)

    # Save results
    results_dir = Path.cwd() / "agent_kappa_results"
    results_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = results_dir / f"{model.replace(':', '_')}_{timestamp}.json"

    result_data = {
        "model": model,
        "num_agents": num_agents,
        "num_problems": len(problems),
        "timestamp": timestamp,
        "kappa_correct": diagnosis.kappa_correct,
        "kappa_raw": diagnosis.kappa_raw,
        "vote_accuracy": diagnosis.vote_accuracy,
        "vote_boost": diagnosis.vote_boost,
        "avg_individual_accuracy": diagnosis.avg_individual_accuracy,
        "verdict": diagnosis.verdict,
    }

    with open(out_path, "w") as f:
        json.dump(result_data, f, indent=2)

    _gloss(f"::ok Results saved to {out_path}")
