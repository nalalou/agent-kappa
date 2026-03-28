"""
CLI entry point: python -m agent_kappa

Usage:
    agent-kappa benchmark --model llama3.2
    agent-kappa benchmark --model llama3.2 --pretty
    agent-kappa benchmark --list-models
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="agent-kappa",
        description="Measure diversity in multi-agent LLM teams",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Benchmark subcommand
    bench = subparsers.add_parser(
        "benchmark",
        help="Run the built-in benchmark on an Ollama model",
    )
    bench.add_argument(
        "--model", type=str, default="llama3.2",
        help="Ollama model name (default: llama3.2)",
    )
    bench.add_argument(
        "--agents", type=int, default=4,
        help="Number of agents (default: 4)",
    )
    bench.add_argument(
        "--problems", type=int, default=0,
        help="Number of problems to run (0 = all, default: all)",
    )
    bench.add_argument(
        "--pretty", action="store_true",
        help="Render output with gloss (auto-downloads on first use)",
    )
    bench.add_argument(
        "--list-models", action="store_true",
        help="List available Ollama models and exit",
    )

    args = parser.parse_args()

    if args.command == "benchmark":
        from agent_kappa.benchmark import run_benchmark, list_models

        if args.list_models:
            list_models()
        elif args.pretty:
            from agent_kappa.gloss import run_with_gloss
            cmd = [sys.executable, "-m", "agent_kappa", "benchmark",
                   "--model", args.model,
                   "--agents", str(args.agents)]
            if args.problems:
                cmd += ["--problems", str(args.problems)]
            run_with_gloss(cmd)
        else:
            run_benchmark(
                model=args.model,
                num_agents=args.agents,
                num_problems=args.problems or None,
            )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
