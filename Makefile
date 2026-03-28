GLOSS := /Users/noraalalou/projects/gloss/gloss

.PHONY: help install test bench bench-pretty bench-all lint clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install in dev mode
	pip install -e ".[dev]" 2>/dev/null || pip install -e .

test: ## Run tests
	python -m pytest tests/ -v

bench: ## Run benchmark on default model (llama3.2)
	python -m agent_kappa benchmark --model llama3.2

bench-pretty: ## Run benchmark with gloss rendering
	python -m agent_kappa benchmark --model llama3.2 2>&1 | $(GLOSS) watch

bench-all: ## Run benchmark on all available Ollama models with gloss
	@for model in $$(ollama list 2>/dev/null | tail -n +2 | grep -v llava | awk '{print $$1}'); do \
		echo "::divider $$model" ; \
		python -m agent_kappa benchmark --model $$model ; \
		echo ; \
	done 2>&1 | $(GLOSS) watch

bench-model: ## Run benchmark on a specific model: make bench-model MODEL=qwen2.5:3b
	python -m agent_kappa benchmark --model $(MODEL) 2>&1 | $(GLOSS) watch

overnight: ## Run full overnight study (all models, 40 problems, 2 runs)
	@echo "::status id=overnight running Starting overnight run..."
	cd /Users/noraalalou/projects/research-agent-networks && \
		bash experiments/scaled-study/run_overnight.sh 2>&1 | $(GLOSS) watch

list-models: ## List available Ollama models
	python -m agent_kappa benchmark --list-models

lint: ## Lint with ruff
	ruff check src/ tests/

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info src/*.egg-info __pycache__ .pytest_cache
	find . -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
