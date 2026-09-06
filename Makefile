.PHONY: install test lint demo sweep

install:
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check src tests experiments
	uv run ruff format --check src tests experiments

# needs OLLAMA_HOST pointing at an ollama w/ a tool calling model, or pass POLICY=gemini
POLICY ?= ollama
demo:
	./experiments/demo.sh $(POLICY)

sweep:
	uv run python experiments/lr_sweep.py --steps 600 --out runs/sweep
