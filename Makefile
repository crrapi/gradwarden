.PHONY: install test lint demo sweep bench

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

# every scenario x none/rules/rules-early. add BENCH_POLICIES="... ollama" to include the agent
BENCH_POLICIES ?= none rules rules-early
bench:
	uv run python experiments/bench.py --steps 600 --out runs/bench --policies $(BENCH_POLICIES)
