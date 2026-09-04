.PHONY: setup test lint smoke benchmark benchmark-ranking build clean

setup:
	python -m pip install -e ".[dev]"

test:
	python -m pytest -q

lint:
	ruff check .

smoke:
	news-ctr make-synthetic --output data/synthetic --seed 42
	news-ctr train --data data/synthetic --output artifacts/synthetic-smoke --model logistic --seed 42 --text-components 16

benchmark:
	@test ! -e "$(CORE_BENCHMARK_OUTPUT)" || (echo "output exists: $(CORE_BENCHMARK_OUTPUT); choose CORE_BENCHMARK_OUTPUT=..."; exit 2)
	news-ctr make-synthetic --output data/benchmark --seed 42 --users 30 --articles 80 --impressions 180
	news-ctr benchmark --data data/benchmark --output "$(CORE_BENCHMARK_OUTPUT)" --models position,popularity,logistic --bootstrap-samples 200 --seed 42

benchmark-ranking:
	@test ! -e "$(RANKING_BENCHMARK_OUTPUT)" || (echo "output exists: $(RANKING_BENCHMARK_OUTPUT); choose RANKING_BENCHMARK_OUTPUT=..."; exit 2)
	news-ctr make-synthetic --output data/benchmark-ranking --seed 42 --users 30 --articles 80 --impressions 180
	news-ctr benchmark --data data/benchmark-ranking --output "$(RANKING_BENCHMARK_OUTPUT)" --models position,popularity,logistic,lightgbm --bootstrap-samples 200 --seed 42

build:
	python -m build

clean:
	python -c "import shutil; [shutil.rmtree(path, ignore_errors=True) for path in ('build', 'dist', '.pytest_cache', '.ruff_cache')]"

CORE_BENCHMARK_OUTPUT ?= artifacts/benchmark-core
RANKING_BENCHMARK_OUTPUT ?= artifacts/benchmark-ranking
