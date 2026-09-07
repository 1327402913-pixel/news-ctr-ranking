.PHONY: setup setup-decision setup-causal test lint smoke benchmark benchmark-ranking analytics experiment causal-impact-v4 portfolio-v3 build clean

setup:
	python -m pip install -e ".[dev]"

setup-decision:
	python -m pip install -e ".[dev,decision]"

setup-causal:
	python -m pip install -e ".[dev,causal]"

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

analytics:
	@test ! -e "$(ANALYTICS_DATA)" || (echo "data exists: $(ANALYTICS_DATA); choose ANALYTICS_DATA=..."; exit 2)
	@test ! -e "$(ANALYTICS_OUTPUT)" || (echo "output exists: $(ANALYTICS_OUTPUT); choose ANALYTICS_OUTPUT=..."; exit 2)
	news-ctr make-synthetic --output "$(ANALYTICS_DATA)" --seed 42 --users 30 --articles 80 --impressions 180
	news-ctr analyze --data "$(ANALYTICS_DATA)" --output "$(ANALYTICS_OUTPUT)" --split validation --min-cell-count 5

experiment:
	@test ! -e "$(EXPERIMENT_INPUT)" || (echo "input exists: $(EXPERIMENT_INPUT); choose EXPERIMENT_INPUT=..."; exit 2)
	@test ! -e "$(EXPERIMENT_OUTPUT)" || (echo "output exists: $(EXPERIMENT_OUTPUT); choose EXPERIMENT_OUTPUT=..."; exit 2)
	news-ctr make-experiment --output "$(EXPERIMENT_INPUT)" --seed 42 --users 20000
	news-ctr experiment --input "$(EXPERIMENT_INPUT)" --output "$(EXPERIMENT_OUTPUT)" --config configs/experiment-v3.json

causal-impact-v4:
	@test ! -e "$(CAUSAL_INPUT)" || (echo "input exists: $(CAUSAL_INPUT); choose CAUSAL_INPUT=..."; exit 2)
	@test ! -e "$(CAUSAL_OUTPUT)" || (echo "output exists: $(CAUSAL_OUTPUT); choose CAUSAL_OUTPUT=..."; exit 2)
	news-ctr make-quasi-experiment --output "$(CAUSAL_INPUT)" --seed 42 --markets 60 --pre-weeks 20 --post-weeks 12
	news-ctr causal-impact --input "$(CAUSAL_INPUT)" --output "$(CAUSAL_OUTPUT)" --config configs/causal-impact-v4.json

portfolio-v3:
	@test ! -e "$(PORTFOLIO_V3_OUTPUT)" || (echo "output exists: $(PORTFOLIO_V3_OUTPUT); choose PORTFOLIO_V3_OUTPUT=..."; exit 2)
	$(MAKE) analytics
	$(MAKE) experiment
	python -c "from news_ctr.portfolio import assemble_portfolio; assemble_portfolio('$(ANALYTICS_OUTPUT)', '$(EXPERIMENT_OUTPUT)', '$(PORTFOLIO_V3_OUTPUT)')"

build:
	python -m build

clean:
	python -c "import shutil; [shutil.rmtree(path, ignore_errors=True) for path in ('build', 'dist', '.pytest_cache', '.ruff_cache')]"

CORE_BENCHMARK_OUTPUT ?= artifacts/benchmark-core
RANKING_BENCHMARK_OUTPUT ?= artifacts/benchmark-ranking
ANALYTICS_DATA ?= data/analytics-v3
ANALYTICS_OUTPUT ?= artifacts/analytics-v3
EXPERIMENT_INPUT ?= data/synthetic-rct-v3.parquet
EXPERIMENT_OUTPUT ?= artifacts/experiment-v3
CAUSAL_INPUT ?= data/synthetic-market-panel-v4.parquet
CAUSAL_OUTPUT ?= artifacts/causal-impact-v4-regenerated
PORTFOLIO_V3_OUTPUT ?= artifacts/portfolio-v3-regenerated
