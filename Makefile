.PHONY: setup test lint smoke build clean

setup:
	python -m pip install -e ".[dev]"

test:
	python -m pytest -q

lint:
	ruff check .

smoke:
	news-ctr make-synthetic --output data/synthetic --seed 42
	news-ctr train --data data/synthetic --output artifacts/synthetic-smoke --model logistic --seed 42 --text-components 16

build:
	python -m build

clean:
	python -c "import shutil; [shutil.rmtree(path, ignore_errors=True) for path in ('build', 'dist', '.pytest_cache', '.ruff_cache')]"
