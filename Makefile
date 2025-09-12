# Equity Factor Backtesting Framework - Development Makefile

.PHONY: help install test clean format lint run-example setup-dev

help:
	@echo "Available commands:"
	@echo "  install     - Install the package and dependencies"
	@echo "  test        - Run unit tests"
	@echo "  clean       - Clean cache and temporary files"
	@echo "  format      - Format code with black"
	@echo "  lint        - Run linting with flake8"
	@echo "  run-example - Run the simple momentum strategy example"
	@echo "  setup-dev   - Set up development environment"
	@echo "  notebook    - Start Jupyter notebook server"

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

test-coverage:
	python -m pytest tests/ --cov=src/equity_backtesting --cov-report=html

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf data/cache/
	rm -rf htmlcov/
	rm -rf .coverage

format:
	black src/ tests/ examples/ config/
	isort src/ tests/ examples/ config/

lint:
	flake8 src/ tests/ examples/ --max-line-length=88 --extend-ignore=E203,W503
	mypy src/equity_backtesting --ignore-missing-imports

run-example:
	python examples/simple_momentum_strategy.py

setup-dev: install-dev
	@echo "Setting up development environment..."
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env file from template"; fi
	@echo "Development setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Edit .env file with your API keys (optional)"
	@echo "2. Run 'make run-example' to test installation"
	@echo "3. Run 'make notebook' to start Jupyter notebooks"

notebook:
	jupyter notebook notebooks/

build:
	python setup.py sdist bdist_wheel

check: lint test
	@echo "All checks passed!"

.PHONY: all
all: clean format lint test
	@echo "Full development cycle completed!"