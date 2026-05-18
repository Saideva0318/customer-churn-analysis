.PHONY: help install lint format test test-cov train predict clean

## help: Show this help
help:
	@echo "Customer Churn Analysis — Available commands:"
	@sed -n 's/^##//p' $(MAKEFILE_LIST) | column -t -s ':' | sed -e 's/^/ /'

## install: Install Python dependencies
install:
	pip install -r requirements.txt
	pre-commit install

## lint: Run flake8 linter
lint:
	flake8 src/ tests/ --max-line-length=120 --ignore=E203,W503

## format: Format code with Black + isort
format:
	black src/ tests/ --line-length 120
	isort src/ tests/

## test: Run unit tests
test:
	pytest tests/ -v --tb=short

## test-cov: Run tests with coverage
test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

## train: Train the XGBoost churn model
train:
	python -m src.train --config config/model_config.yaml

## predict: Run batch prediction on new data
predict:
	python -m src.predict --input data/raw/customers_new.csv

## report: Generate model evaluation report
report:
	python -m src.evaluate --output reports/

## clean: Remove cache and artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	rm -rf .pytest_cache htmlcov .coverage models/*.pkl
