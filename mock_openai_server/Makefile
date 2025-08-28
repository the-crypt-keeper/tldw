.PHONY: help install install-dev test test-cov lint format type-check clean run docker-build docker-run

help:  ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install:  ## Install the package
	pip install -e .

install-dev:  ## Install with all development dependencies
	pip install -e ".[all]"

test:  ## Run tests
	pytest

test-cov:  ## Run tests with coverage report
	pytest --cov --cov-report=term-missing --cov-report=html

lint:  ## Run linting with ruff
	ruff check mock_openai tests

format:  ## Format code with black and isort
	black mock_openai tests
	isort mock_openai tests

type-check:  ## Run type checking with mypy
	mypy mock_openai

quality:  ## Run all code quality checks
	@echo "Running black..."
	@black --check mock_openai tests
	@echo "Running isort..."
	@isort --check-only mock_openai tests
	@echo "Running ruff..."
	@ruff check mock_openai tests
	@echo "Running mypy..."
	@mypy mock_openai
	@echo "All quality checks passed!"

fix:  ## Auto-fix code issues
	black mock_openai tests
	isort mock_openai tests
	ruff check --fix mock_openai tests

clean:  ## Clean build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run:  ## Run the mock server
	python -m mock_openai.server

run-dev:  ## Run the mock server with auto-reload
	python -m mock_openai.server --reload

docker-build:  ## Build Docker image
	docker build -t mock-openai-server .

docker-run:  ## Run Docker container
	docker run -p 8080:8080 mock-openai-server

docker-compose-up:  ## Start services with docker-compose
	docker-compose up

docker-compose-down:  ## Stop services with docker-compose
	docker-compose down