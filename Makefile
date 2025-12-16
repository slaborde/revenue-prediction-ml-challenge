# Makefile for Revenue Prediction System

.PHONY: help install test run docker-build docker-up docker-down clean

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make test          - Run tests"
	@echo "  make run           - Run API locally"
	@echo "  make docker-build  - Build Docker images"
	@echo "  make docker-up     - Start Docker services"
	@echo "  make docker-down   - Stop Docker services"
	@echo "  make clean         - Clean temporary files"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=html

run:
	python -m src.api.app

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage

format:
	black src/ tests/
	isort src/ tests/

lint:
	flake8 src/ tests/
	pylint src/
