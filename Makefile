# rag-multimodal-2026 developer commands.
# Bring up the data services first, then build and test against them.

COMPOSE = docker compose -f docker/docker-compose.yml

.PHONY: help db-up db-down stack-up stack-down install dev worker frontend load-samples test test-int lint format audit

help:
	@echo "db-up      start postgres (pgvector) and redis for local build and tests"
	@echo "db-down    stop the data services"
	@echo "stack-up   build and start the full stack (api, worker, streamlit)"
	@echo "stack-down stop the full stack"
	@echo "install    install python dependencies into .venv (after a scan)"
	@echo "dev        run the api locally"
	@echo "worker     run the celery worker locally"
	@echo "frontend   run the streamlit ui locally"
	@echo "load-samples load the bundled sample documents into the corpus"
	@echo "test       run unit tests"
	@echo "test-int   run integration tests (needs db-up)"
	@echo "lint       run black, flake8, isort checks"
	@echo "format     run black and isort"
	@echo "audit      run pip-audit on requirements.txt"

db-up:
	$(COMPOSE) up -d postgres redis

db-down:
	$(COMPOSE) stop postgres redis

stack-up:
	$(COMPOSE) up -d --build

stack-down:
	$(COMPOSE) down

install:
	. .venv/bin/activate && DEPS_SCANNED=1 pip install -r requirements.txt

dev:
	. .venv/bin/activate && uvicorn src.api.main:app --reload --port 8000

worker:
	. .venv/bin/activate && celery -A src.worker.celery_app.celery_app worker --loglevel=INFO

frontend:
	. .venv/bin/activate && streamlit run frontend/streamlit_app.py --server.port 8501

load-samples:
	. .venv/bin/activate && python scripts/load_sample_data.py

test:
	. .venv/bin/activate && pytest tests/unit -q

test-int:
	. .venv/bin/activate && pytest tests/integration -q

lint:
	. .venv/bin/activate && black --check src frontend tests && flake8 src frontend tests && isort --check-only src frontend tests

format:
	. .venv/bin/activate && black src frontend tests && isort src frontend tests

audit:
	. .venv/bin/activate && pip-audit -r requirements.txt
