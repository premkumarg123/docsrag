.PHONY: up down migrate seed dev test test-unit test-integration lint typecheck clean

# ── Docker ───────────────────────────────────────────────────────────────────

up:
	docker compose -f deploy/docker-compose.yml up -d

down:
	docker compose -f deploy/docker-compose.yml down

# ── Database ─────────────────────────────────────────────────────────────────

migrate:
	psql $$DATABASE_URL -f migrations/001_init.sql

seed:
	python scripts/seed_sample_docs.py

# ── Dev server ───────────────────────────────────────────────────────────────

dev:
	uvicorn api.main:app --reload --port 8000

# ── Tests ────────────────────────────────────────────────────────────────────

test: test-unit test-integration

test-unit:
	pytest tests/unit/ -v --cov=. --cov-report=term-missing

test-integration:
	pytest tests/integration/ -v --tb=short

# ── Code quality ─────────────────────────────────────────────────────────────

lint:
	ruff check .

lint-fix:
	ruff check . --fix

typecheck:
	mypy ingestion retrieval generation eval api pipeline.py --ignore-missing-imports

# ── Cleanup ──────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
