.PHONY: help install dev test lint seed train docker-up docker-down clean

help:
	@echo "install     Install backend + frontend dependencies"
	@echo "dev         Run the FastAPI service with autoreload on :8000"
	@echo "dashboard   Run the Streamlit dashboard on :8501"
	@echo "test        Run the backend test-suite"
	@echo "lint        Run ruff over the backend"
	@echo "seed        Generate demo CSVs and load them into a running API"
	@echo "train       Train models for the demo tenant"
	@echo "docker-up   Build and start api + redis + dashboard"
	@echo "docker-down Stop the compose stack"

install:
	python -m pip install -r backend/requirements.txt -r backend/requirements-dev.txt
	python -m pip install -r frontend/requirements.txt

dev:
	cd backend && python -m uvicorn app.main:app --reload --port 8000

dashboard:
	streamlit run frontend/app.py

test:
	cd backend && python -m pytest -q

lint:
	cd backend && ruff check app tests

seed:
	python backend/scripts/generate_demo_data.py --months 18

train:
	cd backend && python scripts/train.py --api-key $(API_KEY)

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	rm -rf backend/data/app.db backend/data/models .pytest_cache
