.PHONY: help install dev demo train test lint format clean up down logs build push deploy tf-init tf-plan tf-apply

PYTHON := python3
IMAGE_NAME := oil-forecaster
IMAGE_TAG := latest
REGISTRY := ghcr.io/your-username

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# --- Local dev ---
install:  ## Install runtime deps
	pip install -r requirements.txt

dev:  ## Install dev deps
	pip install -r requirements-dev.txt

demo:  ## Run the end-to-end demo (~2-3 min)
	$(PYTHON) scripts/quickstart_demo.py

train:  ## Run full training pipeline
	$(PYTHON) -m src.pipelines.train --horizon 5

run:  ## Run API locally (requires trained model in artifacts/)
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# --- Quality ---
test:  ## Run tests
	pytest -m "not slow and not integration" --cov=src --cov-report=term-missing

test-all:  ## Run all tests including slow
	pytest --cov=src

lint:  ## Lint code
	ruff check src tests
	black --check src tests

format:  ## Format code
	black src tests
	ruff check --fix src tests

clean:  ## Remove caches and artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov

# --- Docker Compose ---
up:  ## Start all services
	docker compose up -d
	@echo "✓ Services:"
	@echo "  API:        http://localhost:8000"
	@echo "  Dashboard:  http://localhost:8000/static/index.html"
	@echo "  MLflow:     http://localhost:5000"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana:    http://localhost:3000"

down:  ## Stop services
	docker compose down

logs:  ## Tail API logs
	docker compose logs -f api

train-docker:  ## Run training in a container
	docker compose --profile training run --rm training

# --- Images ---
build:  ## Build all images
	docker build -t $(IMAGE_NAME)-api:$(IMAGE_TAG) -f Dockerfile .
	docker build -t $(IMAGE_NAME)-trainer:$(IMAGE_TAG) -f Dockerfile.training .

push:  ## Push images
	docker tag $(IMAGE_NAME)-api:$(IMAGE_TAG) $(REGISTRY)/$(IMAGE_NAME)-api:$(IMAGE_TAG)
	docker tag $(IMAGE_NAME)-trainer:$(IMAGE_TAG) $(REGISTRY)/$(IMAGE_NAME)-trainer:$(IMAGE_TAG)
	docker push $(REGISTRY)/$(IMAGE_NAME)-api:$(IMAGE_TAG)
	docker push $(REGISTRY)/$(IMAGE_NAME)-trainer:$(IMAGE_TAG)

# --- Kubernetes ---
k8s-apply:  ## Apply all K8s manifests
	kubectl apply -f k8s/

k8s-status:  ## Show pod status
	kubectl get pods -n oil-forecaster

k8s-logs:  ## Tail API logs in cluster
	kubectl logs -f deployment/oil-forecaster-api -n oil-forecaster

k8s-train:  ## Trigger manual training job
	kubectl create job --from=cronjob/oil-forecaster-train manual-$$(date +%s) -n oil-forecaster

# --- Terraform ---
tf-init:
	cd terraform && terraform init

tf-plan:
	cd terraform && terraform plan

tf-apply:
	cd terraform && terraform apply

tf-destroy:
	cd terraform && terraform destroy
