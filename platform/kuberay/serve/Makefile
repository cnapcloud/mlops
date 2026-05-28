IMAGE_ORG := cnapcloud
IMAGE_REPOSITORY := ${IMAGE_ORG}/ray-ollama-service
IMAGE_TAG := latest

.PHONY: help venv install docker-build docker-push clean

help: ## Show available make targets.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


venv: ## create virtual environment.
	@if [ ! -d ".venv" ]; then \
		echo "[venv] Creating Python virtual environment..."; \
		python3 -m venv .venv; \
	fi;
	@echo "[venv] To activate, run: source .venv/bin/activate"

install: ## Install dependencies.
	./.venv/bin/python3 -m pip install --upgrade pip
	./.venv/bin/python3 -m pip install --no-cache-dir -r requirements.txt

pull-model: ## Pull Ollama models.
	ollama pull qwen2.5-coder:1.5b-instruct

run-ollama: ## Run Ollama server.
	export OLLAMA_HOST=0.0.0.0:11434 && \
	export OLLAMA_NUM_PARALLEL=2 && \
	export OLLAMA_MAX_LOADED_MODELS=3 && \
	ollama serve

docker-build: ## [docker-build] Build the Docker image and clean up.
	docker build --no-cache --load -t $(IMAGE_REPOSITORY):$(IMAGE_TAG) .
	docker tag $(IMAGE_REPOSITORY):$(IMAGE_TAG) $(IMAGE_REPOSITORY):latest
	make clean

docker-push: ## Push the Docker image.
	docker push $(IMAGE_REPOSITORY):$(IMAGE_TAG)

clean: ## Clean up Docker builder cache.
	yes| docker builder prune
