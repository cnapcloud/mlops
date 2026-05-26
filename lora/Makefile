IMAGE_NAME ?= lora-pipeline
IMAGE_TAG ?= latest
REGISTRY ?=
DOCKERFILE ?= Dockerfile
CONTEXT ?= .
VENV ?= .venv
PYTHON ?= python3
PIPELINE_STAGE ?= all

ifeq ($(strip $(REGISTRY)),)
IMAGE ?= $(IMAGE_NAME):$(IMAGE_TAG)
else
IMAGE ?= $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
endif

.PHONY: venv install run-pipeline build push build-push print-image

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev]"

run-pipeline: install
	$(VENV)/bin/python3 -m cli.run_pipeline $(PIPELINE_STAGE)

build:
	docker build -f $(DOCKERFILE) -t $(IMAGE) $(CONTEXT)

push:
	docker push $(IMAGE)

build-push: build push

print-image:
	@echo $(IMAGE)