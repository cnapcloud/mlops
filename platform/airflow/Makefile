.PHONY: pull preview diff apply delete namespace ensure-secrets help

export DEPLOY_ENV ?= dev
KUBECTL := kubectl
KUSTOMIZE ?= kustomize
GITOPS_OVERLAYS := kustomize/overlays/$(DEPLOY_ENV)
SECRETS_DIR := kustomize/base/resources/.generated
API_SECRET_FILE := $(SECRETS_DIR)/airflow-api-secret-key
JWT_SECRET_FILE := $(SECRETS_DIR)/airflow-jwt-secret

namespace: ## Create Kubernetes namespace 'mlops'
	$(KUBECTL) create namespace mlops || true

ensure-secrets: ## Create local secret material for Kustomize generators
	@mkdir -p $(SECRETS_DIR)
	@if [ ! -f "$(API_SECRET_FILE)" ]; then \
		python3 -c 'import secrets; print(secrets.token_hex(32))' > "$(API_SECRET_FILE)"; \
	fi
	@if [ ! -f "$(JWT_SECRET_FILE)" ]; then \
		python3 -c 'import secrets; print(secrets.token_hex(64))' > "$(JWT_SECRET_FILE)"; \
	fi

pull: ## Pull helm chart
	rm -rf kustomize/base/helm/airflow
	helm repo add apache-airflow https://airflow.apache.org >/dev/null 2>&1 || true
	helm repo update
	helm pull apache-airflow/airflow \
		--version 1.21.0 \
		--untar \
		--destination kustomize/base/helm

delete: ensure-secrets ## Delete workload
	-$(KUSTOMIZE) build --enable-alpha-plugins $(GITOPS_OVERLAYS) | $(KUBECTL) delete -f -

apply: namespace ensure-secrets ## Deploy workload
	$(KUSTOMIZE) build --enable-alpha-plugins $(GITOPS_OVERLAYS) | $(KUBECTL) apply --force-conflicts --server-side=true -f -

preview: ensure-secrets ## Preview generated manifests
	$(KUSTOMIZE) build --enable-alpha-plugins $(GITOPS_OVERLAYS)

diff: ensure-secrets ## Show workload diff
	$(KUSTOMIZE) build --enable-alpha-plugins $(GITOPS_OVERLAYS) | $(KUBECTL) diff -f - || true

help: ## Show available targets and descriptions
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk -F ':.*?## ' '{printf "  %-20s %s\n", $$1, $$2}'