.PHONY: pull preview diff apply delete namespace help

export DEPLOY_ENV ?= dev
KUBECTL := kubectl
KUSTOMIZE ?= kustomize
GITOPS_OVERLAYS := kustomize/overlays/$(DEPLOY_ENV)

namespace: ## Create Kubernetes namespace 'mlops'
	$(KUBECTL) create namespace mlops || true

pull: ## Pull helm chart
	rm -rf kustomize/base/helm/mlflow
	helm repo add grafana https://grafana.github.io/helm-charts
	helm repo update
	helm pull grafana/grafana \
		--version 10.5.15 \
		--untar \
		--destination kustomize/base/helm

crds: ## Download Grafana CRDs
	@if [ -d kustomize/base/helm/crds ] && [ "$$(ls -A kustomize/base/helm/crds)" ]; then \
		echo "CRDs already exist, skipping download"; \
	else \
		mkdir -p kustomize/base/helm/crds && \
		curl -L https://github.com/grafana/grafana-operator/releases/download/v5.23.0/crds.yaml \
			-o kustomize/base/helm/crds/crds.yaml; \
	fi
	-$(KUBECTL) apply --server-side --force-conflicts -f kustomize/base/helm/crds/

delete: ## Delete workload
	-$(KUSTOMIZE) build --enable-alpha-plugins $(GITOPS_OVERLAYS) | $(KUBECTL) delete -f -

apply: namespace crds ## Deploy workload
	$(KUSTOMIZE) build --enable-alpha-plugins $(GITOPS_OVERLAYS) | $(KUBECTL) apply --force-conflicts --server-side=true -f -

preview: ## Preview generated manifests
	$(KUSTOMIZE) build --enable-alpha-plugins $(GITOPS_OVERLAYS)

diff: ## Show workload diff
	$(KUSTOMIZE) build --enable-alpha-plugins $(GITOPS_OVERLAYS) | $(KUBECTL) diff -f - || true

help: ## Show available targets and descriptions
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk -F ':.*?## ' '{printf "  %-20s %s\n", $$1, $$2}'