.PHONY: pull preview diff apply delete namespace help

export DEPLOY_ENV ?= dev
KUBECTL := kubectl
KUSTOMIZE ?= kustomize
GITOPS_OVERLAYS := kustomize/overlays/$(DEPLOY_ENV)

namespace: ## Create Kubernetes namespace 'mlops'
	$(KUBECTL) create namespace mlops || true

pull: ## Pull helm chart
	rm -rf kustomize/base/helm/mlflow
	helm repo add community-charts https://community-charts.github.io/helm-charts
	helm repo update
	helm pull community-charts/mlflow \
		--version 1.8.1 \
		--untar \
		--destination kustomize/base/helm
	sed -i '' 's/{{- if and ( .Capabilities.APIVersions.Has "monitoring.coreos.com\/v1" ) ( .Values.serviceMonitor.enabled ) }}/{{- if .Values.serviceMonitor.enabled }}/g' kustomize/base/helm/mlflow/templates/servicemonitor.yaml

delete: ## Delete workload
	-$(KUSTOMIZE) build --enable-alpha-plugins $(GITOPS_OVERLAYS) | $(KUBECTL) delete -f -

apply: namespace ## Deploy workload
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