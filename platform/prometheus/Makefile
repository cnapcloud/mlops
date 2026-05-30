.PHONY: pull preview diff apply delete namespace crds help

export DEPLOY_ENV ?= dev
KUBECTL := kubectl
KUSTOMIZE ?= kustomize
GITOPS_OVERLAYS := kustomize/overlays/$(DEPLOY_ENV)

namespace: ## Create Kubernetes namespace 'mlops'
	$(KUBECTL) create namespace mlops || true

crds: ## Apply kube-prometheus-stack CRDs
	kubectl apply --server-side --force-conflicts \
		-f kustomize/base/helm/kube-prometheus-stack/charts/crds/crds/
		
pull: ## Pull helm chart
	rm -rf kustomize/base/helm/
	helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
	helm repo update
	helm pull prometheus-community/kube-prometheus-stack \
		--version 86.1.0 \
		--untar \
		--destination kustomize/base/helm

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