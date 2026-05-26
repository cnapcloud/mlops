.PHONY: pull preview apply delete help

export DEPLOY_ENV ?= dev
KUBECTL := kubectl
KUSTOMIZE ?= kustomize
GITOPS_OVERLAYS := kustomize/overlays/$(DEPLOY_ENV)

namespace:  ## Create Kubernetes namespace 'gateway'
	kubectl create namespace storage || true

pull: # Pull helm chart
	rm -rf kustomize/base/helm
	helm repo add minio https://charts.min.io/
	helm repo update
	helm pull minio/minio \
	    --version 5.4.0 \
		--untar \
		--destination kustomize/base/helm

delete: ## Delete workload
	-$(KUSTOMIZE) build --enable-alpha-plugins $(GITOPS_OVERLAYS) | $(KUBECTL) delete -f -

apply: ## Deploy workload
	$(KUSTOMIZE) build --enable-alpha-plugins $(GITOPS_OVERLAYS) | $(KUBECTL) apply --force-conflicts --server-side=true  -f -

preview: ## Preview generated manifests
	$(KUSTOMIZE) build --enable-alpha-plugins $(GITOPS_OVERLAYS)

help: ## Show available targets and descriptions
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk -F ':.*?## ' '{printf "  %-20s %s\n", $$1, $$2}'