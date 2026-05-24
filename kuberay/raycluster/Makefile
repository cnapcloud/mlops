IMAGE_ORG := cnapcloud
IMAGE_REPOSITORY := ${IMAGE_ORG}/ray-llm
IMAGE_TAG := 2.55.1-py312-cpu-aarch64

.PHONY: docker-build docker-push docker-run-interactive prune

docker-build:
	docker build -t $(IMAGE_REPOSITORY):$(IMAGE_TAG) .
	$(MAKE) prune

docker-push:
	docker push $(IMAGE_REPOSITORY):$(IMAGE_TAG)

docker-run-interactive:
	docker run --rm -it \
		--add-host mlflow.cnapcloud.com:192.168.0.180 \
		$(IMAGE_REPOSITORY):$(IMAGE_TAG) /bin/bash

prune:
	yes| docker builder prune
