IMAGE_NAME := cnapcloud/nyc-mlops

.PHONY: docker-build docker-run docker-run-interactive

docker-build:
	docker build -t $(IMAGE_NAME):latest .

docker-run:
	docker run --rm \
		--add-host mlflow.cnapcloud.com:192.168.0.180 \
		$(IMAGE_NAME):latest

docker-push:
	docker push $(IMAGE_NAME):latest

docker-run-interactive:
	docker run --rm -it \
		--add-host mlflow.cnapcloud.com:192.168.0.180 \
		$(IMAGE_NAME):latest /bin/bash
