
#!/bin/bash

helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm install kuberay kuberay/kuberay-operator  \
    --namespace mlops \
    --create-namespace \
    --version 1.6.1