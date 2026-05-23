#!/bin/bash

helm repo add community-charts https://community-charts.github.io/helm-charts
helm repo update

# mlflow pvc 생성
if ! kubectl get pvc mlflow-storage-pvc -n mlops &> /dev/null; then
  echo "mlflow-pvc not found. Creating it..."
  kubectl create namespace mlops 2>/dev/null || true
  kubectl apply -f mlflow-pvc.yaml
  kubectl apply -f postgresql-pvc.yaml
else
  echo "mlflow-storage-pvc already exists. Skipping creation."
fi

helm upgrade --install mlflow community-charts/mlflow \
  --version 1.8.1 \
  --namespace mlops \
  --create-namespace \
  -f values.yaml