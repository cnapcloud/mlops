#!/bin/bash

helm repo list | grep -q "apache-airflow" || helm repo add apache-airflow https://airflow.apache.org
helm repo update

# secret 생성
if ! kubectl -n mlops get secret airflow-api-secret-key &> /dev/null; then
  echo "airflow-api-secret-key secret not found. Creating it..."
  # Secret 생성 (key 이름 주의: "api-secret-key")
  kubectl -n mlops create secret generic airflow-api-secret-key \
    --from-literal=api-secret-key=$(python3 -c "import secrets; print(secrets.token_hex(32))")

  # JWT Secret 생성 (key 이름: "jwt-secret")
  kubectl -n mlops create secret generic airflow-jwt-secret \
    --from-literal=jwt-secret=$(python3 -c "import secrets; print(secrets.token_hex(32))")
else
  echo "airflow-api-secret-key secret already exists. Skipping creation."
fi

helm upgrade --install airflow apache-airflow/airflow \
  --namespace mlops \
  --create-namespace \
  --version 1.21.0 \
  --values values.yaml \

kubectl -n mlops apply -f airflow-ray-role.yaml