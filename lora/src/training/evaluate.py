"""Evaluation domain entrypoint."""

from __future__ import annotations

import logging
import math

from common.config import (
    EVAL_MIN_IMPROVEMENT_RATIO,
    EVAL_SAMPLE_COUNT,
    HF_TOKEN,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_ID,
    TRAINING_DATA,
)
from common.device import get_device
from common.mlflow_utils import get_latest_version
from mlflow import MlflowException

log = logging.getLogger("training.evaluate")


def _build_eval_dataset(n: int) -> list[str]:
    return [
        f"질문 {i}: {TRAINING_DATA[0]} "
        f"답변 {i}: {TRAINING_DATA[1]}"
        for i in range(n)
    ]


def _evaluate_model(model_uri: str, texts: list[str], hf_token: str, model_id: str) -> float:
    log.info("========= MLflow 모델 평가 시작 =========")
    log.info("  모델 아티팩트 URI: %s", model_uri)

    try:
        import mlflow.transformers

        components = mlflow.transformers.load_model(
            model_uri=model_uri,
            return_type="components",
            local_files_only=True,
            torch_dtype=None,
        )
    except Exception as exc:
        log.error("MLflow 모델 복원 실패: %s", str(exc))
        raise

    import torch
    from transformers import DataCollatorForSeq2Seq

    device = get_device()
    model = components["model"]
    tokenizer = components["tokenizer"]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval() # 평가 모드로 전환하여 드롭아웃 등 비활성화
    model.to(device)
    log.info("  평가 디바이스 지정 완료: device=%s",device)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)

    total_loss = 0.0
    count = 0
    log.info("  총 %d개의 평가 데이터 텍스트 계산 중...", len(texts))

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, truncation=True, max_length=256)
            enc["labels"] = enc["input_ids"].copy()
            batch = data_collator([enc])
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            count += 1

    avg_loss = total_loss / count if count > 0 else float("inf")
    perplexity = math.exp(avg_loss)

    log.info("========= 모델 평가 완료 =========")
    log.info("  [결과] Avg Cross-Entropy Loss: %.4f", avg_loss)
    log.info("  [결과] 최종 Perplexity (PPL): %.4f", perplexity)
    return perplexity


def _set_alias(client, model_name: str, version: str, alias: str) -> None:
    client.set_registered_model_alias(name=model_name, alias=alias, version=version)
    log.info("  모델 Alias 설정: %s v%s → alias '%s'", model_name, version, alias)


def _resolve_model_uri(train_result: dict | None) -> tuple[str, str]:
    """model_uri와 model_version을 반환. run_id 우선, 없으면 최신 버전 사용."""
    run_id = (train_result or {}).get("run_id")
    
    if run_id:
        return f"runs:/{run_id}/model", (train_result or {}).get("model_version", ""), run_id
    
    latest = get_latest_version(MLFLOW_MODEL_NAME)
    if not latest:
        raise RuntimeError(f"MLflow Registry에서 '{MLFLOW_MODEL_NAME}' 모델을 찾을 수 없습니다.")

    return f"models:/{MLFLOW_MODEL_NAME}/{latest.version}", latest.version, latest.run_id


def _print_summary(new_ver, new_ppl, prev_ver, prev_ppl, promoted: bool) -> None:
    log.info("─" * 40)
    log.info("[ 평가 결과 요약 ]")
    log.info("  신규 모델 v%s  Perplexity: %.4f", new_ver, new_ppl)
    log.info("  기존 모델 v%s  Perplexity: %.4f", prev_ver, prev_ppl)
    if promoted:
        log.info("  결정: 신규 버전 staging 승격")
    else:
        log.warning("  결정: 기존 staging 유지")
    log.info("─" * 40)


def run(train_result: dict | None = None) -> dict:
    try:
        import mlflow
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        
        model_name = MLFLOW_MODEL_NAME
        new_model_uri, new_version, run_id = _resolve_model_uri(train_result)

        eval_texts = _build_eval_dataset(EVAL_SAMPLE_COUNT)
        log.info("평가 데이터셋 준비: %d 건", len(eval_texts))

        log.info("신규 모델 평가 중: run_id=%s, uri=%s", run_id, new_model_uri)
        new_perplexity = _evaluate_model(new_model_uri, eval_texts, HF_TOKEN, MODEL_ID)
        log.info("신규 모델 Perplexity: %.4f", new_perplexity)

        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("eval_perplexity", new_perplexity)

        try:
            staging_version = client.get_model_version_by_alias(model_name, "staging")
            has_staging = True
        except MlflowException:
            staging_version = None
            has_staging = False

        if not has_staging:
            log.info("현재 Staging 모델 없음 → 신규 버전 무조건 승격")
            _set_alias(client, model_name, new_version, "staging")
            decision = "최초 등록: 비교 대상 없음 → Staging 승격"
            return {"status": "success", "promoted": True, "new_version": new_version, "new_perplexity": new_perplexity, "prev_perplexity": None, "decision": decision}

        prev_version = staging_version.version
        log.info("기존 Staging 모델 평가 중: version=%s", prev_version)
        prev_model_uri = f"models:/{model_name}/{prev_version}"
        prev_perplexity = _evaluate_model(prev_model_uri, eval_texts, HF_TOKEN, MODEL_ID)
        log.info("기존 Staging Perplexity: %.4f", prev_perplexity)

        improvement = (prev_perplexity - new_perplexity) / prev_perplexity
        log.info("성능 개선율: %.2f%% (기준: %.2f%%)", improvement * 100, EVAL_MIN_IMPROVEMENT_RATIO * 100)

        if improvement >= EVAL_MIN_IMPROVEMENT_RATIO:
            _set_alias(client, model_name, new_version, "staging")
            promoted = True
            decision = f"신규 버전(v{new_version}) PPL {new_perplexity:.4f} < 기존(v{prev_version}) PPL {prev_perplexity:.4f} (개선율 {improvement * 100:.2f}%) → staging 승격"
            log.info("승격 결정: %s", decision)
        else:
            promoted = False
            decision = f"신규 버전(v{new_version}) PPL {new_perplexity:.4f} 개선율 {improvement * 100:.2f}% < 기준 {EVAL_MIN_IMPROVEMENT_RATIO * 100:.2f}% → 기존 staging(v{prev_version}) 유지"
            log.warning("승격 거부: %s", decision)

        _print_summary(new_version, new_perplexity, prev_version, prev_perplexity, promoted)
        return {"status": "success", "promoted": promoted, "new_version": new_version, "new_perplexity": new_perplexity, "prev_perplexity": prev_perplexity, "decision": decision}
    except Exception as exc:
        log.error("Task 4 실패: %s", exc, exc_info=True)
        return {"status": "failed", "promoted": False, "error": str(exc)}
