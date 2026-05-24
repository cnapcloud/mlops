"""
task_04_evaluate.py
-------------------
Task 4: 신규 모델 평가 및 Staging 승격

수행 내용:
  - MLflow Registry에서 신규 버전(None 상태)과 현재 Staging 모델을 각각 로드
  - 평가 데이터셋(20건)으로 Loss / Perplexity 측정
  - 신규 모델이 EVAL_MIN_IMPROVEMENT_RATIO(기본 1%) 이상 개선되면 Staging 승격
  - 기존 Staging 모델은 Archived로 전환
  - 최초 실행 (비교 대상 없음) → 무조건 Staging 승격

반환값 (dict):
  - status            : "success" | "failed"
  - promoted          : bool  (Staging 승격 여부)
  - new_version       : 평가한 신규 모델 버전
  - new_perplexity    : 신규 모델 Perplexity
  - prev_perplexity   : 이전 Staging 모델 Perplexity (없으면 None)
  - decision          : 판정 사유 문자열
"""

import logging
import math

from helper import _abort, _run_task

log = logging.getLogger(__name__)


def run(train_result: dict) -> dict:
    """
    Parameters
    ----------
    train_result : Task 3 반환값
        run_id, model_version, model_name 포함
    """
    log.info("=" * 60)
    log.info("Task 4: Evaluate & Stage 비교 시작")
    log.info("=" * 60)

    try:
        import mlflow
        from mlflow import MlflowClient
        from config import (
            MLFLOW_TRACKING_URI,
            MLFLOW_MODEL_NAME,
            EVAL_SAMPLE_COUNT,
            EVAL_MIN_IMPROVEMENT_RATIO,
            HF_TOKEN,
            MODEL_ID,
        )

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        new_version  = str(train_result["model_version"])
        model_name   = train_result["model_name"]

        # ── 1. 평가 데이터셋 준비 ────────────────────────────────────
        eval_texts = _build_eval_dataset(EVAL_SAMPLE_COUNT)
        log.info("평가 데이터셋 준비: %d 건", len(eval_texts))

        # ── 2. 신규 모델 평가 ────────────────────────────────────────
        log.info("신규 모델 평가 중: version=%s", new_version)
        new_model_uri   = f"models:/{model_name}/{new_version}"
        new_perplexity  = _evaluate_model(new_model_uri, eval_texts, HF_TOKEN, MODEL_ID)
        log.info("신규 모델 Perplexity: %.4f", new_perplexity)

        # MLflow에 평가 메트릭 기록
        with mlflow.start_run(run_id=train_result["run_id"]):
            mlflow.log_metric("eval_perplexity", new_perplexity)

        # ── 3. 현재 Staging 모델 조회 ────────────────────────────────
        staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
        has_staging      = len(staging_versions) > 0

        if not has_staging:
            # ── 최초 실행: 비교 대상 없음 → 무조건 승격 ────────────
            log.info("현재 Staging 모델 없음 → 신규 버전 무조건 승격")
            _transition(client, model_name, new_version, "Staging")
            decision = "최초 등록: 비교 대상 없음 → Staging 승격"

            return {
                "status":           "success",
                "promoted":         True,
                "new_version":      new_version,
                "new_perplexity":   new_perplexity,
                "prev_perplexity":  None,
                "decision":         decision,
            }

        # ── 4. 기존 Staging 모델 평가 ────────────────────────────────
        prev_version    = staging_versions[0].version
        log.info("기존 Staging 모델 평가 중: version=%s", prev_version)
        prev_model_uri  = f"models:/{model_name}/{prev_version}"
        prev_perplexity = _evaluate_model(prev_model_uri, eval_texts, HF_TOKEN, MODEL_ID)
        log.info("기존 Staging Perplexity: %.4f", prev_perplexity)

        # ── 5. 성능 비교 판정 ────────────────────────────────────────
        improvement = (prev_perplexity - new_perplexity) / prev_perplexity
        log.info(
            "성능 개선율: %.2f%% (기준: %.2f%%)",
            improvement * 100,
            EVAL_MIN_IMPROVEMENT_RATIO * 100,
        )

        if improvement >= EVAL_MIN_IMPROVEMENT_RATIO:
            # 신규 모델이 더 좋음 → 승격 + 기존 Archived
            _transition(client, model_name, new_version,  "Staging")
            _transition(client, model_name, prev_version, "Archived")
            promoted = True
            decision = (
                f"신규 버전(v{new_version}) PPL {new_perplexity:.4f} < "
                f"기존(v{prev_version}) PPL {prev_perplexity:.4f} "
                f"(개선율 {improvement * 100:.2f}%) → Staging 승격"
            )
            log.info("승격 결정: %s", decision)
        else:
            # 신규 모델이 기준 미달 → 기존 Staging 유지
            promoted = False
            decision = (
                f"신규 버전(v{new_version}) PPL {new_perplexity:.4f} 개선율 "
                f"{improvement * 100:.2f}% < 기준 {EVAL_MIN_IMPROVEMENT_RATIO * 100:.2f}% "
                f"→ 기존 Staging(v{prev_version}) 유지"
            )
            log.warning("승격 거부: %s", decision)

        _print_summary(new_version, new_perplexity, prev_version, prev_perplexity, promoted)

        return {
            "status":           "success",
            "promoted":         promoted,
            "new_version":      new_version,
            "new_perplexity":   new_perplexity,
            "prev_perplexity":  prev_perplexity,
            "decision":         decision,
        }

    except Exception as e:
        log.error("Task 4 실패: %s", e, exc_info=True)
        return {"status": "failed", "promoted": False, "error": str(e)}


# ─────────────────────────────────────────────
# 내부 함수
# ─────────────────────────────────────────────

def _build_eval_dataset(n: int) -> list[str]:
    """평가용 텍스트 리스트 생성. 학습 데이터와 겹치지 않는 별도 문장 사용."""
    return [
        f"평가 질문 {i}: MLOps 파이프라인의 핵심 구성 요소를 설명하세요. "
        f"평가 답변 {i}: 데이터 준비, 학습, 평가, 배포 단계로 구성됩니다."
        for i in range(n)
    ]


def _evaluate_model(model_uri: str, texts: list[str], hf_token: str, model_id: str) -> float:
    """
    주어진 모델 URI의 모델을 로드하여 Perplexity를 계산합니다.

    실제 환경: mlflow.pyfunc.load_model(model_uri)로 MLflow 아티팩트 로드.
    데모 환경: HuggingFace 모델을 직접 로드하여 Loss 기반 Perplexity 계산.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("  모델 로드 중: %s", model_uri)

    # 데모: MLflow artifact 대신 HuggingFace 모델 직접 로드
    # 실제: model = mlflow.pyfunc.load_model(model_uri)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
    )
    model.eval()

    total_loss  = 0.0
    count       = 0

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            )
            labels          = enc["input_ids"].clone()
            enc["labels"]   = labels

            outputs     = model(**enc)
            total_loss  += outputs.loss.item()
            count       += 1

    avg_loss    = total_loss / count if count > 0 else float("inf")
    perplexity  = math.exp(avg_loss)
    return perplexity


def _transition(client, model_name: str, version: str, stage: str) -> None:
    """MLflow 모델 버전의 Stage를 전환합니다."""
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=False,
    )
    log.info("  모델 stage 전환: %s v%s → %s", model_name, version, stage)


def _print_summary(new_ver, new_ppl, prev_ver, prev_ppl, promoted: bool) -> None:
    log.info("─" * 40)
    log.info("[ 평가 결과 요약 ]")
    log.info("  신규 모델 v%s  Perplexity: %.4f", new_ver, new_ppl)
    log.info("  기존 모델 v%s  Perplexity: %.4f", prev_ver, prev_ppl)
    if promoted:
        log.info("  결정: 신규 버전 Staging 승격")
    else:
        log.warning("  결정: 기존 Staging 유지")
    log.info("─" * 40)


def main() -> None:
    results = _run_task(
        "Task 4: Evaluate",
        run,
        train_result={"model_name": "llm-finetune", "model_version": "9", "run_id": "ad63e5115b834a36ae2fa418ea9f0946"},
    )

    if results["status"] != "success":
        _abort("Task 4 평가 실패", results)

    if not results["promoted"]:
        _abort(
            f"신규 모델이 기준 미달 → Staging 미승격\n  판정: {results['decision']}",
            results,
        )

if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()