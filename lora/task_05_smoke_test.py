"""
task_05_smoke_test.py
---------------------
Task 5: Smoke Test 후 Production 승격

수행 내용:
  - MLflow Registry의 Staging 모델을 로드
  - SMOKE_TEST_PROMPTS 3~5건으로 추론 실행
  - 검사 항목:
      1. 응답 시간 < SMOKE_MAX_LATENCY_SEC (기본 30초)
      2. 출력 토큰 수 > 0 (빈 응답 아님)
      3. 예외 없이 정상 완료
  - 전체 통과 → Staging → Production 승격
  - 하나라도 실패 → Production 유지, 경고 출력

반환값 (dict):
  - status            : "success" | "failed"
  - promoted          : bool
  - test_results      : 각 프롬프트별 결과 list
  - decision          : 판정 사유 문자열
"""

import logging
import time

import torch

from helper import _abort, _run_task

log = logging.getLogger(__name__)


def _run_single_test(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    max_latency: float,
) -> dict:
    """단일 프롬프트에 대한 추론을 실행하고 결과를 반환합니다."""
    import time
    import torch

    result = {
        "prompt":           prompt,
        "passed":           False,
        "fail_reason":      None,
        "latency_sec":      0.0,
        "output_token_count": 0,
        "output_text":      "",
    }

    try:     
        device = "mps" if torch.mps.is_available() else "cpu"
        model.to(device)
        if tokenizer.pad_token is None:
           tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        start = time.time()
        with torch.no_grad():
            # 4. 생성 매개변수 최적화 (반복 방지 페널티 추가)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        latency = time.time() - start

        # 5. 결과 텍스트 디코딩
        generated_ids = outputs[0][input_len:]
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        output_tokens = len(generated_ids)
        
        # 출력 결과 로그 기록
        log.info("    Answer: %s", output_text)

        result["latency_sec"] = round(latency, 3)
        result["output_token_count"] = output_tokens
        result["output_text"] = output_text

        # 검사 1: 응답 시간 검증
        if latency > max_latency:
            result["fail_reason"] = f"응답 시간 초과: {latency:.2f}s > {max_latency}s"
            return result

        # 검사 2: 빈 응답 검증
        if output_tokens == 0 or not output_text.strip():
            result["fail_reason"] = "빈 응답: 출력 토큰 수 = 0 또는 텍스트 비어있음"
            return result

        result["passed"] = True

    except Exception as e:
        result["fail_reason"] = f"추론 중 에러 발생: {str(e)}"
        log.error("추론 에러: %s", str(e), exc_info=True)

    return result


def _load_model_from_mlflow(model_name: str, hf_token: str, model_id: str, stage: str = "Staging"):
    """
    MLflow Registry에서 Staging 단계의 모델과 토크나이저를 로드합니다.
    """
    import mlflow.transformers
    
    model_uri = f"models:/{model_name}/{stage}"  # 특정 버전 명시 (예시: 1, 2, 123, [Staging, Production] 등)
    log.info("MLflow Model Registry에서 모델 다운로드 중: %s", model_uri)

    components = mlflow.transformers.load_model(model_uri, return_type="components")
    tokenizer = components["tokenizer"]
    model = components["model"]

    log.info("MLflow 모델 및 토크나이저 로드 완료")
    return model, tokenizer


def _transition(client, model_name: str, version: str, stage: str) -> None:
    """MLflow 모델 버전의 Stage를 전환합니다."""
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=False,
    )
    log.info("  모델 stage 전환: %s v%s → %s", model_name, version, stage)



def run() -> dict:
    """
    Parameters
    ----------
    eval_result : Task 4 반환값
        promoted, new_version, model_name 포함
    """

    try:
        import mlflow
        from mlflow import MlflowClient
        from config import (
            MLFLOW_TRACKING_URI,
            MLFLOW_MODEL_NAME,
            SMOKE_TEST_PROMPTS,
            SMOKE_MAX_LATENCY_SEC,
            SMOKE_MAX_NEW_TOKENS,
            HF_TOKEN,
            MODEL_ID,
        )

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client      = MlflowClient()

        # ── 1. Staging 모델 확인 ─────────────────────────────────────
        model_name = MLFLOW_MODEL_NAME
        staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
        if not staging_versions:
            raise RuntimeError(
                "Staging 모델이 없습니다. Task 4가 정상 완료되었는지 확인하세요."
            )

        staging_version = staging_versions[0].version
        log.info("Smoke Test 대상: %s v%s (Staging)", MLFLOW_MODEL_NAME, staging_version)

        # ── 2. 모델 로드 ─────────────────────────────────────────────
        model, tokenizer = _load_model_from_mlflow(MLFLOW_MODEL_NAME, HF_TOKEN, MODEL_ID)

        # ── 3. 각 프롬프트 테스트 ────────────────────────────────────
        test_results = []
        all_passed   = True

        for idx, prompt in enumerate(SMOKE_TEST_PROMPTS):
            log.info("  [%d/%d] 테스트 중", idx + 1, len(SMOKE_TEST_PROMPTS))
            log.info("    Question: \"%s\"", prompt)
            result = _run_single_test(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=SMOKE_MAX_NEW_TOKENS,
                max_latency=SMOKE_MAX_LATENCY_SEC,
            )
            test_results.append(result)

            if result["passed"]:
                log.info(
                    "    PASS | latency=%.2fs | tokens=%d",
                    result["latency_sec"],
                    result["output_token_count"],
                )
            else:
                log.warning(
                    "    FAIL | reason=%s | latency=%.2fs",
                    result["fail_reason"],
                    result["latency_sec"],
                )
                all_passed = False

        # ── 4. 최종 판정 및 승격 ─────────────────────────────────────
        if all_passed:
            _transition(client, model_name, staging_version, "Production")
            decision = (
                f"전체 {len(SMOKE_TEST_PROMPTS)}건 테스트 통과 "
                f"→ v{staging_version} Production 승격"
            )
            log.info("%s", decision)
        else:
            failed_count = sum(1 for r in test_results if not r["passed"])
            decision = (
                f"{failed_count}/{len(SMOKE_TEST_PROMPTS)}건 테스트 실패 "
                f"→ Production 유지 (v{staging_version} Staging 대기)"
            )
            log.warning("%s", decision)

        _print_summary(test_results, all_passed, staging_version)

        return {
            "status":       "success",
            "promoted":     all_passed,
            "test_results": test_results,
            "decision":     decision,
        }

    except Exception as e:
        log.error("Task 5 실패: %s", e, exc_info=True)
        return {"status": "failed", "promoted": False, "error": str(e)}


def _print_summary(test_results: list, all_passed: bool, version: str) -> None:
    log.info("─" * 40)
    log.info("[ Smoke Test 결과 요약 ]")
    for i, r in enumerate(test_results):
        mark = "success" if r["passed"] else "failed"
        log.info(
            "  %s [%d] latency=%.2fs tokens=%d",
            mark, i + 1, r["latency_sec"], r["output_token_count"],
        )
        if not r["passed"]:
            log.warning("      실패 사유: %s", r["fail_reason"])
    log.info("─" * 40)
    if all_passed:
        log.info("  최종: v%s → Production 승격 완료", version)
    else:
        log.warning("  최종: Production 유지 (v%s Staging 대기)", version)
    log.info("─" * 40)


def main() -> None:
    results = _run_task(
        "Task 5: Smoke Test",
        run,
    )

    if results["status"] != "success":
        _abort("Task 5 실행 오류", results)

    if not results["promoted"]:
        _abort(
            f"Smoke Test 실패 → Production 미승격\n  판정: {results['decision']}",
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