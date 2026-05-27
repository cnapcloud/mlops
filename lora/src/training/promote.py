"""Production promotion entrypoint."""

from __future__ import annotations

import logging

from common.config import (
    HF_TOKEN,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_ID,
    SMOKE_MAX_LATENCY_SEC, 
    SMOKE_MAX_NEW_TOKENS,
    SMOKE_TEST_PROMPTS,
)
from common.device import get_device
from mlflow import MlflowException

log = logging.getLogger("training.promote")


def _run_single_test(model, tokenizer, prompt: str, max_new_tokens: int, max_latency: float) -> dict:
    import time

    import torch

    result = {
        "prompt": prompt,
        "passed": False,
        "fail_reason": None,
        "latency_sec": 0.0,
        "output_token_count": 0,
        "output_text": "",
    }

    try:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer(prompt, return_tensors="pt").to(get_device())
        input_len = inputs["input_ids"].shape[1]

        start = time.time()
        with torch.no_grad():
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

        generated_ids = outputs[0][input_len:]
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        output_tokens = len(generated_ids)

        log.info('    Answer: "%s"', output_text)

        result["latency_sec"] = round(latency, 3)
        result["output_token_count"] = output_tokens
        result["output_text"] = output_text

        if latency > max_latency:
            result["fail_reason"] = f"응답 시간 초과: {latency:.2f}s > {max_latency}s"
            return result
        if output_tokens == 0 or not output_text.strip():
            result["fail_reason"] = "빈 응답: 출력 토큰 수 = 0 또는 텍스트 비어있음"
            return result

        result["passed"] = True
    except Exception as exc:
        result["fail_reason"] = f"추론 중 에러 발생: {str(exc)}"
        log.error("추론 에러: %s", str(exc), exc_info=True)

    return result


def _load_model_from_mlflow(model_uri: str):
    import mlflow.transformers

    log.info("MLflow Model Registry에서 모델 다운로드 중: %s", model_uri)
    device = get_device()
    components = mlflow.transformers.load_model(model_uri, return_type="components")
    tokenizer = components["tokenizer"]
    model = components["model"]
    model.to(device)
    model.eval()
    
    log.info("MLflow 모델 및 토크나이저 로드 완료")
    return model, tokenizer


def _set_alias(client, model_name: str, version: str, alias: str) -> None:
    client.set_registered_model_alias(name=model_name, alias=alias, version=version)
    log.info("  모델 Alias 설정: %s v%s → alias '%s'", model_name, version, alias)


def _print_summary(test_results: list, all_passed: bool, version: str) -> None:
    log.info("─" * 40)
    log.info("[ Smoke Test 결과 요약 ]")
    for i, r in enumerate(test_results):
        mark = "success" if r["passed"] else "failed"
        log.info("  %s [%d] latency=%.2fs tokens=%d", mark, i + 1, r["latency_sec"], r["output_token_count"])
        if not r["passed"]:
            log.warning("      실패 사유: %s", r["fail_reason"])
    log.info("─" * 40)
    if all_passed:
        log.info("  최종: v%s → Production 승격 완료", version)
    else:
        log.warning("  최종: Production 유지 (v%s Staging 대기)", version)
    log.info("─" * 40)


def run() -> dict:
    try:
        import mlflow
        from mlflow import MlflowClient

        model_name = MLFLOW_MODEL_NAME
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        try:
            staging_mv = client.get_model_version_by_alias(model_name, "staging")
        except MlflowException:
            raise RuntimeError("Staging 모델이 없습니다. Task 4가 정상 완료되었는지 확인하세요.")

        staging_version = staging_mv.version
        log.info("Smoke Test 대상: %s v%s (staging)", model_name, staging_version)

        model, tokenizer = _load_model_from_mlflow(f"models:/{model_name}@staging")
        test_results = []
        all_passed = True
        for idx, prompt in enumerate(SMOKE_TEST_PROMPTS):
            log.info("  [%d/%d] 테스트 중", idx + 1, len(SMOKE_TEST_PROMPTS))
            log.info('    Question: "%s"', prompt)
            result = _run_single_test(model=model, tokenizer=tokenizer, prompt=prompt, max_new_tokens=SMOKE_MAX_NEW_TOKENS, max_latency=SMOKE_MAX_LATENCY_SEC)
            test_results.append(result)
            if result["passed"]:
                log.info("    PASS | latency=%.2fs | tokens=%d", result["latency_sec"], result["output_token_count"])
            else:
                log.warning("    FAIL | reason=%s | latency=%.2fs", result["fail_reason"], result["latency_sec"])
                all_passed = False

        if all_passed:
            _set_alias(client, model_name, staging_version, "Production")
            decision = f"전체 {len(SMOKE_TEST_PROMPTS)}건 테스트 통과 → v{staging_version} Production 승격"
            log.info("%s", decision)
        else:
            failed_count = sum(1 for r in test_results if not r["passed"])
            decision = f"{failed_count}/{len(SMOKE_TEST_PROMPTS)}건 테스트 실패 → Production 유지 (v{staging_version} Staging 대기)"
            log.warning("%s", decision)

        _print_summary(test_results, all_passed, staging_version)
        return {"status": "success", "promoted": all_passed, "test_results": test_results, "decision": decision}
    except Exception as exc:
        log.error("Task 5 실패: %s", exc, exc_info=True)
        return {"status": "failed", "promoted": False, "error": str(exc)}
