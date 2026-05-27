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
    # Keeping the internal dataset formatting as is, since it maps to TRAINING_DATA structure, 
    # but updated the template labels to English if needed.
    return [
        f"Question {i}: {TRAINING_DATA[0]} "
        f"Answer {i}: {TRAINING_DATA[1]}"
        for i in range(n)
    ]


def _evaluate_model(model_uri: str, texts: list[str], hf_token: str, model_id: str) -> float:
    log.info("========= Starting MLflow Model Evaluation =========")
    log.info("  Model Artifact URI: %s", model_uri)

    try:
        import mlflow.transformers

        components = mlflow.transformers.load_model(
            model_uri=model_uri,
            return_type="components",
            local_files_only=True,
            torch_dtype=None,
        )
    except Exception as exc:
        log.error("Failed to restore MLflow model: %s", str(exc))
        raise

    import torch
    from transformers import DataCollatorForSeq2Seq

    device = get_device()
    model = components["model"]
    tokenizer = components["tokenizer"]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()  # Switch to evaluation mode to disable dropout, etc.
    model.to(device)
    log.info("  Target evaluation device assigned: device=%s", device)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)

    total_loss = 0.0
    count = 0
    log.info("  Calculating loss for %d evaluation data text samples...", len(texts))

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

    log.info("========= Model Evaluation Completed =========")
    log.info("  [Result] Avg Cross-Entropy Loss: %.4f", avg_loss)
    log.info("  [Result] Final Perplexity (PPL): %.4f", perplexity)
    return perplexity


def _set_alias(client, model_name: str, version: str, alias: str) -> None:
    client.set_registered_model_alias(name=model_name, alias=alias, version=version)
    log.info("  Model Alias Set: %s v%s → alias '%s'", model_name, version, alias)


def _resolve_model_uri(train_result: dict | None) -> tuple[str, str, str]:
    """Returns model_uri, model_version, and run_id. Prioritizes run_id, falls back to latest version."""
    run_id = (train_result or {}).get("run_id")
    
    if run_id:
        return f"runs:/{run_id}/model", (train_result or {}).get("model_version", ""), run_id
    
    latest = get_latest_version(MLFLOW_MODEL_NAME)
    if not latest:
        raise RuntimeError(f"Could not find model '{MLFLOW_MODEL_NAME}' in MLflow Registry.")

    return f"models:/{MLFLOW_MODEL_NAME}/{latest.version}", latest.version, latest.run_id


def _print_summary(new_ver, new_ppl, prev_ver, prev_ppl, promoted: bool) -> None:
    log.info("─" * 40)
    log.info("[ Evaluation Result Summary ]")
    log.info("  New Model v%s  Perplexity: %.4f", new_ver, new_ppl)
    log.info("  Current Model v%s  Perplexity: %.4f", prev_ver, prev_ppl)
    if promoted:
        log.info("  Decision: Promote new version to staging")
    else:
        log.warning("  Decision: Maintain current staging version")
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
        log.info("Evaluation dataset prepared: %d samples", len(eval_texts))

        log.info("Evaluating new model: run_id=%s, uri=%s", run_id, new_model_uri)
        new_perplexity = _evaluate_model(new_model_uri, eval_texts, HF_TOKEN, MODEL_ID)
        log.info("New model Perplexity: %.4f", new_perplexity)

        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("eval_perplexity", new_perplexity)

        try:
            staging_version = client.get_model_version_by_alias(model_name, "staging")
            has_staging = True
        except MlflowException:
            staging_version = None
            has_staging = False

        if not has_staging:
            log.info("No current Staging model found → Automatically promoting new version")
            _set_alias(client, model_name, new_version, "staging")
            decision = "Initial registration: No baseline to compare → Promoted to Staging"
            return {"status": "success", "promoted": True, "new_version": new_version, "new_perplexity": new_perplexity, "prev_perplexity": None, "decision": decision}

        prev_version = staging_version.version
        log.info("Evaluating current Staging model: version=%s", prev_version)
        prev_model_uri = f"models:/{model_name}/{prev_version}"
        prev_perplexity = _evaluate_model(prev_model_uri, eval_texts, HF_TOKEN, MODEL_ID)
        log.info("Current Staging Perplexity: %.4f", prev_perplexity)

        improvement = (prev_perplexity - new_perplexity) / prev_perplexity
        log.info("Improvement Rate: %.2f%% (Threshold: %.2f%%)", improvement * 100, EVAL_MIN_IMPROVEMENT_RATIO * 100)

        if improvement >= EVAL_MIN_IMPROVEMENT_RATIO:
            _set_alias(client, model_name, new_version, "staging")
            promoted = True
            decision = f"New version(v{new_version}) PPL {new_perplexity:.4f} < Current(v{prev_version}) PPL {prev_perplexity:.4f} (Improvement: {improvement * 100:.2f}%) → Promoted to Staging"
            log.info("Promotion approved: %s", decision)
        else:
            promoted = False
            decision = f"New version(v{new_version}) PPL {new_perplexity:.4f} Improvement {improvement * 100:.2f}% < Threshold {EVAL_MIN_IMPROVEMENT_RATIO * 100:.2f}% → Maintained current Staging(v{prev_version})"
            log.warning("Promotion rejected: %s", decision)

        _print_summary(new_version, new_perplexity, prev_version, prev_perplexity, promoted)
        return {"status": "success", "promoted": promoted, "new_version": new_version, "new_perplexity": new_perplexity, "prev_perplexity": prev_perplexity, "decision": decision}
    except Exception as exc:
        log.error("Task 4 failed: %s", exc, exc_info=True)
        return {"status": "failed", "promoted": False, "error": str(exc)}