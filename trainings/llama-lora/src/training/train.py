"""Training domain entrypoint."""

from __future__ import annotations

import json
import logging
import os

from common.config import (
    HF_HOME,
    HF_TOKEN,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_R,
    MAX_SEQ_LEN,
    MLFLOW_EXPERIMENT,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_ID,
    RAY_ADDRESS,
    RAY_NUM_WORKERS,
    RAY_STORAGE,
    TRAIN_BATCH,
    TRAIN_EPOCHS,
    USE_GPU,
    RAY_WORKER_CPUS,
    RAY_WORKER_GPUS,
)
from common.device import get_device
from common.data import ensure_eos_suffix, load_validated_minio_data

log = logging.getLogger("training.train")


def _build_dataset(model_id: str, hf_token: str):
    """Construct Ray Data dataset (executed on driver)."""
    import ray
    from datasets import Dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    validated_texts = load_validated_minio_data()
    training_texts = [ensure_eos_suffix(text, tokenizer.eos_token) for text in validated_texts]

    # `Dataset.from_dict` expects a mapping of column name -> list, not a bare list.
    hf_dataset = Dataset.from_dict({"text": training_texts})
    return ray.data.from_huggingface(hf_dataset)


def _train_func_per_worker(config: dict) -> None:
    """Training function executed on each KubeRay worker Pod."""
    import json
    import logging

    import mlflow
    import mlflow.transformers
    import torch
    from datasets import Dataset as HFDataset
    from peft import LoraConfig, get_peft_model
    from ray import train as rt
    from ray.train import Checkpoint
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    logger = logging.getLogger("training.train.worker")

    epochs = config["epochs"]
    model_id = config["model_id"]
    mlflow_cfg = config["mlflow_cfg"]

    world_size = rt.get_context().get_world_size()
    world_rank = rt.get_context().get_world_rank()
    logger.info("[Rank %d/%d] Worker started", world_rank, world_size)

    logger.info("[Rank %d] Loading model: %s", world_rank, model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token

    device = get_device()
    torch_dtype = torch.bfloat16 if device == "mps" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    model.to(device)

    peft_config = LoraConfig(
        r=mlflow_cfg["params"]["lora_r"],
        lora_alpha=mlflow_cfg["params"]["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.gradient_checkpointing_enable()

    logger.info(
        "[Rank %d] Model loading completed (trainable params: %d)",
        world_rank,
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=mlflow_cfg["params"]["learning_rate"])

    local_shard = rt.get_dataset_shard("train_set")
    rows = []
    for batch in local_shard.iter_batches(batch_size=256):
        if "text" in batch:
            for item in batch["text"]:
                if isinstance(item, dict):
                    text_str = item.get("text", "")
                else:
                    text_str = str(item)

                text_str = text_str.strip()
                if not text_str.endswith(tokenizer.eos_token):
                    text_str += tokenizer.eos_token

                rows.append({"text": text_str})

    if not rows:
        raise RuntimeError(f"[Rank {world_rank}] No valid text data parsed from the training shard.")

    hf_dataset = HFDataset.from_list(rows)

    def tokenize_fn(examples):
        model_inputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=mlflow_cfg["params"]["max_seq_len"],
        )
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    tokenized = hf_dataset.map(tokenize_fn, batched=True, remove_columns=hf_dataset.column_names)
    split = tokenized.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)

    output_dir = f"./lora_out/{model_id.replace('/', '_')}/train"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=mlflow_cfg["params"]["batch_size"],
        per_device_eval_batch_size=mlflow_cfg["params"]["batch_size"],
        num_train_epochs=epochs,
        learning_rate=mlflow_cfg["params"]["learning_rate"],
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="steps",
        eval_steps=3,
        bf16=False,
        fp16=(device.type == "cuda"),
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        dataloader_pin_memory=(device.type == "cuda"),
        push_to_hub=False,
        use_cpu=(device.type == "cpu"),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        optimizers=(optimizer, None),
    )

    train_result = trainer.train()
    last_loss = train_result.training_loss
    logger.info("[Rank %d] Full training pipeline completed (Final loss: %.6f)", world_rank, last_loss)

    run_id = None
    if world_rank != 0:
        logger.info("[Rank %d] Skipping MLflow saving since this is not Rank 0.", world_rank)
        rt.report({"loss": last_loss, "epoch": epochs - 1})
        return

    logger.info("[Rank 0] Starting MLflow logging integration (Tracking URI: %s)", mlflow_cfg["tracking_uri"])

    try:
        mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
        mlflow.set_experiment(mlflow_cfg["experiment"])

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info("[Rank 0] MLflow Run successfully created (Run ID: %s)", run_id)

            mlflow.log_params(mlflow_cfg["params"])
            mlflow.set_tags(mlflow_cfg["tags"])
            mlflow.log_metric("final_loss", last_loss)

            raw_model = model.module if hasattr(model, "module") else model
            components = {"model": raw_model, "tokenizer": tokenizer}

            logger.info("[Rank 0] Registering model to MLflow (Model Name: %s)...", mlflow_cfg["registered_model_name"])
            mlflow.transformers.log_model(
                transformers_model=components,
                task="text-generation",
                name="model",
                registered_model_name=mlflow_cfg["registered_model_name"],
            )
            logger.info("[Rank 0] MLflow model registration completed")

    except Exception as exc:
        logger.error("[Rank 0] Error occurred during MLflow saving: %s", str(exc), exc_info=True)
        raise

    checkpoint_dir = os.path.join(output_dir, f"checkpoint_run_{run_id or 'unknown'}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {"run_id": run_id, "loss": last_loss, "epoch": epochs - 1, "model_id": model_id},
            handle,
            ensure_ascii=False,
            indent=2,
        )

    rt.report(
        {"loss": last_loss, "epoch": epochs - 1, "mlflow_run_id": run_id},
        checkpoint=Checkpoint.from_directory(checkpoint_dir),
    )
    logger.info("[Rank 0] Final result report completed and worker process terminated normally")


def _run_ray_training(
    hf_token: str,
    ray_address: str,
    storage_path: str,
    num_workers: int,
    use_gpu: bool,
    epochs: int,
    model_id: str,
    hf_home: str,
    mlflow_cfg: dict,
) -> dict:
    import ray
    from ray.train import RunConfig, ScalingConfig
    from ray.train.torch import TorchConfig, TorchTrainer

    runtime_env = {
        "working_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        "pip": ["transformers", "peft", "datasets", "torch", "mlflow", "torchvision"],
        "env_vars": {
            "HF_TOKEN": hf_token,
            "HF_HOME": hf_home,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING": "1",
            "RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION": "0.5",
        },
    }

    log.info("Connecting to Ray Cluster: %s", ray_address)
    ray.init(address=ray_address, ignore_reinit_error=True, runtime_env=runtime_env)

    try:
        ray_dataset = _build_dataset(model_id=model_id, hf_token=hf_token)
        resources_per_worker = {"GPU": RAY_WORKER_GPUS} if use_gpu else {"CPU": RAY_WORKER_CPUS}
        scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu, resources_per_worker=resources_per_worker)

        trainer = TorchTrainer(
            train_loop_per_worker=_train_func_per_worker,
            train_loop_config={"epochs": epochs, "model_id": model_id, "mlflow_cfg": mlflow_cfg},
            scaling_config=scaling_config,
            torch_config=TorchConfig(backend="nccl" if use_gpu else "gloo"),
            datasets={"train_set": ray_dataset},
            run_config=RunConfig(storage_path=storage_path),
        )

        log.info("Starting KubeRay distributed training (workers=%d, gpu=%s, epochs=%d)", num_workers, use_gpu, epochs)
        result = trainer.fit()
        log.info("KubeRay distributed training completed")
        return result.metrics or {}
    finally:
        ray.shutdown()
        log.info("Ray connection closed")


def run() -> dict:
    try:
        import mlflow
        from mlflow import MlflowClient

        if not HF_TOKEN:
            raise EnvironmentError("HF_TOKEN environment variable is not set.")

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        mlflow_cfg = {
            "tracking_uri": MLFLOW_TRACKING_URI,
            "experiment": MLFLOW_EXPERIMENT,
            "registered_model_name": MLFLOW_MODEL_NAME,
            "params": {
                "model_id": MODEL_ID,
                "epochs": TRAIN_EPOCHS,
                "num_workers": RAY_NUM_WORKERS,
                "use_gpu": USE_GPU,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "learning_rate": LEARNING_RATE,
                "max_seq_len": MAX_SEQ_LEN,
                "batch_size": TRAIN_BATCH,
            },
            "tags": {"pipeline": "mlops-demo", "task": "llm-finetune", "framework": "transformers+peft"},
        }

        metrics = _run_ray_training(
            hf_token=HF_TOKEN,
            ray_address=RAY_ADDRESS,
            storage_path=RAY_STORAGE,
            num_workers=RAY_NUM_WORKERS,
            use_gpu=USE_GPU,
            epochs=TRAIN_EPOCHS,
            model_id=MODEL_ID,
            hf_home=HF_HOME,
            mlflow_cfg=mlflow_cfg,
        )

        run_id = metrics.get("mlflow_run_id")
        if not run_id:
            raise RuntimeError("Failed to receive mlflow_run_id from Rank 0 worker. Please check the worker logs.")
        log.info("Received run_id from Rank 0 worker: %s", run_id)

        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["None"])
        if not versions:
            raise RuntimeError(f"Could not find model '{MLFLOW_MODEL_NAME}' in MLflow Registry.")

        model_version = next((v.version for v in versions if v.run_id == run_id), versions[0].version)
        log.info("Verified MLflow model version: name=%s, version=%s", MLFLOW_MODEL_NAME, model_version)

        return {
            "status": "success",
            "run_id": run_id,
            "model_version": model_version,
            "model_name": MLFLOW_MODEL_NAME,
            "metrics": metrics,
        }
    except Exception as exc:
        log.error("Task 3 failed: %s", exc, exc_info=True)
        return {"status": "failed", "error": str(exc)}