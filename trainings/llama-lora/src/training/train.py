"""Training domain entrypoint."""

from __future__ import annotations

import logging
import os

from common import config
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
    import os
    import tempfile  # 로컬 고속 임시 디렉토리 활용을 위해 추가

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

    # 디바이스 객체 할당 및 통일
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if device.type == "mps" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
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

    # 공통 제어 변수 선언
    run_id = None
    model_version = None
    checkpoint_dir = None

    # 🛠️ 개선: 각 Pod의 로컬 고속 임시 디렉토리를 사용하여 파일 충돌 및 용량 낭비 방지
    with tempfile.TemporaryDirectory() as local_temp_dir:
        training_args = TrainingArguments(
            output_dir=local_temp_dir,  # 로컬 저장소 활용
            per_device_train_batch_size=mlflow_cfg["params"]["batch_size"],
            per_device_eval_batch_size=mlflow_cfg["params"]["batch_size"],
            num_train_epochs=epochs,
            learning_rate=mlflow_cfg["params"]["learning_rate"],
            logging_steps=1,
            save_strategy="no",        # 공유 볼륨 폭발 방지를 위해 HF의 매 에폭 저장은 끕니다.
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

        # 모든 워커가 흐름을 끝까지 유지하도록 Early Return 제거
        if world_rank == 0:
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
                    model_info = mlflow.transformers.log_model(
                        transformers_model=components,
                        task="text-generation",
                        name="model",
                        registered_model_name=mlflow_cfg["registered_model_name"],
                    )
                    model_version = model_info.registered_model_version
                    logger.info("[Rank 0] MLflow model registration completed")

                # Rank 0만 공유 저장소에 최종 수집용 메타데이터 폴더 생성
                storage_path = config["storage_path"]
                checkpoint_dir = os.path.join(storage_path, f"checkpoint_run_{run_id or 'unknown'}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # 필요 시 최종 완성된 모델 가중치도 여기에 함께 백업합니다.
                trainer.save_model(checkpoint_dir)

                with open(os.path.join(checkpoint_dir, "metadata.json"), "w", encoding="utf-8") as handle:
                    json.dump(
                        {"run_id": run_id, "model_version": model_version, "loss": last_loss, "epoch": epochs - 1, "model_id": model_id},
                        handle, ensure_ascii=False, indent=2,
                    )

            except Exception as exc:
                logger.error("[Rank 0] Error occurred during MLflow saving: %s", str(exc), exc_info=True)
                raise
        else:
            logger.info("[Rank %d] Waiting for Rank 0 to complete MLflow and storage operations.", world_rank)

        # 모든 워커가 다 함께 같은 라인에서 rt.report()를 호출해 동기화합니다.(모든 워커의 작업 동기화)
        # Rank 0만 유효한 디렉토리 객체를 전달하고, 나머지 워커는 None을 보내 레이가 중복 저장을 방지하게 합니다.
        ray_checkpoint = Checkpoint.from_directory(checkpoint_dir) if checkpoint_dir else None

        rt.report(
            {
                "loss": last_loss, 
                "epoch": epochs - 1, 
                "mlflow_run_id": run_id, 
                "mlflow_model_version": model_version
            },
            checkpoint=ray_checkpoint,
        )
        
    logger.info("[Rank %d] Final result report completed and worker process terminated normally", world_rank)


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
            "MLFLOW_HTTP_REQUEST_MAX_RETRIES": str(config.MLFLOW_HTTP_REQUEST_MAX_RETRIES),
        },
    }

    log.info("Connecting to Ray Cluster: %s", ray_address)
    ray.init(address=ray_address, ignore_reinit_error=True, runtime_env=runtime_env)

    try:
        ray_dataset = _build_dataset(model_id=model_id, hf_token=hf_token)
        resources_per_worker = {"GPU": config.RAY_WORKER_GPUS} if use_gpu else {"CPU": config.RAY_WORKER_CPUS}
        scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu, resources_per_worker=resources_per_worker)

        trainer = TorchTrainer(
            train_loop_per_worker=_train_func_per_worker,
            train_loop_config={"epochs": epochs, "model_id": model_id, "mlflow_cfg": mlflow_cfg, "storage_path": storage_path},
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
        if not config.HF_TOKEN:
            raise EnvironmentError("HF_TOKEN environment variable is not set.")

        mlflow_cfg = {
            "tracking_uri": config.MLFLOW_TRACKING_URI,
            "experiment": config.MLFLOW_EXPERIMENT,
            "registered_model_name": config.MLFLOW_MODEL_NAME,
            "params": {
                "model_id": config.MODEL_ID,
                "epochs": config.TRAIN_EPOCHS,
                "num_workers": config.RAY_NUM_WORKERS,
                "use_gpu": config.USE_GPU,
                "lora_r": config.LORA_R,
                "lora_alpha": config.LORA_ALPHA,
                "learning_rate": config.LEARNING_RATE,
                "max_seq_len": config.MAX_SEQ_LEN,
                "batch_size": config.TRAIN_BATCH,
            },
            "tags": {"pipeline": "mlops-demo", "task": "llm-finetune", "framework": "transformers+peft"},
        }

        metrics = _run_ray_training(
            hf_token=config.HF_TOKEN,
            ray_address=config.RAY_ADDRESS,
            storage_path=config.RAY_STORAGE,
            num_workers=config.RAY_NUM_WORKERS,
            use_gpu=config.USE_GPU,
            epochs=config.TRAIN_EPOCHS,
            model_id=config.MODEL_ID,
            hf_home=config.HF_HOME,
            mlflow_cfg=mlflow_cfg,
        )

        run_id = metrics.get("mlflow_run_id")
        model_version = metrics.get("mlflow_model_version")
        if not run_id:
            raise RuntimeError("Failed to receive mlflow_run_id from Rank 0 worker. Please check the worker logs.")
        log.info("Received run_id from Rank 0 worker: %s", run_id)

        log.info("Verified MLflow model version: name=%s, version=%s", config.MLFLOW_MODEL_NAME, model_version)

        return {
            "status": "success",
            "run_id": run_id,
            "model_version": model_version,
            "model_name": config.MLFLOW_MODEL_NAME,
            "metrics": metrics,
        }
    except Exception as exc:
        log.error("Task 3 failed: %s", exc, exc_info=True)
        raise