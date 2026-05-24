"""
task_03_train.py
----------------
Task 3: KubeRay 분산 학습 + MLflow 모델 등록
"""

import json
import logging
import os

from helper import _run_task, _abort

log = logging.getLogger(__name__)


def run() -> dict:
    log.info("=" * 60)
    log.info("Task 3: Train + MLflow 등록 시작")
    log.info("=" * 60)

    try:
        import mlflow
        from mlflow import MlflowClient
        from config import (
            MLFLOW_TRACKING_URI,
            MLFLOW_MODEL_NAME,
            MLFLOW_EXPERIMENT,
            HF_TOKEN,
            RAY_ADDRESS,
            RAY_STORAGE,
            RAY_NUM_WORKERS,
            USE_GPU,
            TRAIN_EPOCHS,
            MODEL_ID,
            HF_HOME,
            LORA_R,
            LORA_ALPHA,
            LEARNING_RATE,
            MAX_SEQ_LEN,
            TRAIN_BATCH,
        )

        if not HF_TOKEN:
            raise EnvironmentError("HF_TOKEN이 설정되어 있지 않습니다.")

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        mlflow_cfg = {
            "tracking_uri":          MLFLOW_TRACKING_URI,
            "experiment":            MLFLOW_EXPERIMENT,
            "registered_model_name": MLFLOW_MODEL_NAME,
            "params": {
                "model_id":      MODEL_ID,
                "epochs":        TRAIN_EPOCHS,
                "num_workers":   RAY_NUM_WORKERS,
                "use_gpu":       USE_GPU,
                "lora_r":        LORA_R,
                "lora_alpha":    LORA_ALPHA,
                "learning_rate": LEARNING_RATE,
                "max_seq_len":   MAX_SEQ_LEN,
                "batch_size":    TRAIN_BATCH,
            },
            "tags": {
                "pipeline":  "mlops-demo",
                "task":      "llm-finetune",
                "framework": "transformers+peft",
            },
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
            raise RuntimeError(
                "Rank 0 워커로부터 mlflow_run_id를 받지 못했습니다. "
                "워커 로그를 확인하세요."
            )
        log.info("Rank 0 워커에서 run_id 수신: %s", run_id)

        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["None"])
        if not versions:
            raise RuntimeError(
                f"MLflow Registry에서 '{MLFLOW_MODEL_NAME}' 모델을 찾을 수 없습니다."
            )
        model_version = next(
            (v.version for v in versions if v.run_id == run_id),
            versions[0].version,
        )
        log.info("MLflow 모델 버전 확인: name=%s, version=%s", MLFLOW_MODEL_NAME, model_version)

        return {
            "status":        "success",
            "run_id":        run_id,
            "model_version": model_version,
            "model_name":    MLFLOW_MODEL_NAME,
            "metrics":       metrics,
        }

    except Exception as e:
        log.error("Task 3 실패: %s", e, exc_info=True)
        return {"status": "failed", "error": str(e)}


def _build_dataset(model_id: str, hf_token: str):
    """Ray Data 데이터셋 구성 (드라이버에서 실행)."""
    import ray
    from datasets import Dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    raw_data = [
        {
            "text": (
                f"질문 {i}: KubeRay 분산 학습 테스트 시나리오입니다. "
                f"답변 {i}: 정상 작동 중입니다.{tokenizer.eos_token}"
            )
        }
        for i in range(5)
    ]
    hf_dataset = Dataset.from_dict({"text": [d["text"] for d in raw_data]})
    return ray.data.from_huggingface(hf_dataset)


# ─────────────────────────────────────────────
# 워커 학습 함수 (KubeRay 워커 Pod에서 실행)
# ─────────────────────────────────────────────

def _train_func_per_worker(config: dict) -> None:
    """
    각 KubeRay 워커 Pod에서 실행되는 학습 함수.

    핵심 패턴:
      - world_rank != 0 → 학습만 수행, MLflow 저장 없이 rt.report()만 호출
      - world_rank == 0 → 학습 완료 후 MLflow run 생성 + log_model + register
                          rt.report()에 mlflow_run_id 포함하여 드라이버에 전달
    """
    import logging
    import torch
    import mlflow
    import mlflow.transformers
    from ray import train as rt
    from ray.train import Checkpoint
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq,
    )
    from datasets import Dataset as HFDataset
    from peft import LoraConfig, get_peft_model

    logger = logging.getLogger(__name__)

    # ── 내부 헬퍼: Ray 배치 → HF Dataset 행 변환 ────────────────────
    # → 워커 함수 내부에 중첩 정의하여 직렬화 문제 방지
    def _batch_to_rows(batch: dict) -> list:
        keys   = list(batch.keys())
        length = len(batch[keys[0]])
        return [{k: batch[k][i] for k in keys} for i in range(length)]

    epochs     = config["epochs"]
    model_id   = config["model_id"]
    mlflow_cfg = config["mlflow_cfg"]

    world_size = rt.get_context().get_world_size()
    world_rank = rt.get_context().get_world_rank()
    logger.info("[Rank %d/%d] 워커 시작", world_rank, world_size)

    # ── 모델 및 토크나이저 로드 ──────────────────────────────────────
    logger.info("[Rank %d] 모델 로드 중: %s", world_rank, model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # GPU 전환 시 torch.bfloat16
    )

    # ── LoRA 어댑터 적용 ─────────────────────────────────────────────
    peft_config = LoraConfig(
        r=mlflow_cfg["params"]["lora_r"],
        lora_alpha=mlflow_cfg["params"]["lora_alpha"],
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.gradient_checkpointing_enable()

    logger.info("[Rank %d] 모델 로드 완료 (trainable params: %d)",
                world_rank,
                sum(p.numel() for p in model.parameters() if p.requires_grad))

    # ── LoRA 파라미터만 옵티마이저에 등록 ───────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=mlflow_cfg["params"]["learning_rate"],
    )

    # ── Ray Dataset 샤드 → HuggingFace Dataset 변환 ─────────────────
    # Trainer는 HF Dataset만 지원하므로 변환 필요
    local_shard = rt.get_dataset_shard("train_set")
    rows = [
        row
        for batch in local_shard.iter_batches(batch_size=256)
        for row in _batch_to_rows(batch)
    ]
    hf_dataset = HFDataset.from_list(rows)

    def tokenize_fn(examples):
        enc = tokenizer(
            [str(t) for t in examples["text"]],
            padding="max_length",
            truncation=True,
            max_length=mlflow_cfg["params"]["max_seq_len"],
        )
        enc["labels"] = [
            [-100 if mask == 0 else token_id
             for mask, token_id in zip(m, ids)]
            for m, ids in zip(enc["attention_mask"], enc["input_ids"])
        ]
        return enc

    tokenized = hf_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=hf_dataset.column_names,
    )

    # train / eval 분리 (9:1)
    split    = tokenized.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    eval_ds  = split["test"]

    # ── DataCollator ─────────────────────────────────────────────────
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )

    # ── TrainingArguments ────────────────────────────────────────────
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
        fp16=False,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        dataloader_pin_memory=False,
        push_to_hub=False,
        use_cpu=not torch.cuda.is_available(),
        report_to="none",
    )

    # ── Trainer (옵티마이저 명시적 전달) ─────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        optimizers=(optimizer, None),  # (optimizer, lr_scheduler) — scheduler는 Trainer가 생성
    )

    train_result = trainer.train()
    last_loss    = train_result.training_loss

    logger.info("[Rank %d] 전체 학습 완료 (최종 loss: %.6f)", world_rank, last_loss)


    # 초기화 안 해주면 함수 스코프 밖에서 에러가 날 수 있으므로 미리 선언
    run_id = None

    # ── Rank 0이 아닌 워커: MLflow 저장 없이 결과만 리포트 ──────────
    if world_rank != 0:
        logger.info("[Rank %d] Rank 0이 아니므로 MLflow 저장을 건너뜁니다.", world_rank)
        rt.report({"loss": last_loss, "epoch": epochs - 1})
        return

    # ── Rank 0 전용: MLflow run 생성 + 모델 등록 ────────────────────
    logger.info("[Rank 0] MLflow 연동 시작 (Tracking URI: %s)", mlflow_cfg["tracking_uri"])

    try:
        mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
        mlflow.set_experiment(mlflow_cfg["experiment"])

        logger.info("[Rank 0] MLflow Run 시작 중 (Experiment: %s)", mlflow_cfg["experiment"])

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info("[Rank 0] MLflow Run 생성 완료 (Run ID: %s)", run_id)

            mlflow.log_params(mlflow_cfg["params"])
            mlflow.set_tags(mlflow_cfg["tags"])
            mlflow.log_metric("final_loss", last_loss)
            logger.info("[Rank 0] 파라미터 / 태그 / 메트릭 기록 완료")

            # DDP 래핑 해제: 원본 모델 추출
            raw_model = model.module if hasattr(model, "module") else model

            logger.info(
                "[Rank 0] MLflow에 모델 등록 중 (Model Name: %s)...",
                mlflow_cfg["registered_model_name"],
            )

            # ✨ [수정] mlflow_tf 대신 mlflow.transformers.log_model 사용
            # 인자 역시 transformers_model dict 가 아니라 각각 풀어줍니다.
            mlflow.transformers.log_model(
                transformers_model={
                    "model": raw_model,
                    "tokenizer": tokenizer
                },
                artifact_path="model",
                registered_model_name=mlflow_cfg["registered_model_name"],
            )
            logger.info("[Rank 0] MLflow 모델 및 레지스트리 등록 완료!")

    except Exception as e:
        logger.error("[Rank 0] MLflow 저장 중 에러 발생: %s", str(e), exc_info=True)
        raise
    
    checkpoint_dir = os.path.join(
        output_dir,
        f"checkpoint_run_{run_id or 'unknown'}",
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "loss": last_loss,
                "epoch": epochs - 1,
                "model_id": model_id,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    rt.report({
        "loss":          last_loss,
        "epoch":         epochs - 1,
        "mlflow_run_id": run_id,
    }, checkpoint=Checkpoint.from_directory(checkpoint_dir))
    logger.info("[Rank 0] 최종 결과 리포트 완료 및 워커 종료")


# ─────────────────────────────────────────────
# KubeRay 학습 실행 (드라이버)
# ─────────────────────────────────────────────

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
    from ray.train.torch import TorchTrainer, TorchConfig

    runtime_env = {
        "pip": ["transformers", "peft", "datasets", "torch", "mlflow", "torchvision"],
        "env_vars": {
            "HF_TOKEN":                                   hf_token,
            "HF_HOME":                                    hf_home,
            "HF_XET_HIGH_PERFORMANCE":                    "1",
            "RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING":       "1",
            "RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION": "0.5",
        },
    }

    log.info("Ray 클러스터 연결: %s", ray_address)
    ray.init(address=ray_address,
             ignore_reinit_error=True,
             runtime_env=runtime_env)

    try:
        ray_dataset = _build_dataset(model_id=model_id, hf_token=hf_token)

        resources_per_worker = {"GPU": 1} if use_gpu else {"CPU": 4}
        scaling_config = ScalingConfig(
            num_workers=num_workers,
            use_gpu=use_gpu,
            resources_per_worker=resources_per_worker,
        )

        trainer = TorchTrainer(
            train_loop_per_worker=_train_func_per_worker,
            train_loop_config={
                "epochs":     epochs,
                "model_id":   model_id,
                "mlflow_cfg": mlflow_cfg,
            },
            scaling_config=scaling_config,
            torch_config=TorchConfig(backend="nccl" if use_gpu else "gloo"),
            datasets={"train_set": ray_dataset},
            run_config=RunConfig(storage_path=storage_path),
        )

        log.info(
            "KubeRay 분산 학습 시작 (workers=%d, gpu=%s, epochs=%d)",
            num_workers, use_gpu, epochs,
        )
        result = trainer.fit()
        log.info("KubeRay 학습 완료")

        return result.metrics or {}

    finally:
        ray.shutdown()
        log.info("Ray 연결 종료")


def main() -> None:
    results = _run_task("Task 3: Train", run)
    if results["status"] != "success":
        _abort("Task 3 학습 실패", results)


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()