"""
실행 환경 및 결과 요약
-----------------
실행 환경:
- 로컬 드라이버: macOS, Python 3.12.13, Ray Client로 ray://192.168.0.182:10001 에 연결.
- KubeRay 클러스터: Ray 2.55.1, CPU 전용 이미지 rayproject/ray:2.55.1-py312-cpu-aarch64 사용.
- `ray-cluster.yaml` 기준 자원 사양:
    - Head: requests cpu 200m / memory 2Gi, limits cpu 2 / memory 8G
    - Worker: requests cpu 100m / memory 10Gi, limits cpu 4 / memory 18G
    - Worker 수: replicas 1, minReplicas 1, maxReplicas 3
    - Worker는 `podAntiAffinity` 설정으로 다른 노드에 분산 배치를 선호
- 학습 설정: `use_gpu=False`, `backend=gloo`, CPU 리소스만 사용
- 런타임 환경: HF_TOKEN, HF_HOME=/shared/hf_home, HF_HUB_ENABLE_HF_TRANSFER 등을 `runtime_env.env_vars`로 전달

관찰 결과:
- 워커에서 패키지 설치와 모델 로드는 정상 완료됨
- 학습은 CPU로만 진행되어 느리지만, 실제로 epoch 단위로 `train.report`가 찍히며 진행됨
- 관찰된 실행에서는 loss가 epoch 0, 1, 2 순서로 보고되고 3 epoch를 완료함
- Ray Data에서 object store 메모리 비율 경고가 한 번 출력되었지만 작업은 최종 성공함
- 로그 기준으로 `Setting up process group`(21:17:12)부터 최종 `Reporting training result 3`(21:57:28)까지 약 40분 16초가 소요됨
- 처음 RayCluster가 구성된 뒤에는 패키지 설치와 모델 로딩에 시간이 많이 들지만, 이후 epoch 진행은 상대적으로 빠름
- 다만 worker가 OOM으로 재시작되면 이 패키지 설치/모델 로딩 과정을 다시 반복하게 되어 전체 시간이 크게 늘어남

결론:
- 패키지가 포함된 이미지를 미리 준비해야 함
- 모델도 PVC에 사전 배치해 두는 방식이 필요함
- 현재 구성은 CPU 전용이라 너무 느리므로, 최종적으로는 GPU 학습으로 전환해야 함
- 다만 macOS에서 Multipass로 구성한 KubeRay/K8s 환경은 GPU를 사용할 수 없음
"""


import os
import ray
from ray import train
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer, TorchConfig

# ==========================================
# 1. 분산 학습용 데이터셋 준비 (100건 예시)
# ==========================================
def prepare_dataset():
    from datasets import Dataset
    from transformers import AutoTokenizer
    
    model_id = "meta-llama/Llama-3.2-1B"
    # 데이터 생성 단에서 미리 해당 모델의 진짜 eos_token을 텍스트 끝에 붙여줍니다.
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=os.environ.get("HF_TOKEN"))
    
    raw_data = [
        {"text": f"질문 {i}: KubeRay 분산 학습 테스트 시나리오입니다. 답변 {i}: 정상 작동 중입니다.{tokenizer.eos_token}"} 
        for i in range(100)
    ]
    hf_dataset = Dataset.from_dict({"text": [d["text"] for d in raw_data]})
    return ray.data.from_huggingface(hf_dataset)

# ==========================================
# 2. 각 워커 Pod에서 독립적으로 실행될 학습 함수
# ==========================================
def train_func_per_worker(config):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    world_size = train.get_context().get_world_size()
    world_rank = train.get_context().get_world_rank()

    model_id = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32
    )

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    # 내부에서 자동으로 .to(device)를 수행하는 Ray의 TorchTrainer를 활용하기 때문에,
    # 여기서는 명시적으로 모델을 특정 디바이스로 옮기지 않습니다
    model = train.torch.prepare_model(model)

    # Gradient Checkpointing 활성화
    # float32로 올리면 메모리(RAM) 사용량이 늘어나 스왑 지옥에 빠질 수 있으므로, 
    # 중간 계산값을 저장하지 않고 재계산하는 이 옵션을 켜서 RAM 부담을 절반으로 낮춥니다.
    model.gradient_checkpointing_enable()

    local_dataset_shard = train.get_dataset_shard("train_set")
    train_dataloader = local_dataset_shard.iter_batches(
        batch_size=1,
        prefetch_batches=2,
    )

    # 오직 LoRA 레이어(학습 대상) 파라미터만 옵티마이저에 전달
    # 기존 model.parameters()는 동결된 Llama 가중치 10억 개를 매번 스캔하게 하여 
    # CPU 점유율을 뚝 떨어뜨리는 원인이 됩니다. 이를 원천 차단합니다.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=2e-4)

    def collate_fn(batch):
        # Ray 데이터셋의 배치 형태에서 텍스트 리스트 추출
        texts = [str(text) for text in batch["text"]]
        
        tokenized = tokenizer(
            texts,
            padding="max_length", # 고정 크기 배치 구성을 위해 max_length 설정
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        
        labels = tokenized["input_ids"].clone()
        labels[tokenized["attention_mask"] == 0] = -100
        
        tokenized["labels"] = labels
        return tokenized

    # ==========================================
    # 3. 실전 학습 루프 (Training Loop)
    # ==========================================
    current_device = train.torch.get_device()
    for epoch in range(config["epochs"]):
        model.train()
        for batch in train_dataloader:
            inputs = collate_fn(batch)
            inputs = {k: v.to(current_device) for k, v in inputs.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train.report({"loss": loss.item(), "epoch": epoch})


# ==========================================
# 4. KubeRay 분산 오케스트레이션 세팅 및 실행
# ==========================================
if __name__ == "__main__":
    from datasets import Dataset
    import sys
    
    HF_TOKEN = os.environ.get("HF_TOKEN")

    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable is not set. Aborting.", file=sys.stderr)
        sys.exit(1)
    RAY_REMOTE_ADDRESS = "ray://192.168.0.182:10001"

    runtime_env = {
        "pip": [
            "transformers",
            "peft",
            "datasets",
            "torch",
            "xgboost",
        ],
        "env_vars": {
            "HF_TOKEN": HF_TOKEN,
            "HF_HOME": "/shared/hf_home",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING": "1"
        },
    }

    ray.init(
        address=RAY_REMOTE_ADDRESS,
        ignore_reinit_error=True,
        runtime_env=runtime_env
    )

    print(f"Connecting to remote KubeRay cluster at: {RAY_REMOTE_ADDRESS}...")

    ray_dataset = prepare_dataset()

    scaling_config = ScalingConfig(
        num_workers=1,
        use_gpu=False,
        resources_per_worker={"CPU": 1},
    )
    

    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config={"epochs": 3},
        scaling_config=scaling_config,
        torch_config=TorchConfig(backend="gloo"), # CPU 환경에서는 NCCL 대신 Gloo 백엔드 사용
        # torch_config=TorchConfig(backend="nccl"),
        datasets={"train_set": ray_dataset},
        run_config=RunConfig(storage_path="/shared/ray-checkpoints")
    )

    print("--- Starting KubeRay Distributed CPU Training Pipeline ---")
    result = trainer.fit()
    print("--- Pipeline Completed Successfully ---")
    print(f"Best Training Checkpoint Result: {result.checkpoint}")