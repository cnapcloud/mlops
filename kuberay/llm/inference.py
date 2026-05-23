# inference.py
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SAVE_PATH = "./lora_output"
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"

def load_model(save_path: str = SAVE_PATH):
    tokenizer = AutoTokenizer.from_pretrained(save_path)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, save_path)
    model.eval()
    return model, tokenizer


def inference(prompt: str, model=None, tokenizer=None, save_path: str = SAVE_PATH):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    if model is None or tokenizer is None:
        model, tokenizer = load_model(save_path)
    
    model.to(device)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    if tokenizer.eos_token_id in generated:
        eos_index = (generated == tokenizer.eos_token_id).nonzero(as_tuple=True)[0][0].item()
        generated = generated[:eos_index]
        
    return tokenizer.decode(generated, skip_special_tokens=True)

if __name__ == "__main__":
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN environment variable is not set. Aborting.", file=sys.stderr)
        sys.exit(1)

    print("모델 로딩 중...")
    model, tokenizer = load_model()
    print("모델 로딩 완료")

    # 단일 추론
    prompt = "질문 5: KubeRay 분산 학습 테스트 시나리오입니다."
    result = inference(prompt, model=model, tokenizer=tokenizer)
    print(f"\n[추론 결과]\n{result}")

    # 배치 추론
    prompts = [
        "질문 1: KubeRay 분산 학습 테스트 시나리오입니다.",
        "질문 10: KubeRay 분산 학습 테스트 시나리오입니다.",
        "질문 50: KubeRay 분산 학습 테스트 시나리오입니다.",
    ]
    print("\n[배치 추론 결과]")
    for p in prompts:
        result = inference(p, model=model, tokenizer=tokenizer)
        print(f"입력: {p}")
        print(f"출력: {result}\n")