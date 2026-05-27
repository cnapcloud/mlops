import logging
import torch

log = logging.getLogger(__name__)


def get_device(prefer: str | None = None) -> torch.device:
    """
    prefer: "cpu" | "mps" | "cuda" | None
      - None 또는 "auto" → mps > cuda > cpu 순으로 자동 감지
      - 명시하면 해당 device가 실제로 사용 가능한지 확인 후 반환
        (불가능하면 fallback 없이 RuntimeError)
    """
    prefer = (prefer or "auto").lower()

    if prefer == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    elif prefer == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS를 요청했지만 이 환경에서 사용할 수 없습니다.")
        device = torch.device("mps")
    elif prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA를 요청했지만 이 환경에서 사용할 수 없습니다.")
        device = torch.device("cuda")
    elif prefer == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"알 수 없는 device 값: '{prefer}' (cpu | mps | cuda | auto)")

    log.debug("device=%s (prefer=%s)", device, prefer)
    return device
