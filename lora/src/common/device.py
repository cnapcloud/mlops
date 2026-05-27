import logging
import torch

log = logging.getLogger(__name__)


def get_device(prefer: str | None = None) -> torch.device:
    """
    prefer: "cpu" | "mps" | "cuda" | None
      - None or "auto" → Auto-detects in order: mps > cuda > cpu
      - Specific value → Verifies if the requested device is available and returns it
        (Throws a RuntimeError if unavailable, with no fallback)
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
            raise RuntimeError("MPS was requested but is not available in this environment.")
        device = torch.device("mps")
    elif prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this environment.")
        device = torch.device("cuda")
    elif prefer == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Unknown device specified: '{prefer}' (Supported: cpu | mps | cuda | auto)")

    log.debug("device=%s (prefer=%s)", device, prefer)
    return device