"""vLLM multi-LoRA helpers for STRIDE-FT (one base model, swap adapters per module)."""

from __future__ import annotations

from typing import Any

try:
    from vllm.lora.request import LoRARequest
except ImportError:
    LoRARequest = None


def make_lora_request(name: str, lora_int_id: int, path: str | None) -> Any:
    """Return ``LoRARequest`` or ``None`` if ``path`` is empty."""
    if not path or not str(path).strip():
        return None
    if LoRARequest is None:
        raise RuntimeError(
            "vLLM LoRA is unavailable (could not import vllm.lora.request.LoRARequest). "
            "Install a CUDA build of vllm matching your stack."
        )
    return LoRARequest(name, lora_int_id, str(path).strip())


def any_lora_paths(*paths: str | None) -> bool:
    return any(p and str(p).strip() for p in paths)


def llm_lora_init_kwargs(
    *,
    use_lora: bool,
    max_lora_rank: int = 64,
    max_loras: int = 8,
) -> dict[str, Any]:
    """Extra keyword arguments for ``vllm.LLM`` when serving PEFT adapters."""
    if not use_lora:
        return {}
    return {
        "enable_lora": True,
        "max_lora_rank": max_lora_rank,
        "max_loras": max_loras,
    }
