"""
Dense encoder for retrieval: loads Contriever weights from the Hugging Face Hub.

Default checkpoint: ``facebook/contriever``. Pass any compatible HF model id or local path
to ``--retriever_model_path``.
"""

from __future__ import annotations

import torch
from transformers import AutoModel, AutoTokenizer


class _HFContrieverEncoder(torch.nn.Module):
    """Mean-pooled sentence embeddings (matches common Contriever usage)."""

    def __init__(self, model_name_or_path: str) -> None:
        super().__init__()
        self.inner = AutoModel.from_pretrained(model_name_or_path)

    @property
    def device(self) -> torch.device:
        return next(self.inner.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        kwargs.pop("token_type_ids", None)
        out = self.inner(input_ids=input_ids, attention_mask=attention_mask)
        token_emb = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
        summed = (token_emb * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts


def load_contriever_and_tokenizer(model_name_or_path: str = "facebook/contriever"):
    """
    Returns ``(model, tokenizer)`` where ``model(**batch)`` yields [B, H] embeddings
    (DenseRetriever applies L2 normalize).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = _HFContrieverEncoder(model_name_or_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, tokenizer
