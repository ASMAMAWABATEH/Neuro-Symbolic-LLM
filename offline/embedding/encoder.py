"""Frozen transformer hidden-state encoder (mean pool non-padding tokens)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class FrozenLLMEncoder:
    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        max_length: int = 256,
        batch_size: int = 8,
    ) -> None:
        if max_length < 1 or batch_size < 1:
            raise ValueError("max_length and batch_size must be positive")
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is None:
                raise ValueError(
                    "Tokenizer has neither a padding token nor an EOS token"
                )
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device).eval()
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            raise ValueError("texts must not be empty")
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("each text must be a nonempty string")
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = list(texts[start : start + self.batch_size])
            tokens = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_attention_mask=True,
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            outputs = self.model(**tokens)
            hidden = outputs.last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            batches.append(pooled.float().cpu().numpy())
        return np.concatenate(batches, axis=0)
