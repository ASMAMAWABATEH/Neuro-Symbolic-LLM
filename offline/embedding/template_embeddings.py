"""Initialize and persist template-associated key/value vectors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from .context_builder import TemplateContexts


class TextEncoder(Protocol):
    model_name: str

    def encode(self, texts: tuple[str, ...]) -> np.ndarray: ...


def initialize_template_kv(template: TemplateContexts, encoder: TextEncoder) -> dict:
    """Use the same averaged supporting-context embedding for initial K and V.

    This is initialization only; learned projections/updates are outside scope.
    """
    embeddings = np.asarray(encoder.encode(template.contexts), dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(template.contexts):
        raise ValueError("encoder must return [num_contexts, hidden_dim]")
    if embeddings.shape[1] == 0 or not np.isfinite(embeddings).all():
        raise ValueError("encoder returned invalid embeddings")
    average = embeddings.mean(axis=0)
    vector = average.tolist()
    return {
        "pattern_id": template.pattern_id,
        "key": vector,
        "value": vector.copy(),
        "num_contexts": len(template.contexts),
        "embedding_dim": len(vector),
        "encoder_model": encoder.model_name,
        "pooling": "last_hidden_state_attention_masked_mean",
        "initialization": "mean_supporting_context_embeddings_for_key_and_value",
    }


def save_template_embeddings(records: list[dict], output_path: str | Path) -> None:
    """Write a JSONL file: one template-associated K/V record per line."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, allow_nan=False) + "\n")
