import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.protocol_evaluation import (
    mean_pool_and_normalize,
    retrieval_metrics,
    rsa_metrics,
)


def test_locked_pooling_retrieval_and_rsa():
    tokens = torch.randn(4, 256, 16)
    pooled, normalized = mean_pool_and_normalize(tokens)
    assert pooled.shape == normalized.shape == (4, 16)
    assert torch.allclose(normalized.norm(dim=-1), torch.ones(4), atol=1e-6)
    # A one-hot embedding has a constant off-diagonal cosine RDM, for which
    # correlation is correctly undefined.  Use a fixed non-degenerate array.
    embedding = np.random.default_rng(42).normal(size=(6, 8)).astype(np.float32)
    retrieval = retrieval_metrics(embedding, embedding)
    assert retrieval["candidate_count"] == 6
    assert retrieval["brain_to_image"]["recall_at_1"] == 1.0
    assert retrieval["image_to_brain"]["recall_at_1"] == 1.0
    rsa = rsa_metrics(embedding, embedding)
    assert np.isclose(rsa["spearman_rsa"], 1.0)
    assert np.isclose(rsa["pearson_rsa"], 1.0)
