from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CETSTGLConfig:
    lookback: int = 24
    patch_len: int = 3
    hidden_dim: int = 128
    graph_top_k: int = 5
    alarm_decay_eta_hours: float = 6.0
    horizons: tuple[int, ...] = (1, 3, 6, 12)


class CETSTGLModel:
    """Interface placeholder for the proposed neural model.

    The current workspace does not have PyTorch installed. Keep this interface
    stable so the first neural implementation can reuse the data pipeline.
    """

    def __init__(self, config: CETSTGLConfig):
        self.config = config

    def fit(self, *args, **kwargs):
        raise NotImplementedError(
            "CET-STGL neural training requires a deep learning backend. "
            "Use run_smoke.py for the currently runnable sklearn baseline."
        )

    def predict(self, *args, **kwargs):
        raise NotImplementedError("CET-STGL neural inference is not implemented in this skeleton.")
