"""Pytest fixtures for services.ml tests."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch


@pytest.fixture
def device() -> torch.device:
    """Test cihazı — CUDA varsa cuda, yoksa cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Örnek batch — (degraded, clean) çifti, 64x64 float32 [0, 1]."""
    torch.manual_seed(42)
    degraded = torch.rand(2, 1, 64, 64, dtype=torch.float32)
    clean = torch.rand(2, 1, 64, 64, dtype=torch.float32)
    return degraded, clean


@pytest.fixture
def sample_image() -> torch.Tensor:
    """Tek görüntü — (1, 1, 64, 64) float32 [0, 1]."""
    torch.manual_seed(42)
    return torch.rand(1, 1, 64, 64, dtype=torch.float32)


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Geçici çıktı dizini — checkpoint + artifact testleri için."""
    out = tmp_path / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out
