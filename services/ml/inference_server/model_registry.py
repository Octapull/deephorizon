"""Model registry for the gRPC inference server.

Manages a collection of ONNX models that can be served via gRPC. Each
model is loaded lazily on first request and cached in memory. The
registry exposes:

- ``register``: add a model to the registry
- ``get``: retrieve a loaded model session
- ``list_models``: enumerate registered models with metadata
- ``unload``: free memory for a specific model

The registry is thread-safe via a ``threading.Lock`` to support
concurrent gRPC requests.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a registered model.

    Attributes:
        model_id: Unique identifier (e.g. ``"pix2pix-v1"``).
        architecture: Model architecture — ``"unet"``, ``"pix2pix"``, etc.
        version: Semantic version string (e.g. ``"1.0.0"``).
        onnx_path: Path to the ``.onnx`` model file.
        metadata_path: Optional path to the ``.onnx.json`` metadata sidecar.
        psnr: Validation PSNR (from metadata).
        ssim: Validation SSIM (from metadata).
    """

    model_id: str
    architecture: str
    version: str
    onnx_path: Path
    metadata_path: Path | None = None
    psnr: float = 0.0
    ssim: float = 0.0


@dataclass
class _LoadedModel:
    """Internal: a loaded ONNX inference session + metadata."""

    spec: ModelSpec
    session: object  # onnxruntime.InferenceSession
    input_name: str
    output_name: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]


class ModelRegistry:
    """Thread-safe registry of ONNX models for inference.

    Models are loaded lazily on first ``get()`` call and cached. Use
    ``register()`` to add models at startup, ``get()`` to retrieve a
    session for inference, and ``list_models()`` to enumerate.
    """

    def __init__(self) -> None:
        self._specs: dict[str, ModelSpec] = {}
        self._loaded: dict[str, _LoadedModel] = {}
        self._lock = threading.Lock()

    def register(self, spec: ModelSpec) -> None:
        """Register a model specification.

        Args:
            spec: ModelSpec describing the model.

        Raises:
            ValueError: If model_id is already registered.
            FileNotFoundError: If onnx_path doesn't exist.
        """
        if not spec.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {spec.onnx_path}")

        with self._lock:
            if spec.model_id in self._specs:
                raise ValueError(f"Model already registered: {spec.model_id}")
            self._specs[spec.model_id] = spec

    def unregister(self, model_id: str) -> None:
        """Remove a model from the registry and free its memory."""
        with self._lock:
            self._specs.pop(model_id, None)
            self._loaded.pop(model_id, None)

    def get(self, model_id: str) -> _LoadedModel:
        """Get a loaded model session, loading lazily if needed.

        Args:
            model_id: The model identifier.

        Returns:
            _LoadedModel with the ONNX session and metadata.

        Raises:
            KeyError: If model_id is not registered.
        """
        with self._lock:
            if model_id not in self._specs:
                raise KeyError(f"Model not registered: {model_id}")

            if model_id not in self._loaded:
                self._loaded[model_id] = self._load_model(self._specs[model_id])

            return self._loaded[model_id]

    def is_loaded(self, model_id: str) -> bool:
        """Check if a model is currently loaded in memory."""
        with self._lock:
            return model_id in self._loaded

    def list_specs(self) -> list[ModelSpec]:
        """List all registered model specifications."""
        with self._lock:
            return list(self._specs.values())

    def list_loaded(self) -> list[str]:
        """List model IDs that are currently loaded."""
        with self._lock:
            return list(self._loaded.keys())

    @staticmethod
    def _load_model(spec: ModelSpec) -> _LoadedModel:
        """Load an ONNX model into an inference session."""
        import onnxruntime as ort  # type: ignore[import-untyped]

        # Use CPU by default; GPU support requires onnxruntime-gpu
        session = ort.InferenceSession(
            str(spec.onnx_path),
            providers=["CPUExecutionProvider"],
        )

        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()[0]

        return _LoadedModel(
            spec=spec,
            session=session,
            input_name=input_meta.name,
            output_name=output_meta.name,
            input_shape=tuple(input_meta.shape),
            output_shape=tuple(output_meta.shape),
        )

    def run_inference(
        self,
        model_id: str,
        input_tensor: np.ndarray,
    ) -> np.ndarray:
        """Run inference on a loaded model.

        Args:
            model_id: Model identifier.
            input_tensor: Input array of shape ``(B, C, H, W)`` float32.

        Returns:
            Output array of shape ``(B, C, H', W')`` float32.
        """
        loaded = self.get(model_id)
        outputs = loaded.session.run(
            None,
            {loaded.input_name: input_tensor},
        )
        return outputs[0]
