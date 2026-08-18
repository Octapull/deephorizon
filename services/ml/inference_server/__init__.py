"""gRPC inference server for DeepHorizon models.

Modules:
    server: gRPC servicer + CLI entry point.
    model_registry: Thread-safe ONNX model registry.
    image_utils: Image encoding/decoding (PNG, JPEG, FITS, raw float32).
"""
from __future__ import annotations

from services.ml.inference_server.image_utils import (
    decode_image_to_tensor,
    encode_tensor_to_image,
)
from services.ml.inference_server.model_registry import ModelRegistry, ModelSpec
from services.ml.inference_server.server import (
    InferenceServicer,
    build_registry_from_args,
    serve,
)

__all__ = [
    "InferenceServicer",
    "ModelRegistry",
    "ModelSpec",
    "build_registry_from_args",
    "decode_image_to_tensor",
    "encode_tensor_to_image",
    "serve",
]
