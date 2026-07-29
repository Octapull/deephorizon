"""End-to-end integration test for the DeepHorizon inference pipeline.

Tests the full flow:
    1. Export a trained model to ONNX (or use an existing ONNX file)
    2. Start the Python gRPC inference server
    3. Send Enhance / EnhanceBatch / ListModels / Health requests
    4. Validate responses (status, MIME type, image size, metrics)
    5. Stop the server and cleanup

Usage:
    # Full pipeline (export + serve + test)
    python scripts/e2e_integration_test.py \\
        --checkpoint checkpoints/best_model.pt \\
        --model-type pix2pix_generator

    # Use existing ONNX file
    python scripts/e2e_integration_test.py \\
        --onnx-path exports/deephorizon_generator.onnx

    # Skip server startup (test only the export step)
    python scripts/e2e_integration_test.py \\
        --checkpoint checkpoints/best_model.pt \\
        --skip-server

Exit codes:
    0 — all tests passed
    1 — one or more tests failed
    2 — setup error (missing dependencies, etc.)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Proje kökünü path'e ekle
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.e2e_test_helpers import (  # noqa: E402
    assert_response_valid,
    create_grpc_channel,
    generate_synthetic_batch,
    generate_synthetic_image,
    get_inference_stub,
    inference_server,
    numpy_to_png_bytes,
    png_bytes_to_numpy,
    validate_enhance_response,
)


# ---------------------------------------------------------------------------
# Test phases
# ---------------------------------------------------------------------------


def phase_export(
    checkpoint_path: Path | None,
    onnx_path: Path | None,
    model_type: str,
    input_shape: tuple[int, ...],
) -> Path:
    """Phase 1: Export checkpoint to ONNX (if needed).

    Returns:
        Path to the ONNX model file.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: ONNX Export")
    print("=" * 60)

    if onnx_path is not None and onnx_path.exists():
        print(f"  Using existing ONNX: {onnx_path}")
        return onnx_path

    if checkpoint_path is None:
        raise ValueError("Either --checkpoint or --onnx-path must be provided")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Model type: {model_type}")
    print(f"  Input shape: {input_shape}")

    from services.ml.export import export_checkpoint_to_onnx

    onnx_path = onnx_path or checkpoint_path.with_suffix(".onnx")
    onnx_path, metadata_path = export_checkpoint_to_onnx(
        checkpoint_path=checkpoint_path,
        output_path=onnx_path,
        model_type=model_type,
        input_shape=input_shape,
        validate=False,  # Skip validation if onnxruntime not available
    )

    print(f"  ✓ ONNX exported: {onnx_path}")
    print(f"  ✓ Metadata: {metadata_path}")
    return onnx_path


def phase_server_start(
    onnx_path: Path,
    model_id: str,
    skip_server: bool,
) -> dict | None:
    """Phase 2: Start the inference server.

    Returns:
        Server context dict, or None if skipped.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Server Startup")
    print("=" * 60)

    if skip_server:
        print("  Skipped (--skip-server)")
        return None

    print(f"  ONNX: {onnx_path}")
    print(f"  Model ID: {model_id}")

    # Use context manager — server auto-stops on exit
    ctx_manager = inference_server(
        onnx_path=onnx_path,
        model_id=model_id,
        startup_timeout=30.0,
    )
    ctx = ctx_manager.__enter__()
    print(f"  ✓ Server started on port {ctx['port']}")
    return {"ctx_manager": ctx_manager, "ctx": ctx}


def phase_server_stop(server_ctx: dict | None) -> None:
    """Phase 2 cleanup: Stop the inference server."""
    if server_ctx is None:
        return
    ctx_manager = server_ctx["ctx_manager"]
    ctx_manager.__exit__(None, None, None)
    print("  ✓ Server stopped")


def phase_test_enhance(server_ctx: dict | None) -> bool:
    """Phase 3: Test single-image Enhance RPC.

    Returns:
        True if test passed, False otherwise.
    """
    print("\n" + "=" * 60)
    print("PHASE 3: Enhance RPC Test")
    print("=" * 60)

    if server_ctx is None:
        print("  Skipped (no server)")
        return True

    ctx = server_ctx["ctx"]
    port = ctx["port"]
    model_id = ctx["model_id"]

    # Generate synthetic image
    image = generate_synthetic_image(width=64, height=64, seed=42)
    png_bytes = numpy_to_png_bytes(image)
    print(f"  Input: {image.shape} float32 → {len(png_bytes)} bytes PNG")

    # Build gRPC request
    from services.ml.inference_server.pb import inference_pb2  # type: ignore[import-untyped]

    channel = create_grpc_channel(port=port)
    stub = get_inference_stub(channel)

    request = inference_pb2.EnhanceRequest(
        image=inference_pb2.ImagePayload(
            data=png_bytes,
            mime_type="image/png",
            width=64,
            height=64,
        ),
        model_id=model_id,
        scale_factor=1,
        output_format="png",
    )

    # Send request
    start_time = time.perf_counter()
    try:
        response = stub.Enhance(request, timeout=30.0)
    except Exception as exc:
        print(f"  ✗ Enhance RPC failed: {exc}")
        channel.close()
        return False

    elapsed = (time.perf_counter() - start_time) * 1000
    print(f"  Round-trip: {elapsed:.1f}ms")

    # Validate response
    result = validate_enhance_response(response)
    print(f"  Status: {result['status_ok']} ({response.status})")
    print(f"  MIME: {result['mime_ok']} ({response.image.mime_type})")
    print(f"  Size: {result['size_ok']} ({len(response.image.data)} bytes)")
    print(f"  Metrics: {result['has_metrics']} (inference_time_ms={result['inference_time_ms']})")
    print(f"  Output shape: {result['output_shape']}")

    try:
        assert_response_valid(result, context="Enhance")
        print("  ✓ Enhance RPC test PASSED")
        channel.close()
        return True
    except AssertionError as exc:
        print(f"  ✗ Enhance RPC test FAILED: {exc}")
        channel.close()
        return False


def phase_test_enhance_batch(server_ctx: dict | None) -> bool:
    """Phase 4: Test batch EnhanceBatch RPC.

    Returns:
        True if test passed, False otherwise.
    """
    print("\n" + "=" * 60)
    print("PHASE 4: EnhanceBatch RPC Test")
    print("=" * 60)

    if server_ctx is None:
        print("  Skipped (no server)")
        return True

    ctx = server_ctx["ctx"]
    port = ctx["port"]
    model_id = ctx["model_id"]

    # Generate batch
    batch_size = 3
    batch = generate_synthetic_batch(batch_size=batch_size, width=64, height=64)
    print(f"  Input: {batch.shape} float32 batch")

    from services.ml.inference_server.pb import inference_pb2  # type: ignore[import-untyped]

    channel = create_grpc_channel(port=port)
    stub = get_inference_stub(channel)

    requests = []
    for i in range(batch_size):
        png_bytes = numpy_to_png_bytes(batch[i])
        requests.append(
            inference_pb2.EnhanceRequest(
                image=inference_pb2.ImagePayload(
                    data=png_bytes,
                    mime_type="image/png",
                    width=64,
                    height=64,
                ),
                model_id=model_id,
                scale_factor=1,
                output_format="png",
            )
        )

    batch_request = inference_pb2.EnhanceBatchRequest(
        requests=requests,
        max_batch_size=batch_size,
    )

    # Send request
    start_time = time.perf_counter()
    try:
        batch_response = stub.EnhanceBatch(batch_request, timeout=60.0)
    except Exception as exc:
        print(f"  ✗ EnhanceBatch RPC failed: {exc}")
        channel.close()
        return False

    elapsed = (time.perf_counter() - start_time) * 1000
    print(f"  Round-trip: {elapsed:.1f}ms")
    print(f"  Responses: {len(batch_response.responses)}")

    # Validate each response
    all_passed = True
    for i, response in enumerate(batch_response.responses):
        result = validate_enhance_response(response)
        try:
            assert_response_valid(result, context=f"EnhanceBatch[{i}]")
            print(f"  ✓ Response {i}: {result['output_shape']} ({result['inference_time_ms']}ms)")
        except AssertionError as exc:
            print(f"  ✗ Response {i} FAILED: {exc}")
            all_passed = False

    channel.close()
    if all_passed:
        print("  ✓ EnhanceBatch RPC test PASSED")
    else:
        print("  ✗ EnhanceBatch RPC test FAILED")
    return all_passed


def phase_test_list_models(server_ctx: dict | None) -> bool:
    """Phase 5: Test ListModels RPC.

    Returns:
        True if test passed, False otherwise.
    """
    print("\n" + "=" * 60)
    print("PHASE 5: ListModels RPC Test")
    print("=" * 60)

    if server_ctx is None:
        print("  Skipped (no server)")
        return True

    ctx = server_ctx["ctx"]
    port = ctx["port"]
    model_id = ctx["model_id"]

    from services.ml.inference_server.pb import inference_pb2  # type: ignore[import-untyped]

    channel = create_grpc_channel(port=port)
    stub = get_inference_stub(channel)

    try:
        response = stub.ListModels(inference_pb2.ListModelsRequest(), timeout=10.0)
    except Exception as exc:
        print(f"  ✗ ListModels RPC failed: {exc}")
        channel.close()
        return False

    print(f"  Models: {len(response.models)}")
    for model in response.models:
        print(f"    - {model.id} ({model.architecture} v{model.version})")

    # Validate
    found = any(m.id == model_id for m in response.models)
    channel.close()

    if found:
        print(f"  ✓ ListModels RPC test PASSED (found {model_id})")
        return True
    else:
        print(f"  ✗ ListModels RPC test FAILED (model {model_id} not found)")
        return False


def phase_test_health(server_ctx: dict | None) -> bool:
    """Phase 6: Test Health RPC.

    Returns:
        True if test passed, False otherwise.
    """
    print("\n" + "=" * 60)
    print("PHASE 6: Health RPC Test")
    print("=" * 60)

    if server_ctx is None:
        print("  Skipped (no server)")
        return True

    ctx = server_ctx["ctx"]
    port = ctx["port"]

    from services.ml.inference_server.pb import inference_pb2  # type: ignore[import-untyped]

    channel = create_grpc_channel(port=port)
    stub = get_inference_stub(channel)

    try:
        response = stub.Health(inference_pb2.HealthRequest(), timeout=5.0)
    except Exception as exc:
        print(f"  ✗ Health RPC failed: {exc}")
        channel.close()
        return False

    print(f"  OK: {response.ok}")
    print(f"  GPU available: {response.gpu_available}")
    print(f"  Detail: {response.detail}")

    channel.close()

    if response.ok:
        print("  ✓ Health RPC test PASSED")
        return True
    else:
        print("  ✗ Health RPC test FAILED (ok=False)")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepHorizon end-to-end integration test",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to PyTorch checkpoint (.pt)",
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        default=None,
        help="Path to existing ONNX model (skips export)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["unet", "pix2pix_generator"],
        default="pix2pix_generator",
        help="Model architecture (default: pix2pix_generator)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="e2e-test-model",
        help="Model identifier (default: e2e-test-model)",
    )
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs=4,
        default=[1, 1, 64, 64],
        help="Dummy input shape BCHW (default: 1 1 64 64)",
    )
    parser.add_argument(
        "--skip-server",
        action="store_true",
        help="Skip server startup (test only export step)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Save test results to JSON file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 60)
    print("DeepHorizon End-to-End Integration Test")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  ONNX path: {args.onnx_path}")
    print(f"  Model type: {args.model_type}")
    print(f"  Model ID: {args.model_id}")
    print(f"  Input shape: {args.input_shape}")
    print(f"  Skip server: {args.skip_server}")

    results: dict[str, bool] = {}
    server_ctx = None

    try:
        # Phase 1: Export
        onnx_path = phase_export(
            checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
            onnx_path=Path(args.onnx_path) if args.onnx_path else None,
            model_type=args.model_type,
            input_shape=tuple(args.input_shape),
        )
        results["export"] = True

        # Phase 2: Server startup
        server_ctx = phase_server_start(
            onnx_path=onnx_path,
            model_id=args.model_id,
            skip_server=args.skip_server,
        )
        results["server_start"] = server_ctx is not None or args.skip_server

        # Phase 3-6: RPC tests
        results["enhance"] = phase_test_enhance(server_ctx)
        results["enhance_batch"] = phase_test_enhance_batch(server_ctx)
        results["list_models"] = phase_test_list_models(server_ctx)
        results["health"] = phase_test_health(server_ctx)

    except Exception as exc:
        print(f"\n✗ Setup error: {exc}")
        return 2
    finally:
        phase_server_stop(server_ctx)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {name}")

    all_passed = all(results.values())
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")

    # Save results
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "results": results,
                    "all_passed": all_passed,
                    "checkpoint": args.checkpoint,
                    "onnx_path": str(onnx_path) if "onnx_path" in dir() else None,
                    "model_type": args.model_type,
                    "model_id": args.model_id,
                },
                f,
                indent=2,
            )
        print(f"\nResults saved to: {output_path}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
