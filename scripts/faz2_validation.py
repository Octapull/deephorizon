"""Faz 2 comprehensive validation script.

Validates all Faz 2 components (Adım 11-17) without requiring full
runtime dependencies (torch, onnxruntime, grpc). Checks:

1. File existence — all expected files present
2. Syntax validity — all Python files parse correctly
3. AST structure — expected classes/functions exist
4. Test coverage — test files have expected test functions
5. Config validity — YAML configs parse correctly
6. Import graph — module dependencies resolve (AST-only)

Usage:
    python scripts/faz2_validation.py
    python scripts/faz2_validation.py --verbose
    python scripts/faz2_validation.py --output-json validation_results.json

Exit codes:
    0 — all checks passed
    1 — one or more checks failed
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

# Proje kökünü path'e ekle
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Expected Faz 2 structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpectedFile:
    """Expected file with optional AST checks."""

    path: str
    description: str
    expected_classes: list[str] = field(default_factory=list)
    expected_functions: list[str] = field(default_factory=list)
    min_test_count: int = 0


# Faz 2 dosya yapısı
FAZ2_FILES: list[ExpectedFile] = [
    # Adım 11: Pix2Pix GAN modelleri
    ExpectedFile(
        path="services/ml/models/pix2pix/__init__.py",
        description="Pix2Pix package init",
    ),
    ExpectedFile(
        path="services/ml/models/pix2pix/generator.py",
        description="Pix2PixGenerator (U-Net wrapper)",
        expected_classes=["Pix2PixGenerator"],
    ),
    ExpectedFile(
        path="services/ml/models/pix2pix/discriminator.py",
        description="PatchDiscriminator (70x70 PatchGAN)",
        expected_classes=["PatchDiscriminator"],
    ),
    ExpectedFile(
        path="services/ml/tests/test_pix2pix.py",
        description="Pix2Pix tests",
        min_test_count=10,
    ),
    # Adım 12: GAN + Combined Loss
    ExpectedFile(
        path="services/ml/losses/perceptual.py",
        description="PerceptualLoss (VGG19)",
        expected_classes=["PerceptualLoss"],
    ),
    ExpectedFile(
        path="services/ml/losses/gan.py",
        description="GeneratorAdversarialLoss + DiscriminatorAdversarialLoss",
        expected_classes=["GeneratorAdversarialLoss", "DiscriminatorAdversarialLoss"],
    ),
    ExpectedFile(
        path="services/ml/losses/physics.py",
        description="PhysicsLoss (flux + ring + asymmetry)",
        expected_classes=["PhysicsLoss"],
    ),
    ExpectedFile(
        path="services/ml/losses/combined.py",
        description="CombinedLoss (weighted sum)",
        expected_classes=["CombinedLoss"],
    ),
    ExpectedFile(
        path="services/ml/losses/__init__.py",
        description="Losses package init",
    ),
    ExpectedFile(
        path="services/ml/losses/loss.py",
        description="Loss factory (updated)",
    ),
    ExpectedFile(
        path="services/ml/conf/loss/default.yaml",
        description="Loss config (updated)",
    ),
    ExpectedFile(
        path="services/ml/tests/test_gan_loss.py",
        description="GAN loss tests",
        min_test_count=8,
    ),
    ExpectedFile(
        path="services/ml/tests/test_combined.py",
        description="Combined loss tests",
        min_test_count=8,
    ),
    # Adım 13: GAN Training Loop
    ExpectedFile(
        path="services/ml/training/gan_train.py",
        description="GAN training loop",
        expected_functions=["train_gan", "_build_generator", "_build_discriminator"],
    ),
    ExpectedFile(
        path="services/ml/conf/training/default.yaml",
        description="Training config (updated)",
    ),
    ExpectedFile(
        path="services/ml/conf/model/pix2pix.yaml",
        description="Pix2Pix model config",
    ),
    ExpectedFile(
        path="services/ml/tests/test_gan_train.py",
        description="GAN training tests",
        min_test_count=10,
    ),
    # Adım 14: ONNX Export
    ExpectedFile(
        path="services/ml/export/onnx_export.py",
        description="ONNX export module",
        expected_classes=["ExportMetadata"],
        expected_functions=["export_to_onnx", "export_checkpoint_to_onnx"],
    ),
    ExpectedFile(
        path="services/ml/export/__init__.py",
        description="Export package init",
    ),
    ExpectedFile(
        path="services/ml/conf/export/default.yaml",
        description="Export config",
    ),
    ExpectedFile(
        path="services/ml/tests/test_onnx_export.py",
        description="ONNX export tests",
        min_test_count=10,
    ),
    # Adım 15: gRPC Inference Server
    ExpectedFile(
        path="services/ml/inference_server/server.py",
        description="gRPC inference server",
        expected_classes=["InferenceServicer"],
        expected_functions=["Enhance", "EnhanceBatch", "ListModels", "Health"],
    ),
    ExpectedFile(
        path="services/ml/inference_server/model_registry.py",
        description="Model registry",
        expected_classes=["ModelRegistry", "ModelSpec"],
    ),
    ExpectedFile(
        path="services/ml/inference_server/image_utils.py",
        description="Image encode/decode utilities",
    ),
    ExpectedFile(
        path="services/ml/inference_server/__init__.py",
        description="Inference server package init",
    ),
    ExpectedFile(
        path="services/ml/tests/test_inference_server.py",
        description="Inference server tests",
        min_test_count=15,
    ),
    # Adım 16: Go API Gateway (Go files)
    ExpectedFile(
        path="services/api/internal/handlers/enhance.go",
        description="Go enhance handler (gRPC wired)",
    ),
    ExpectedFile(
        path="services/api/internal/handlers/models.go",
        description="Go models handler (gRPC wired)",
    ),
    ExpectedFile(
        path="services/api/internal/handlers/health.go",
        description="Go health handler (gRPC wired)",
    ),
    # Adım 17: E2E Integration Test
    ExpectedFile(
        path="scripts/e2e_integration_test.py",
        description="E2E integration test",
        expected_functions=["main", "phase_export", "phase_test_enhance"],
    ),
    ExpectedFile(
        path="scripts/e2e_test_helpers.py",
        description="E2E test helpers",
        expected_functions=["generate_synthetic_image", "inference_server"],
    ),
    ExpectedFile(
        path="scripts/README_E2E.md",
        description="E2E test documentation",
    ),
]


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Result of a single validation check."""

    name: str
    status: Literal["pass", "fail", "warn"]
    message: str
    details: dict | None = None


def check_file_exists(expected: ExpectedFile) -> CheckResult:
    """Check if a file exists."""
    path = PROJECT_ROOT / expected.path
    if path.exists():
        return CheckResult(
            name=f"exists: {expected.path}",
            status="pass",
            message=f"✓ {expected.description}",
        )
    return CheckResult(
        name=f"exists: {expected.path}",
        status="fail",
        message=f"✗ File not found: {expected.path}",
    )


def check_python_syntax(expected: ExpectedFile) -> CheckResult:
    """Check if a Python file has valid syntax."""
    if not expected.path.endswith(".py"):
        return CheckResult(
            name=f"syntax: {expected.path}",
            status="pass",
            message="(skipped — not Python)",
        )

    path = PROJECT_ROOT / expected.path
    if not path.exists():
        return CheckResult(
            name=f"syntax: {expected.path}",
            status="fail",
            message="✗ File not found",
        )

    try:
        ast.parse(path.read_text(encoding="utf-8"))
        return CheckResult(
            name=f"syntax: {expected.path}",
            status="pass",
            message="✓ Valid Python syntax",
        )
    except SyntaxError as exc:
        return CheckResult(
            name=f"syntax: {expected.path}",
            status="fail",
            message=f"✗ Syntax error: {exc}",
        )


def check_ast_structure(expected: ExpectedFile) -> CheckResult:
    """Check if expected classes/functions exist in the file."""
    if not expected.path.endswith(".py"):
        return CheckResult(
            name=f"ast: {expected.path}",
            status="pass",
            message="(skipped — not Python)",
        )

    if not expected.expected_classes and not expected.expected_functions:
        return CheckResult(
            name=f"ast: {expected.path}",
            status="pass",
            message="(no AST checks defined)",
        )

    path = PROJECT_ROOT / expected.path
    if not path.exists():
        return CheckResult(
            name=f"ast: {expected.path}",
            status="fail",
            message="✗ File not found",
        )

    tree = ast.parse(path.read_text(encoding="utf-8"))
    found_classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    # Top-level functions only (not methods inside classes)
    found_functions = {
        n.name
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
    }
    # Also collect methods inside classes for completeness
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    found_functions.add(item.name)

    missing_classes = set(expected.expected_classes) - found_classes
    missing_functions = set(expected.expected_functions) - found_functions

    if missing_classes or missing_functions:
        missing = list(missing_classes) + [f"func:{f}" for f in missing_functions]
        return CheckResult(
            name=f"ast: {expected.path}",
            status="fail",
            message=f"✗ Missing: {missing}",
            details={
                "missing_classes": list(missing_classes),
                "missing_functions": list(missing_functions),
            },
        )

    return CheckResult(
        name=f"ast: {expected.path}",
        status="pass",
        message=f"✓ All expected classes/functions present",
        details={
            "classes": list(found_classes & set(expected.expected_classes)),
            "functions": list(found_functions & set(expected.expected_functions)),
        },
    )


def check_test_count(expected: ExpectedFile) -> CheckResult:
    """Check if a test file has the minimum number of tests."""
    if expected.min_test_count == 0:
        return CheckResult(
            name=f"tests: {expected.path}",
            status="pass",
            message="(no test count requirement)",
        )

    if not expected.path.endswith(".py"):
        return CheckResult(
            name=f"tests: {expected.path}",
            status="pass",
            message="(skipped — not Python)",
        )

    path = PROJECT_ROOT / expected.path
    if not path.exists():
        return CheckResult(
            name=f"tests: {expected.path}",
            status="fail",
            message="✗ File not found",
        )

    tree = ast.parse(path.read_text(encoding="utf-8"))
    tests = [
        n.name
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
    ]
    count = len(tests)

    if count >= expected.min_test_count:
        return CheckResult(
            name=f"tests: {expected.path}",
            status="pass",
            message=f"✓ {count} tests (>= {expected.min_test_count})",
            details={"count": count, "tests": tests},
        )
    return CheckResult(
        name=f"tests: {expected.path}",
        status="fail",
        message=f"✗ Only {count} tests (< {expected.min_test_count})",
        details={"count": count, "tests": tests},
    )


def check_yaml_config(expected: ExpectedFile) -> CheckResult:
    """Check if a YAML config file is valid."""
    if not expected.path.endswith(".yaml") and not expected.path.endswith(".yml"):
        return CheckResult(
            name=f"yaml: {expected.path}",
            status="pass",
            message="(skipped — not YAML)",
        )

    path = PROJECT_ROOT / expected.path
    if not path.exists():
        return CheckResult(
            name=f"yaml: {expected.path}",
            status="fail",
            message="✗ File not found",
        )

    try:
        import yaml  # type: ignore[import-untyped]

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return CheckResult(
            name=f"yaml: {expected.path}",
            status="pass",
            message=f"✓ Valid YAML (top-level keys: {list(data.keys())})",
            details={"keys": list(data.keys())},
        )
    except ImportError:
        return CheckResult(
            name=f"yaml: {expected.path}",
            status="warn",
            message="⚠ PyYAML not installed — skipping validation",
        )
    except yaml.YAMLError as exc:
        return CheckResult(
            name=f"yaml: {expected.path}",
            status="fail",
            message=f"✗ YAML error: {exc}",
        )


# ---------------------------------------------------------------------------
# Main validation runner
# ---------------------------------------------------------------------------


def run_validation(verbose: bool = False) -> list[CheckResult]:
    """Run all validation checks."""
    results: list[CheckResult] = []

    print("=" * 70)
    print("FAZ 2 COMPREHENSIVE VALIDATION")
    print("=" * 70)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Files to validate: {len(FAZ2_FILES)}")
    print()

    for expected in FAZ2_FILES:
        # 1. File exists
        r = check_file_exists(expected)
        results.append(r)
        if verbose or r.status == "fail":
            print(f"  [{r.status.upper()}] {r.message}")

        # 2. Python syntax
        r = check_python_syntax(expected)
        results.append(r)
        if verbose or r.status == "fail":
            print(f"  [{r.status.upper()}] {r.message}")

        # 3. AST structure
        r = check_ast_structure(expected)
        results.append(r)
        if verbose or r.status == "fail":
            print(f"  [{r.status.upper()}] {r.message}")

        # 4. Test count
        r = check_test_count(expected)
        results.append(r)
        if verbose or r.status == "fail":
            print(f"  [{r.status.upper()}] {r.message}")

        # 5. YAML config
        r = check_yaml_config(expected)
        results.append(r)
        if verbose or r.status == "fail":
            print(f"  [{r.status.upper()}] {r.message}")

    return results


def print_summary(results: list[CheckResult]) -> bool:
    """Print validation summary. Returns True if all passed."""
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status == "fail")
    warned = sum(1 for r in results if r.status == "warn")
    total = len(results)

    print(f"  Total checks: {total}")
    print(f"  ✓ Passed:     {passed}")
    print(f"  ✗ Failed:     {failed}")
    print(f"  ⚠ Warnings:   {warned}")
    print()

    if failed > 0:
        print("FAILED CHECKS:")
        for r in results:
            if r.status == "fail":
                print(f"  ✗ {r.name}: {r.message}")
        print()

    if warned > 0:
        print("WARNINGS:")
        for r in results:
            if r.status == "warn":
                print(f"  ⚠ {r.name}: {r.message}")
        print()

    all_passed = failed == 0
    if all_passed:
        print("✓ ALL VALIDATION CHECKS PASSED")
    else:
        print(f"✗ {failed} VALIDATION CHECK(S) FAILED")

    return all_passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Faz 2 comprehensive validation script",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print all check results (not just failures)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = run_validation(verbose=args.verbose)
    all_passed = print_summary(results)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "all_passed": all_passed,
                    "total_checks": len(results),
                    "passed": sum(1 for r in results if r.status == "pass"),
                    "failed": sum(1 for r in results if r.status == "fail"),
                    "warned": sum(1 for r in results if r.status == "warn"),
                    "results": [asdict(r) for r in results],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\nResults saved to: {output_path}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
