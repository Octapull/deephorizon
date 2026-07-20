"""
DeepHorizon — Great Expectations Doğrulama Suite'i
====================================================
great-expectations >= 1.4.0 (Fluent API / GE v1) ile çalışır.

Üç veri setini doğrular:
  • training_512  — data/training/{clean,degraded}/  → 10K x (512,512) float32
  • synthetic_128 — data/raw/simulated/{clean,degraded}/ → 1K x (128,128) float32
  • eht_raw       — data/raw/eht/**/*.uvfits → dosya varlığı

Airflow task'ından çağrılabileceği gibi bağımsız da çalışır:
  python scripts/ge_suite.py
  python scripts/ge_suite.py --source training_512
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import great_expectations as gx
    from great_expectations.core import ExpectationSuite
    from great_expectations import expectations as gxe
except ImportError:
    print("great-expectations kurulu değil. `pip install great-expectations>=1.4.0`", file=sys.stderr)
    sys.exit(1)

REPO_ROOT = Path(__file__).parent.parent


def _build_stats_df(directory: Path, expected_shape: tuple[int, int], max_sample: int = 0) -> pd.DataFrame:
    """
    Dizindeki .npy dosyalarını yükleyip per-dosya istatistiklerini döner.
    max_sample > 0 ise sadece o kadar rastgele dosyayı örnekler (hız için).
    """
    all_files = sorted(directory.glob("*.npy"))
    if not all_files:
        raise FileNotFoundError(f"{directory} içinde .npy dosyası bulunamadı.")

    files = all_files
    if max_sample > 0 and len(all_files) > max_sample:
        rng = np.random.default_rng(seed=0)
        indices = rng.choice(len(all_files), size=max_sample, replace=False)
        files = [all_files[i] for i in sorted(indices)]

    rows = []
    for fpath in files:
        try:
            arr = np.load(fpath, allow_pickle=False)
        except Exception:
            rows.append({
                "filename": fpath.name, "load_error": True, "ndim": None,
                "shape_h": None, "shape_w": None, "dtype": None,
                "min_val": None, "max_val": None, "mean_val": None,
                "has_nan": None, "has_inf": None, "is_all_zero": None,
                "expected_h": expected_shape[0], "expected_w": expected_shape[1],
            })
            continue
        rows.append({
            "filename":    fpath.name,
            "load_error":  False,
            "ndim":        int(arr.ndim),
            "shape_h":     int(arr.shape[0]),
            "shape_w":     int(arr.shape[1]) if arr.ndim > 1 else 0,
            "dtype":       str(arr.dtype),
            "min_val":     float(arr.min()),
            "max_val":     float(arr.max()),
            "mean_val":    float(arr.mean()),
            "has_nan":     bool(np.isnan(arr).any()),
            "has_inf":     bool(np.isinf(arr).any()),
            "is_all_zero": bool((arr == 0).all()),
            "expected_h":  expected_shape[0],
            "expected_w":  expected_shape[1],
        })
    return pd.DataFrame(rows)


def _build_suite(
    suite_name: str,
    expected_shape: tuple[int, int],
) -> "ExpectationSuite":
    """
    GE suite oluşturur.

    Satır sayısı beklentisi YOK — örnekleme yapıldığında gerçek
    dosya sayısı değişir, hardcode etmek yanlış sonuç verir.
    Dosya sayısı kontrolü çağrıdan önce ayrıca yapılır.
    """
    suite = gx.ExpectationSuite(name=suite_name)
    h, w = expected_shape

    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(column="load_error", value_set=[False])
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="ndim", min_value=2, max_value=2)
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="shape_h", min_value=h, max_value=h)
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="shape_w", min_value=w, max_value=w)
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(column="dtype", value_set=["float32"])
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="min_val", min_value=0.0)
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(column="has_nan", value_set=[False])
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(column="has_inf", value_set=[False])
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(column="is_all_zero", value_set=[False])
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="max_val", min_value=1e-6, max_value=10.0)
    )
    return suite


def _run_ge(
    split_name: str,
    directory: Path,
    expected_shape: tuple[int, int],
    max_sample: int = 0,
) -> bool:
    """
    Tek bir split için GE doğrulaması çalıştırır.
    Satır sayısı beklentisi yok — örneklenen gerçek sayı kullanılır.
    """
    print(f"\n  [{split_name}] istatistikler hesaplanıyor…")
    stats_df = _build_stats_df(directory, expected_shape, max_sample=max_sample)

    context = gx.get_context(mode="ephemeral")
    ds    = context.data_sources.add_pandas(name=f"ds_{split_name}")
    asset = ds.add_dataframe_asset(name=f"asset_{split_name}")
    batch_def = asset.add_batch_definition_whole_dataframe(f"batch_{split_name}")

    suite_name = f"deephorizon_{split_name}_suite"
    suite = context.suites.add(_build_suite(suite_name, expected_shape))

    vd = context.validation_definitions.add(
        gx.ValidationDefinition(
            name=f"vd_{split_name}",
            data=batch_def,
            suite=suite,
        )
    )

    result = vd.run(batch_parameters={"dataframe": stats_df})

    if result.success:
        print(f"  ✓ [{split_name}] TÜM KONTROLLER GEÇTİ  ({len(stats_df)} dosya)")
    else:
        print(f"  ✗ [{split_name}] BAZI KONTROLLER BAŞARISIZ:", file=sys.stderr)
        for er in result.results:
            if not er.success:
                print(f"      • {er.expectation_config.type}: {er.result}", file=sys.stderr)

    return bool(result.success)


def validate_training_512(max_sample: int = 0) -> bool:
    """10K x (512,512) float32 çift doğrulama."""
    base = REPO_ROOT / "data" / "training"
    clean_dir    = base / "clean"
    degraded_dir = base / "degraded"

    n_clean    = len(list(clean_dir.glob("*.npy")))
    n_degraded = len(list(degraded_dir.glob("*.npy")))
    if n_clean != n_degraded:
        print(f"  ✗ Çift sayısı eşleşmiyor: clean={n_clean}, degraded={n_degraded}", file=sys.stderr)
        return False

    ok_clean    = _run_ge("training512_clean",    clean_dir,    expected_shape=(512, 512), max_sample=max_sample)
    ok_degraded = _run_ge("training512_degraded", degraded_dir, expected_shape=(512, 512), max_sample=max_sample)
    return ok_clean and ok_degraded


def validate_synthetic_128(max_sample: int = 0) -> bool:
    """1K x (128,128) float32 çift doğrulama."""
    base = REPO_ROOT / "data" / "raw" / "simulated"
    clean_dir    = base / "clean"
    degraded_dir = base / "degraded"

    n_clean    = len(list(clean_dir.glob("*.npy")))
    n_degraded = len(list(degraded_dir.glob("*.npy")))
    if n_clean != n_degraded:
        print(f"  ✗ Çift sayısı eşleşmiyor: clean={n_clean}, degraded={n_degraded}", file=sys.stderr)
        return False

    ok_clean    = _run_ge("synthetic128_clean",    clean_dir,    expected_shape=(128, 128), max_sample=max_sample)
    ok_degraded = _run_ge("synthetic128_degraded", degraded_dir, expected_shape=(128, 128), max_sample=max_sample)
    return ok_clean and ok_degraded


def validate_eht_raw() -> bool:
    """EHT UVFITS dosya varlığı ve boyut kontrolü."""
    eht_dir = REPO_ROOT / "data" / "raw" / "eht"
    uvfits_files = list(eht_dir.rglob("*.uvfits"))
    if not uvfits_files:
        print(f"  ✗ {eht_dir} içinde .uvfits dosyası bulunamadı.", file=sys.stderr)
        return False

    expected_counts = {
        "m87_2017": 8, "3c279_2017": 8, "sgra_2017": 20,
        "m87_2018": 24, "cena_2017": 4, "m87_2017_pol": 16,
        "sgra_2017_pol": 8,
    }
    all_ok = True
    for dataset, expected in expected_counts.items():
        dataset_dir = eht_dir / dataset
        if not dataset_dir.exists():
            print(f"  ! [{dataset}] dizini yok", file=sys.stderr)
            continue
        found = len(list(dataset_dir.glob("*.uvfits")))
        if found != expected:
            print(f"  ✗ [{dataset}] beklenen {expected}, bulunan {found}", file=sys.stderr)
            all_ok = False
        else:
            print(f"  ✓ [{dataset}] {found}/{expected} dosya tamam")
    return all_ok


def run_all(sources: list[str] | None = None, max_sample: int = 0) -> None:
    if sources is None:
        sources = ["training_512", "synthetic_128", "eht_raw"]

    results: dict[str, bool] = {}

    if "training_512" in sources:
        print("\n" + "=" * 55)
        print("  Training 512×512 doğrulanıyor…")
        print("=" * 55)
        results["training_512"] = validate_training_512(max_sample=max_sample)

    if "synthetic_128" in sources:
        print("\n" + "=" * 55)
        print("  Synthetic 128×128 doğrulanıyor…")
        print("=" * 55)
        results["synthetic_128"] = validate_synthetic_128(max_sample=max_sample)

    if "eht_raw" in sources:
        print("\n" + "=" * 55)
        print("  EHT UVFITS dosya sayımı doğrulanıyor…")
        print("=" * 55)
        results["eht_raw"] = validate_eht_raw()

    print("\n" + "─" * 55)
    failed = [k for k, v in results.items() if not v]
    if failed:
        raise RuntimeError(f"GE doğrulama BAŞARISIZ — kaynaklar: {', '.join(failed)}")
    print("  ✓ Tüm doğrulamalar başarılı!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepHorizon GE doğrulama suite'i")
    parser.add_argument(
        "--source", nargs="*",
        default=["training_512", "synthetic_128", "eht_raw"],
        choices=["training_512", "synthetic_128", "eht_raw"],
    )
    parser.add_argument(
        "--max-sample", type=int, default=0,
        help="Örneklenecek max dosya sayısı (0=hepsi)",
    )
    args = parser.parse_args()
    run_all(sources=args.source, max_sample=args.max_sample)
