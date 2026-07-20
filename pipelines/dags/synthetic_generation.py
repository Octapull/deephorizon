"""
DeepHorizon — Sentetik 128×128 Veri Üretim DAG'ı
==================================================
eht-imaging kütüphanesi ile fiziğe dayalı 128×128 clean/degraded çiftleri
üretir (1K çift, ~100 MB), doğrular, MinIO'ya yükler, DVC ile izler.

Tetikleme : Manuel (schedule=None)
Idempotency: generate_synthetic_data.py mevcut dosyaların üzerine yazar;
             mc mirror --overwrite sadece değişenleri yükler.
Sahibi     : Stajyer 2

Airflow ortam değişkenleri:
    DEEPHORIZON_ROOT  — repo kök dizini
    MINIO_ENDPOINT    — http://10.10.1.132:30900
    MINIO_USER        — data-pipeline
    MINIO_ALIAS       — dh

Airflow Variables:
    minio_data_pipeline_secret
    dvc_secret
"""
from __future__ import annotations

import os
import re
import shutil
import sys
from collections import Counter
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from airflow.sdk import DAG, task
from deephorizon_utils import dvc_env, mc_env, run

# ─── Sabitler ────────────────────────────────────────────────────────────────

REPO_ROOT     = Path(os.environ["DEEPHORIZON_ROOT"])
SCRIPTS_DIR   = REPO_ROOT / "scripts"
SIMULATED_DIR = REPO_ROOT / "data" / "raw" / "simulated"
MINIO_ALIAS   = os.environ["MINIO_ALIAS"]

N_EXPECTED_PAIRS = 1_000
GE_MAX_SAMPLE    = 200

_FILENAME_RE = re.compile(r"^(crescent|ring)_(light|medium|heavy|extreme)_")

# ─── DAG ─────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="synthetic_generation",
    description="ehtim ile 128×128 1K sentetik çift üret, doğrula, MinIO'ya yükle, DVC'ye ekle",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["data", "synthetic", "stajyer2"],
) as dag:

    @task()
    def check_ehtim() -> str:
        """
        ehtim kütüphanesinin kurulu olduğunu doğrular.

        ehtim.__version__ attribute'u olmadığı için importlib.metadata
        üzerinden sürüm okunur (bkz. services/ml/_compat.py).

        Returns:
            "ehtim_ok"
        """
        try:
            import ehtim  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "ehtim bulunamadı. "
                "`uv sync --extra data` veya `pip install ehtim>=1.2.10`"
            ) from exc

        try:
            ehtim_ver = version("ehtim")
        except PackageNotFoundError:
            ehtim_ver = "unknown"

        print(f"✓ ehtim {ehtim_ver} kurulu")
        return "ehtim_ok"

    @task()
    def generate_synthetic() -> str:
        """
        scripts/generate_synthetic_data.py'yi çalıştırır.
        1K adet 128×128 clean/degraded .npy çifti ve PNG önizlemeleri üretir.
        Idempotent: mevcut dosyaların üzerine yazar.

        Returns:
            "generate_ok"
        """
        run(
            [sys.executable, str(SCRIPTS_DIR / "generate_synthetic_data.py")],
            cwd=REPO_ROOT,
            timeout=3600,
            label="generate_synthetic_data",
        )
        return "generate_ok"

    @task()
    def validate_synthetic() -> str:
        """
        Üretilen 128×128 sentetik veriyi doğrular.
        GE kuruluysa örnekleyerek doğrular; değilse lightweight kontrol yapar.

        Returns:
            "validate_ok"
        """
        clean_dir    = SIMULATED_DIR / "clean"
        degraded_dir = SIMULATED_DIR / "degraded"

        n_clean    = len(list(clean_dir.glob("*.npy")))
        n_degraded = len(list(degraded_dir.glob("*.npy")))

        if n_clean != n_degraded:
            raise ValueError(f"Çift sayısı eşleşmiyor: clean={n_clean}, degraded={n_degraded}")
        if n_clean != N_EXPECTED_PAIRS:
            raise ValueError(f"Beklenen {N_EXPECTED_PAIRS} çift, bulunan {n_clean}")

        ge_suite_path = SCRIPTS_DIR / "ge_suite.py"
        if ge_suite_path.exists():
            scripts_str = str(SCRIPTS_DIR)
            if scripts_str not in sys.path:
                sys.path.insert(0, scripts_str)
            try:
                from ge_suite import validate_synthetic_128  # type: ignore[import]
                ok = validate_synthetic_128(max_sample=GE_MAX_SAMPLE)
                if not ok:
                    raise ValueError("GE doğrulaması başarısız.")
                print(f"✓ GE doğrulaması başarılı ({GE_MAX_SAMPLE} örnekle)")
            except ImportError as exc:
                raise RuntimeError("ge_suite.py import edilemedi. great-expectations kurulu mu?") from exc
        else:
            import random
            import numpy as np

            for split_dir in (clean_dir, degraded_dir):
                files  = list(split_dir.glob("*.npy"))
                sample = random.sample(files, min(50, len(files)))
                for fpath in sample:
                    arr = np.load(fpath, allow_pickle=False)
                    if arr.shape != (128, 128):
                        raise ValueError(f"{split_dir.name}/{fpath.name}: beklenmeyen shape {arr.shape}")
                    if arr.dtype != np.float32:
                        raise ValueError(f"{split_dir.name}/{fpath.name}: beklenmeyen dtype {arr.dtype}")
                    if arr.min() < 0:
                        raise ValueError(f"{split_dir.name}/{fpath.name}: negatif değer var")
            print("✓ Lightweight doğrulama başarılı")

        model_counts: Counter[str] = Counter()
        level_counts: Counter[str] = Counter()
        for fpath in clean_dir.glob("*.npy"):
            m = _FILENAME_RE.match(fpath.name)
            if m:
                model_counts[m.group(1)] += 1
                level_counts[m.group(2)] += 1
        print(f"  Model dağılımı  : {dict(model_counts)}")
        print(f"  Seviye dağılımı : {dict(level_counts)}")

        return "validate_ok"

    @task()
    def upload_to_minio() -> str:
        """
        Sentetik veriyi MinIO raw/simulated-128/<timestamp>/'ye yükler.
        Her çalıştırma timestamp'li yeni prefix alır.
        pairs/ opsiyonel PNG önizlemeleri; eksikse atlanır.

        Returns:
            "upload_ok_<version>"
        """
        if not SIMULATED_DIR.exists():
            raise RuntimeError(f"{SIMULATED_DIR} bulunamadı.")

        env     = mc_env()
        version_tag = datetime.now().strftime("%Y%m%d-%H%M")
        base    = f"{MINIO_ALIAS}/raw/simulated-128/{version_tag}"

        for split in ("clean", "degraded"):
            split_dir = SIMULATED_DIR / split
            if not split_dir.exists():
                raise RuntimeError(f"{split_dir} bulunamadı.")
            run(
                ["mc", "mirror", str(split_dir) + "/", f"{base}/{split}/", "--overwrite"],
                env=env,
                timeout=1800,
                label=f"mc mirror {split}",
            )
            print(f"  ✓ {split} → {base}/{split}/")

        pairs_dir = SIMULATED_DIR / "pairs"
        if pairs_dir.exists():
            run(
                ["mc", "mirror", str(pairs_dir) + "/", f"{base}/pairs/", "--overwrite"],
                env=env,
                timeout=300,
                label="mc mirror pairs",
            )
            print(f"  ✓ pairs → {base}/pairs/")

        print(f"\n✓ Yükleme tamamlandı: {base}/")
        return f"upload_ok_{version_tag}"

    @task()
    def dvc_track() -> str:
        """
        data/raw/simulated dizinini DVC ile izlemeye alır ve uzak depoya yükler.

        Returns:
            "dvc_ok" ya da "dvc_skipped"
        """
        if not shutil.which("dvc"):
            print("⚠ dvc komutu PATH'te bulunamadı, izleme atlanıyor.")
            return "dvc_skipped"

        run(
            ["dvc", "add", "data/raw/simulated"],
            cwd=REPO_ROOT,
            timeout=120,
            label="dvc add",
        )
        print("✓ data/raw/simulated DVC'ye eklendi.")

        run(
            ["dvc", "push"],
            cwd=REPO_ROOT,
            env=dvc_env(),
            timeout=1800,
            label="dvc push",
        )
        print("✓ dvc push tamamlandı → dvc-cache bucket'ına yüklendi.")
        print("  → data/raw/simulated.dvc dosyasını git'e commit etmeyi unutma.")
        return "dvc_ok"

    # ── Bağımlılık zinciri ────────────────────────────────────────────────────
    ehtim_ok  = check_ehtim()
    generated = generate_synthetic()
    validated = validate_synthetic()
    uploaded  = upload_to_minio()
    tracked   = dvc_track()

    ehtim_ok >> generated >> validated >> uploaded >> tracked