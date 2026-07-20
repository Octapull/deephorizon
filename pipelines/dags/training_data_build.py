"""
DeepHorizon — Training 512×512 Veri Pipeline DAG'ı
====================================================
10K adet 512×512 clean/degraded çift üretir, doğrular,
DVC ile izlemeye alır ve MinIO'ya yükler.

Tetikleme : Manuel (schedule=None)
Idempotency: generate_training_data.py mevcut dosyaları üzerine yazar;
             mc mirror --overwrite sadece değişenleri yükler.
Sahibi     : Stajyer 1

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
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

from airflow.sdk import DAG, task
from deephorizon_utils import dvc_env, mc_env, run

# ─── Sabitler ────────────────────────────────────────────────────────────────

REPO_ROOT    = Path(os.environ["DEEPHORIZON_ROOT"])
SCRIPTS_DIR  = REPO_ROOT / "scripts"
TRAINING_DIR = REPO_ROOT / "data" / "training"
MINIO_ALIAS  = os.environ["MINIO_ALIAS"]

N_EXPECTED_PAIRS = 10_000
GE_MAX_SAMPLE    = 500

# ─── Yardımcılar ─────────────────────────────────────────────────────────────

def _validate_counts() -> None:
    clean_dir    = TRAINING_DIR / "clean"
    degraded_dir = TRAINING_DIR / "degraded"

    n_clean    = len(list(clean_dir.glob("*.npy")))
    n_degraded = len(list(degraded_dir.glob("*.npy")))

    if n_clean != n_degraded:
        raise ValueError(f"Çift sayısı eşleşmiyor: clean={n_clean}, degraded={n_degraded}")
    if n_clean != N_EXPECTED_PAIRS:
        raise ValueError(f"Beklenen {N_EXPECTED_PAIRS} çift, bulunan {n_clean}")


# ─── DAG ─────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="training_data_build",
    description="10K 512×512 çift üret, doğrula, DVC'ye ekle, MinIO'ya yükle",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["data", "training", "stajyer1"],
) as dag:

    @task()
    def generate_training() -> str:
        """
        scripts/generate_training_data.py'yi çalıştırır.
        10K adet 512×512 clean/degraded .npy çifti üretir.
        Idempotent: mevcut dosyaların üzerine yazar.

        Returns:
            "generate_ok"
        """
        run(
            [sys.executable, str(SCRIPTS_DIR / "generate_training_data.py")],
            cwd=REPO_ROOT,
            timeout=7200,
            label="generate_training_data",
        )
        return "generate_ok"

    @task()
    def validate_training() -> str:
        """
        Training verilerini doğrular.
        GE kuruluysa GE_MAX_SAMPLE örnekleyerek doğrular;
        değilse 50'şer örnekle lightweight kontrol yapar.

        Returns:
            "validate_ok"
        """
        _validate_counts()

        ge_suite_path = SCRIPTS_DIR / "ge_suite.py"
        if ge_suite_path.exists():
            scripts_str = str(SCRIPTS_DIR)
            if scripts_str not in sys.path:
                sys.path.insert(0, scripts_str)
            try:
                from ge_suite import validate_training_512  # type: ignore[import]
                ok = validate_training_512(max_sample=GE_MAX_SAMPLE)
                if not ok:
                    raise ValueError("GE doğrulaması başarısız.")
                print(f"✓ GE doğrulaması başarılı ({GE_MAX_SAMPLE} örnekle)")
            except ImportError as exc:
                raise RuntimeError("ge_suite.py import edilemedi. great-expectations kurulu mu?") from exc
        else:
            import numpy as np

            for split in ("clean", "degraded"):
                files  = list((TRAINING_DIR / split).glob("*.npy"))
                sample = random.sample(files, min(50, len(files)))
                for fpath in sample:
                    arr = np.load(fpath, allow_pickle=False)
                    if arr.shape != (512, 512):
                        raise ValueError(f"{split}/{fpath.name}: beklenmeyen shape {arr.shape}")
                    if arr.dtype != np.float32:
                        raise ValueError(f"{split}/{fpath.name}: beklenmeyen dtype {arr.dtype}")
                    if arr.min() < 0:
                        raise ValueError(f"{split}/{fpath.name}: negatif değer var")
            print("✓ Lightweight doğrulama başarılı (clean + degraded, 50'şer örnekle)")

        return "validate_ok"

    @task()
    def dvc_track() -> str:
        """
        data/training dizinini DVC ile izlemeye alır ve uzak depoya yükler.

        NOT: 10K çift × 2 = ~20 GiB. dvc push uzun sürebilir (timeout=7200).
        MinIO dvc-cache ve eğitim seti aynı fiziksel diskte — disk doluluk
        kontrolü için 'mc admin info dh' çıktısına bakın.

        Returns:
            "dvc_ok" ya da "dvc_skipped"
        """
        if not shutil.which("dvc"):
            print("⚠ dvc komutu PATH'te bulunamadı, izleme atlanıyor.")
            return "dvc_skipped"

        run(
            ["dvc", "add", "data/training"],
            cwd=REPO_ROOT,
            timeout=120,
            label="dvc add",
        )
        print("✓ data/training DVC'ye eklendi.")

        run(
            ["dvc", "push"],
            cwd=REPO_ROOT,
            env=dvc_env(),
            timeout=7200,   # 20 GiB için geniş tutuluyor
            label="dvc push",
        )
        print("✓ dvc push tamamlandı → dvc-cache bucket'ına yüklendi.")
        print("  → data/training.dvc dosyasını git'e commit etmeyi unutma.")
        return "dvc_ok"

    @task()
    def upload_to_minio() -> str:
        """
        Training verilerini MinIO datasets/training-512/<timestamp>/'ye yükler.
        Her çalıştırma yeni timestamp prefix alır — eski sürümler korunur.

        Returns:
            "upload_ok_<version>"
        """
        if not TRAINING_DIR.exists():
            raise RuntimeError(f"{TRAINING_DIR} bulunamadı.")

        env         = mc_env()
        version_tag = datetime.now().strftime("%Y%m%d-%H%M")
        base        = f"{MINIO_ALIAS}/datasets/training-512/{version_tag}"

        for split in ("clean", "degraded"):
            src = TRAINING_DIR / split
            if not src.exists():
                raise RuntimeError(f"{src} bulunamadı.")
            run(
                ["mc", "mirror", str(src) + "/", f"{base}/{split}/", "--overwrite"],
                env=env,
                timeout=3600,
                label=f"mc mirror {split}",
            )
            print(f"  ✓ {split} → {base}/{split}/")

        print(f"\n✓ Yükleme tamamlandı: {base}/")
        return f"upload_ok_{version_tag}"

    # ── Bağımlılık zinciri ────────────────────────────────────────────────────
    gen       = generate_training()
    validated = validate_training()
    tracked   = dvc_track()
    uploaded  = upload_to_minio()

    gen >> validated >> tracked >> uploaded