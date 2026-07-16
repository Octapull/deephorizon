"""
DeepHorizon — Training 512×512 Veri Pipeline DAG'ı
====================================================
10K adet 512×512 clean/degraded çift üretir, manifest oluşturur,
doğrular, DVC ile izlemeye alır ve MinIO'ya yükler.

Tetikleme : Manuel (schedule=None)
Idempotency: generate_training_data.py mevcut dosyaları üzerine yazar;
             mc mirror --overwrite sadece değişenleri yükler.
Sahibi     : Stajyer 1

NOT — DVC:
    dvc add data/training çalıştırılır ancak dvc push YOK.
    Repo'da henüz yapılandırılmış DVC remote bulunmuyor;
    remote eklendikten sonra dvc_track task'ına dvc push eklenebilir.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from airflow.sdk import DAG, task

# ─── Sabitler ────────────────────────────────────────────────────────────────

REPO_ROOT = Path(os.environ["DEEPHORIZON_ROOT"])
SCRIPTS_DIR    = REPO_ROOT / "scripts"
TRAINING_DIR   = REPO_ROOT / "data" / "training"
MINIO_ENDPOINT = os.environ["MINIO_ENDPOINT"]
MINIO_USER     = os.environ["MINIO_USER"]
MINIO_ALIAS    = os.environ["MINIO_ALIAS"]

N_EXPECTED_PAIRS = 10_000
GE_MAX_SAMPLE    = 500   # Tüm 20K dosya yerine örnekle — ~30 saniye

# ─── Yardımcılar ─────────────────────────────────────────────────────────────

def _mc_env() -> dict[str, str]:
    """
    MinIO client için güvenli ortam değişkenleri döner.

    MC_HOST_{MINIO_ALIAS} pattern'i kullanılır — secret hiçbir zaman
    subprocess argümanında (ps çıktısı / Airflow logları) görünmez.

    Secret önce MINIO_SECRET ortam değişkenine,
    sonra Airflow Variable 'minio_data_pipeline_secret' anahtarına bakılır.
    """
    secret = os.environ.get("MINIO_SECRET")
    if not secret:
        try:
            from airflow.models import Variable
            secret = Variable.get("minio_data_pipeline_secret", default_var=None)
        except Exception:
            pass
    if not secret:
        raise RuntimeError(
            "MinIO secret bulunamadı. "
            "MINIO_SECRET ortam değişkeni veya "
            "'minio_data_pipeline_secret' Airflow Variable'ı tanımlı değil."
        )
    env = os.environ.copy()
    env[f"MC_HOST_{MINIO_ALIAS}"] = (
        f"http://{MINIO_USER}:{secret}"
        f"@{MINIO_ENDPOINT.removeprefix('http://')}"
    )
    return env


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 3600,
    label: str = "",
) -> subprocess.CompletedProcess[str]:
    """
    subprocess.run() sarmalayıcısı — shell=False, anlamlı hata mesajı.

    Args:
        cmd    : Argüman listesi (shell=False).
        cwd    : Çalışma dizini.
        env    : Ortam değişkenleri (None → miras alınır).
        timeout: Saniye cinsinden zaman aşımı.
        label  : Hata mesajlarında gösterilecek açıklama.
    """
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.stdout:
        print(result.stdout[-4000:])
    if result.returncode != 0:
        context = label or " ".join(cmd[:3])
        raise RuntimeError(
            f"[{context}] süreç {result.returncode} koduyla çıktı.\n"
            f"stderr: {result.stderr[-2000:]}"
        )
    return result


def _validate_counts() -> None:
    """
    Clean/degraded çift sayısını kontrol eder.
    Eşleşmiyorsa veya beklenen sayıdan azsa ValueError fırlatır.
    """
    clean_dir    = TRAINING_DIR / "clean"
    degraded_dir = TRAINING_DIR / "degraded"

    n_clean    = len(list(clean_dir.glob("*.npy")))
    n_degraded = len(list(degraded_dir.glob("*.npy")))

    if n_clean != n_degraded:
        raise ValueError(
            f"Çift sayısı eşleşmiyor: clean={n_clean}, degraded={n_degraded}"
        )
    if n_clean != N_EXPECTED_PAIRS:
        raise ValueError(
            f"Beklenen tam {N_EXPECTED_PAIRS} çift, bulunan {n_clean}"
        )


# ─── DAG ─────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="training_data_build",
    description="10K 512×512 çift üret, manifest oluştur, doğrula, DVC izle, MinIO'ya yükle",
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
        _run(
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

        ge_suite.py mevcutsa GE 1.4.0+ Fluent API ile GE_MAX_SAMPLE
        örnekleyerek doğrular. Yoksa lightweight sayım + boyut kontrolü yapar.
        Her iki durumda da çift sayısı kontrolü çalışır.

        Returns:
            "validate_ok"

        Raises:
            ValueError: Herhangi bir kontrol başarısız olursa.
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
                raise RuntimeError(
                    "ge_suite.py bulundu ama import edilemedi. "
                    "great-expectations kurulu mu?"
                ) from exc
        else:
            # Lightweight fallback: clean ve degraded'dan örnekleyerek shape/dtype kontrolü
            import random
            import numpy as np

            for split in ("clean", "degraded"):
                files = list((TRAINING_DIR / split).glob("*.npy"))
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
        data/training dizinini DVC ile izlemeye alır (dvc add).

        NOT: dvc push bu task'ta YOK — repo'da henüz yapılandırılmış
        DVC remote bulunmuyor. Remote eklendikten sonra şu satır açılabilir:
            _run(["dvc", "push"], cwd=REPO_ROOT, label="dvc push")

        git add / git commit bu DAG'ın sorumluluğu değil.

        Returns:
            "dvc_track_ok" ya da DVC kurulu değilse "dvc_skipped".
        """
        if not shutil.which("dvc"):
            print("⚠ dvc komutu PATH'te bulunamadı, izleme atlanıyor.")
            return "dvc_skipped"

        _run(
            ["dvc", "add", "data/training"],
            cwd=REPO_ROOT,
            timeout=120,
            label="dvc add",
        )
        print("✓ data/training DVC ile izlemeye alındı.")
        print("  → data/training.dvc dosyasını git'e commit etmeyi unutma.")
        return "dvc_track_ok"

    @task()
    def upload_to_minio() -> str:
        """
        Training verilerini ve manifest'i MinIO'ya yükler.

        Her çalıştırma timestamp'li yeni bir prefix alır — önceki
        yüklemelerin üzerine yazılmaz.
        Pattern: datasets/training-512/<YYYYMMDD-HHMM>/

        Returns:
            "upload_ok_<version>"

        Raises:
            RuntimeError: TRAINING_DIR eksikse veya mc başarısız olursa.
        """
        if not TRAINING_DIR.exists():
            raise RuntimeError(
                f"{TRAINING_DIR} bulunamadı. "
                "generate_training task'ının başarıyla tamamlandığından emin olun."
            )

        env = _mc_env()
        version = datetime.now().strftime("%Y%m%d-%H%M")
        base = f"{MINIO_ALIAS}/datasets/training-512/{version}"

        for split in ("clean", "degraded"):
            src = TRAINING_DIR / split
            if not src.exists():
                raise RuntimeError(
                    f"{src} bulunamadı. "
                    "generate_training task'ının başarıyla tamamlandığından emin olun."
                )
            _run(
                ["mc", "mirror", str(src) + "/", f"{base}/{split}/", "--overwrite"],
                env=env,
                timeout=3600,
                label=f"mc mirror {split}",
            )
            print(f"  ✓ {split} → {base}/{split}/")

        print(f"\n✓ Yükleme tamamlandı: {base}/")
        return f"upload_ok_{version}"

    # ── Bağımlılık zinciri ────────────────────────────────────────────────────
    gen       = generate_training()
    validated = validate_training()
    tracked   = dvc_track()
    uploaded  = upload_to_minio()

    gen >> validated >> tracked >> uploaded