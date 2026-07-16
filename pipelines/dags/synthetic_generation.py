"""
DeepHorizon — Sentetik 128×128 Veri Üretim DAG'ı
==================================================
eht-imaging kütüphanesi ile fiziğe dayalı 128×128 clean/degraded çiftleri
üretir. Bu set hızlı mimari deneyleri için kullanılır (1K çift, ~100 MB).

Tetikleme : Manuel (schedule=None)
Idempotency: generate_synthetic_data.py mevcut dosyaların üzerine yazar;
             mc mirror --overwrite sadece değişenleri yükler.
Sahibi     : Stajyer 2

NOT — DVC:
    dvc add data/raw/simulated çalıştırılır ancak dvc push YOK.
    Repo'da henüz yapılandırılmış DVC remote bulunmuyor;
    remote eklendikten sonra dvc_track task'ına dvc push eklenebilir.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from airflow.sdk import DAG, task

# ─── Sabitler ────────────────────────────────────────────────────────────────

REPO_ROOT = Path(os.environ["DEEPHORIZON_ROOT"])
SCRIPTS_DIR    = REPO_ROOT / "scripts"
SIMULATED_DIR  = REPO_ROOT / "data" / "raw" / "simulated"
MINIO_ENDPOINT = os.environ["MINIO_ENDPOINT"]
MINIO_USER     = os.environ["MINIO_USER"]
MINIO_ALIAS    = os.environ["MINIO_ALIAS"]

N_EXPECTED_PAIRS = 1_000
GE_MAX_SAMPLE    = 200

_FILENAME_RE = re.compile(r"^(crescent|ring)_(light|medium|heavy|extreme)_")

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


# ─── DAG ─────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="synthetic_generation",
    description="ehtim ile 128×128 1K sentetik çift üret, doğrula, MinIO'ya yükle",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["data", "synthetic", "stajyer2"],
) as dag:

    @task()
    def check_ehtim() -> str:
        """
        ehtim kütüphanesinin kurulu olduğunu doğrular.

        Returns:
            "ehtim_ok"

        Raises:
            RuntimeError: ehtim kurulu değilse.
        """
        try:
            import ehtim  # noqa: F401
            print(f"✓ ehtim {ehtim.__version__} kurulu")
        except ImportError as exc:
            raise RuntimeError(
                "ehtim bulunamadı. "
                "`pip install ehtim>=1.2.10` veya requirements/data.txt kurun."
            ) from exc
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
        _run(
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

        ge_suite.py mevcutsa GE_MAX_SAMPLE örnekleyerek doğrular.
        Yoksa clean + degraded'dan örnekleyerek lightweight kontrol yapar.
        Her iki durumda da çift sayısı ve dosya adı formatı kontrol edilir.

        Returns:
            "validate_ok"

        Raises:
            ValueError: Herhangi bir kontrol başarısız olursa.
        """
        clean_dir    = SIMULATED_DIR / "clean"
        degraded_dir = SIMULATED_DIR / "degraded"

        # ── Çift sayısı kontrolü ─────────────────────────────────────────────
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

        # ── GE veya lightweight doğrulama ────────────────────────────────────
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
                raise RuntimeError(
                    "ge_suite.py bulundu ama import edilemedi. "
                    "great-expectations kurulu mu?"
                ) from exc
        else:
            import random
            import numpy as np

            for split_dir in (clean_dir, degraded_dir):
                files  = list(split_dir.glob("*.npy"))
                sample = random.sample(files, min(50, len(files)))
                for fpath in sample:
                    arr = np.load(fpath, allow_pickle=False)
                    if arr.shape != (128, 128):
                        raise ValueError(
                            f"{split_dir.name}/{fpath.name}: "
                            f"beklenmeyen shape {arr.shape}"
                        )
                    if arr.dtype != np.float32:
                        raise ValueError(
                            f"{split_dir.name}/{fpath.name}: "
                            f"beklenmeyen dtype {arr.dtype}"
                        )
                    if arr.min() < 0:
                        raise ValueError(
                            f"{split_dir.name}/{fpath.name}: negatif değer var"
                        )
            print("✓ Lightweight doğrulama başarılı (clean + degraded, 50'şer örnekle)")

        # ── Dağılım logu ─────────────────────────────────────────────────────
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
        Sentetik veriyi MinIO raw/simulated-128/<version>/'ye yükler.

        Her çalıştırma timestamp'li yeni bir prefix alır.
        pairs/ opsiyonel PNG önizlemeleri içerir; eksikse atlanır.
        Manifest zorunludur; eksikse RuntimeError fırlatır.

        Returns:
            "upload_ok_<version>"
        """
        if not SIMULATED_DIR.exists():
            raise RuntimeError(
                f"{SIMULATED_DIR} bulunamadı. "
                "generate_synthetic task'ının başarıyla tamamlandığından emin olun."
            )

        env     = _mc_env()
        version = datetime.now().strftime("%Y%m%d-%H%M")
        base    = f"{MINIO_ALIAS}/raw/simulated-128/{version}"

        # clean ve degraded zorunlu; pairs opsiyonel (PNG önizlemeler)
        for split in ("clean", "degraded"):
            split_dir = SIMULATED_DIR / split
            if not split_dir.exists():
                raise RuntimeError(
                    f"{split_dir} bulunamadı. "
                    "generate_synthetic task'ının başarıyla tamamlandığından emin olun."
                )
            _run(
                ["mc", "mirror", str(split_dir) + "/",
                 f"{base}/{split}/", "--overwrite"],
                env=env,
                timeout=1800,
                label=f"mc mirror {split}",
            )
            print(f"  ✓ {split} → {base}/{split}/")

        pairs_dir = SIMULATED_DIR / "pairs"
        if pairs_dir.exists():
            _run(
                ["mc", "mirror", str(pairs_dir) + "/",
                 f"{base}/pairs/", "--overwrite"],
                env=env,
                timeout=300,
                label="mc mirror pairs",
            )
            print(f"  ✓ pairs → {base}/pairs/")

        print(f"\n✓ Yükleme tamamlandı: {base}/")
        return f"upload_ok_{version}"

    @task()
    def dvc_track() -> str:
        """
        data/raw/simulated dizinini DVC ile izlemeye alır (dvc add).

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
            ["dvc", "add", "data/raw/simulated"],
            cwd=REPO_ROOT,
            timeout=120,
            label="dvc add",
        )
        print("✓ data/raw/simulated DVC ile izlemeye alındı.")
        print("  → data/raw/simulated.dvc dosyasını git'e commit etmeyi unutma.")
        return "dvc_track_ok"

    # ── Bağımlılık zinciri ────────────────────────────────────────────────────
    ehtim_ok  = check_ehtim()
    generated = generate_synthetic()
    validated = validate_synthetic()
    uploaded  = upload_to_minio()
    tracked   = dvc_track()

    ehtim_ok >> generated >> validated >> uploaded >> tracked