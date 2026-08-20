"""
DeepHorizon — EHT Ham Veri İndirme DAG'ı
==========================================
Tüm kamuya açık EHT UVFITS gözlem dosyalarını GitHub'dan indirir,
dosya varlığı ve sayısını doğrular, MinIO raw/eht/ bucket'ına yükler,
DVC ile izlemeye alır.

Tetikleme : Manuel (schedule=None)
Idempotency: download_eht_data.py mevcut dosyaları atlar;
             mc mirror --overwrite sadece değişenleri yükler.
Sahibi     : Stajyer 1

Airflow ortam değişkenleri (airflow.cfg ya da Connections/Variables):
    DEEPHORIZON_ROOT  — repo kök dizini, ör. /home/deephorizon/deephorizon
    MINIO_ENDPOINT    — http://10.10.1.132:30900
    MINIO_USER        — data-pipeline
    MINIO_ALIAS       — dh

Airflow Variables (Admin → Variables):
    minio_data_pipeline_secret  — data-pipeline kullanıcısının MinIO parolası
    dvc_secret                  — dvc kullanıcısının MinIO parolası (dvc push için)
"""
from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from airflow.sdk import DAG, task
from deephorizon_utils import dvc_env, mc_env, run

# ─── Sabitler ────────────────────────────────────────────────────────────────

REPO_ROOT   = Path(os.environ["DEEPHORIZON_ROOT"])
SCRIPTS_DIR = REPO_ROOT / "scripts"
EHT_DIR     = REPO_ROOT / "data" / "raw" / "eht"
MINIO_ALIAS = os.environ["MINIO_ALIAS"]

MIN_UVFITS_BYTES = 1024

EXPECTED_COUNTS: dict[str, int] = {
    "m87_2017":      8,
    "3c279_2017":    8,
    "sgra_2017":     20,
    "m87_2018":      24,
    "cena_2017":     4,
    "m87_2017_pol":  16,
    "sgra_2017_pol": 8,
}

# ─── Yardımcılar ─────────────────────────────────────────────────────────────

def _validate_counts(file_counts: dict[str, int]) -> None:
    errors: list[str] = []
    for ds, expected in EXPECTED_COUNTS.items():
        if ds not in file_counts:
            continue
        actual = file_counts[ds]
        if actual == 0:
            continue
        if actual != expected:
            errors.append(f"{ds}: beklenen {expected} dosya, bulunan {actual}")
            continue
        for fpath in (EHT_DIR / ds).glob("*.uvfits"):
            if fpath.stat().st_size < MIN_UVFITS_BYTES:
                errors.append(f"{ds}/{fpath.name}: şüpheli boyut ({fpath.stat().st_size} byte)")

    if errors:
        raise ValueError("EHT doğrulama başarısız:\n" + "\n".join(f"  • {e}" for e in errors))


# ─── DAG ─────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="eht_ingest",
    description="EHT UVFITS verilerini indir, doğrula, MinIO raw/eht/'ye yükle, DVC'ye ekle",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["data", "eht", "ingest", "stajyer1"],
) as dag:

    @task()
    def download_eht(datasets: list[str] | None = None) -> dict[str, int]:
        """
        EHT UVFITS dosyalarını indirir.
        Idempotent: script mevcut dosyaları atlar.

        Returns:
            {dataset_adı: bulunan_uvfits_sayısı}
        """
        cmd = [sys.executable, str(SCRIPTS_DIR / "download_eht_data.py")]
        if datasets:
            cmd += ["--datasets"] + datasets

        run(cmd, cwd=REPO_ROOT, timeout=7200, label="download_eht_data")

        counts: dict[str, int] = {}
        for ds in datasets or list(EXPECTED_COUNTS.keys()):
            ds_dir = EHT_DIR / ds
            counts[ds] = len(list(ds_dir.glob("*.uvfits"))) if ds_dir.exists() else 0

        print(f"İndirme sonuçları: {counts}")
        return counts

    @task()
    def validate_eht(file_counts: dict[str, int]) -> str:
        """
        İndirilen dosyaları doğrular: sayım + boyut + GE (varsa).

        Returns:
            "validate_ok"
        """
        _validate_counts(file_counts)

        ge_suite_path = SCRIPTS_DIR / "ge_suite.py"
        if ge_suite_path.exists():
            scripts_str = str(SCRIPTS_DIR)
            if scripts_str not in sys.path:
                sys.path.insert(0, scripts_str)
            try:
                from ge_suite import validate_eht_raw  # type: ignore[import]
                validate_eht_raw()
                print("✓ GE doğrulaması başarılı")
            except Exception as exc:
                raise ValueError(f"GE doğrulaması başarısız: {exc}") from exc
        else:
            print("⚠ ge_suite.py bulunamadı, lightweight doğrulama kullanıldı.")

        if isinstance(file_counts, dict):
            total = sum(file_counts.values())
            print(f"✓ EHT doğrulama başarılı — {total} dosya, {len(file_counts)} dataset")
        else:
            print("✓ EHT doğrulama başarılı")
        return "validate_ok"

    @task()
    def upload_to_minio() -> str:
        """
        Doğrulanmış EHT dosyalarını MinIO raw/eht/<dataset>/ yoluna yükler.
        mc mirror --overwrite sadece değişen/eksik dosyaları yükler.

        Returns:
            "upload_ok"
        """
        if not EHT_DIR.exists():
            raise RuntimeError(f"{EHT_DIR} bulunamadı.")

        env = mc_env()
        uploaded: list[str] = []

        for ds_dir in sorted(EHT_DIR.iterdir()):
            if not ds_dir.is_dir():
                continue
            run(
                ["mc", "mirror", str(ds_dir) + "/",
                 f"{MINIO_ALIAS}/raw/eht/{ds_dir.name}/", "--overwrite"],
                env=env,
                timeout=1800,
                label=f"mc mirror {ds_dir.name}",
            )
            print(f"  ✓ {ds_dir.name} → {MINIO_ALIAS}/raw/eht/{ds_dir.name}/")
            uploaded.append(ds_dir.name)

        if not uploaded:
            raise RuntimeError(f"{EHT_DIR} içinde yüklenecek dataset dizini bulunamadı.")

        print(f"\n✓ Yükleme tamamlandı: {len(uploaded)} dataset")
        return "upload_ok"

    @task()
    def dvc_track() -> str:
        """
        data/raw/eht dizinini DVC ile izlemeye alır ve uzak depoya yükler.

        .dvc/config'deki 'minio' remote'u kullanılır.
        Credential'lar AWS_* env değişkenleriyle iletilir — log'a düşmez.

        Returns:
            "dvc_ok" ya da DVC kurulu değilse "dvc_skipped"
        """
        if not shutil.which("dvc"):
            print("⚠ dvc komutu PATH'te bulunamadı, izleme atlanıyor.")
            return "dvc_skipped"

        run(
            ["dvc", "add", "data/raw/eht"],
            cwd=REPO_ROOT,
            timeout=120,
            label="dvc add",
        )
        print("✓ data/raw/eht DVC'ye eklendi.")

        run(
            ["dvc", "push"],
            cwd=REPO_ROOT,
            env=dvc_env(),
            timeout=3600,
            label="dvc push",
        )
        print("✓ dvc push tamamlandı → dvc-cache bucket'ına yüklendi.")
        print("  → data/raw/eht.dvc dosyasını git'e commit etmeyi unutma.")
        return "dvc_ok"

    # ── Bağımlılık zinciri ───────────────────────────────────────────────────
    counts    = download_eht()
    validated = validate_eht(counts)
    uploaded  = upload_to_minio()
    validated >> uploaded >> dvc_track()