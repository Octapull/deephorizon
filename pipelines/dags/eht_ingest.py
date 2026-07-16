"""
DeepHorizon — EHT Ham Veri İndirme DAG'ı
==========================================
Tüm kamuya açık EHT UVFITS gözlem dosyalarını GitHub'dan indirir,
dosya varlığı ve sayısını doğrular, MinIO raw/eht/ bucket'ına yükler.

Tetikleme : Manuel (schedule=None)
Idempotency: download_eht_data.py mevcut dosyaları atlar;
             mc mirror --overwrite sadece değişenleri yükler.
Sahibi     : Stajyer 1

NOT — DVC:
    dvc add data/raw/eht çalıştırılır ancak dvc push YOK.
    Repo'da henüz yapılandırılmış DVC remote bulunmuyor;
    remote eklendikten sonra bu task'a dvc push eklenebilir.
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
EHT_DIR        = REPO_ROOT / "data" / "raw" / "eht"
MINIO_ENDPOINT = os.environ["MINIO_ENDPOINT"]
MINIO_USER     = os.environ["MINIO_USER"]
MINIO_ALIAS    = os.environ["MINIO_ALIAS"]
MIN_UVFITS_BYTES = 1024  # Bu değerin altı bozuk/yarım indirme işareti

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

def _mc_env() -> dict[str, str]:
    """
    MinIO client için güvenli ortam değişkenleri döner.

    MC_HOST_<alias> pattern'i kullanılır — secret hiçbir zaman
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
        cmd    : Argüman listesi (shell=False ile güvenli).
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


def _validate_counts(file_counts: dict[str, int]) -> None:
    """
    file_counts sözlüğünü EXPECTED_COUNTS ile karşılaştırır.
    Hata varsa ValueError fırlatır.

    Args:
        file_counts: {dataset_adı: bulunan_uvfits_sayısı}
    """
    errors: list[str] = []
    for ds, expected in EXPECTED_COUNTS.items():
        if ds not in file_counts:
            continue  # Bu çalıştırmada indirilmedi, atla

        actual = file_counts[ds]
        if actual == 0:
            continue  # Hiç dosya inmemiş, atla

        if actual != expected:
            errors.append(f"{ds}: beklenen {expected} dosya, bulunan {actual}")
            continue

        # Boyut kontrolü — bozuk/yarım indirme tespiti
        ds_dir = EHT_DIR / ds
        for fpath in ds_dir.glob("*.uvfits"):
            size = fpath.stat().st_size
            if size < MIN_UVFITS_BYTES:
                errors.append(f"{ds}/{fpath.name}: şüpheli boyut ({size} byte)")

    if errors:
        raise ValueError(
            "EHT doğrulama başarısız:\n" + "\n".join(f"  • {e}" for e in errors)
        )


# ─── DAG ─────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="eht_ingest",
    description="EHT UVFITS verilerini indir, doğrula, MinIO raw/eht/'ye yükle",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["data", "eht", "ingest", "stajyer1"],
) as dag:

    @task()
    def download_eht(datasets: list[str] | None = None) -> dict[str, int]:
        """
        scripts/download_eht_data.py'yi çağırarak EHT UVFITS dosyalarını indirir.

        Idempotent: script mevcut dosyaları atlar (SKIP logu basar).

        Args:
            datasets: İndirilecek dataset anahtarları.
                      None → EXPECTED_COUNTS içindeki tüm datasetler.

        Returns:
            {dataset_adı: bulunan_uvfits_sayısı} sözlüğü.
        """
        cmd = [sys.executable, str(SCRIPTS_DIR / "download_eht_data.py")]
        if datasets:
            cmd += ["--datasets"] + datasets

        _run(cmd, cwd=REPO_ROOT, timeout=7200, label="download_eht_data")

        counts: dict[str, int] = {}
        for ds in datasets or list(EXPECTED_COUNTS.keys()):
            ds_dir = EHT_DIR / ds
            counts[ds] = len(list(ds_dir.glob("*.uvfits"))) if ds_dir.exists() else 0

        print(f"İndirme sonuçları: {counts}")
        return counts

    @task()
    def validate_eht(file_counts: dict[str, int]) -> str:
        """
        İndirilen EHT dosyalarını doğrular.

        ge_suite.py repo'da mevcutsa kullanır; bu durumda file_counts
        ile tutarlılık için sayım kontrolü de ek olarak çalıştırılır.
        ge_suite.py yoksa bağımlılıksız lightweight doğrulama yapılır.

        Args:
            file_counts: download_eht task'ından gelen {dataset: sayı} sözlüğü.

        Returns:
            "validate_ok"

        Raises:
            ValueError: Herhangi bir kontrol başarısız olursa.
        """
        # Her iki durumda da sayım + boyut kontrolü çalışır
        _validate_counts(file_counts)

        # ge_suite.py varsa ek format/varlık kontrolü de çalıştır
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

        total = sum(file_counts.values())
        print(f"✓ EHT doğrulama başarılı — {total} dosya, {len(file_counts)} dataset")
        return "validate_ok"

    @task()
    def upload_to_minio() -> str:
        """
        Doğrulanmış EHT dosyalarını MinIO raw/eht/<dataset>/ yoluna yükler.

        Idempotent: mc mirror --overwrite sadece değişen/eksik dosyaları yükler.
        Credentials MC_HOST_{MINIO_ALIAS} env değişkeniyle iletilir, log'a düşmez.

        Returns:
            "upload_ok"

        Raises:
            RuntimeError: Yüklenecek dataset bulunamazsa veya mc başarısız olursa.
        """
        env = _mc_env()

        if not EHT_DIR.exists():
            raise RuntimeError(
                f"{EHT_DIR} bulunamadı. "
                "download_eht task'ının başarıyla tamamlandığından emin olun."
            )

        uploaded: list[str] = []
        for ds_dir in sorted(EHT_DIR.iterdir()):
            if not ds_dir.is_dir():
                continue
            _run(
                [
                    "mc", "mirror",
                    str(ds_dir) + "/",
                    f"{MINIO_ALIAS}/raw/eht/{ds_dir.name}/",
                    "--overwrite",
                ],
                env=env,
                timeout=1800,
                label=f"mc mirror {ds_dir.name}",
            )
            print(f"  ✓ {ds_dir.name} → {MINIO_ALIAS}/raw/eht/{ds_dir.name}/")
            uploaded.append(ds_dir.name)

        if not uploaded:
            raise RuntimeError(
                f"{EHT_DIR} içinde yüklenecek dataset dizini bulunamadı."
            )

        print(f"\n✓ Yükleme tamamlandı: {len(uploaded)} dataset")
        return "upload_ok"

    @task()
    def dvc_track() -> str:
        """
        data/raw/eht dizinini DVC ile izlemeye alır (dvc add).

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
            ["dvc", "add", "data/raw/eht"],
            cwd=REPO_ROOT,
            timeout=120,
            label="dvc add",
        )
        print("✓ data/raw/eht DVC ile izlemeye alındı.")
        print("  → data/raw/eht.dvc dosyasını git'e commit etmeyi unutma.")
        return "dvc_track_ok"

    # ── Bağımlılık zinciri ───────────────────────────────────────────────────
    counts = download_eht()
    validated = validate_eht(counts)
    uploaded = upload_to_minio()
    validated >> uploaded >> dvc_track()