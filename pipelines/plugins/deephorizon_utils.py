"""
DeepHorizon — Airflow Ortak Yardımcılar
=========================================
Tüm DAG'larda tekrar eden _run() / _mc_env() / _dvc_env() buraya taşındı.
Airflow plugins/ dizinindeki her modül DAG'lardan doğrudan import edilebilir.

Kullanım (DAG dosyasında):
    from deephorizon_utils import run, mc_env, dvc_env
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


def run(
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


def mc_env(alias: str | None = None, user: str | None = None) -> dict[str, str]:
    """
    MinIO client (mc) için güvenli ortam değişkenleri döner.

    MC_HOST_{alias} pattern'i kullanılır — secret hiçbir zaman subprocess
    argümanında (ps çıktısı / Airflow logları) görünmez.

    Secret arama sırası:
      1. MINIO_SECRET ortam değişkeni
      2. Airflow Variable 'minio_data_pipeline_secret'

    Args:
        alias: MinIO alias adı. None → MINIO_ALIAS env değişkeni.
        user : MinIO kullanıcı adı. None → MINIO_USER env değişkeni.
    """
    _alias    = alias or os.environ["MINIO_ALIAS"]
    _user     = user  or os.environ["MINIO_USER"]
    _endpoint = os.environ["MINIO_ENDPOINT"]

    secret = os.environ.get("MINIO_SECRET")
    if not secret:
        try:
            from airflow.models import Variable  # type: ignore[import]
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
    env[f"MC_HOST_{_alias}"] = (
        f"http://{_user}:{secret}"
        f"@{_endpoint.removeprefix('http://')}"
    )
    return env


def dvc_env() -> dict[str, str]:
    """
    dvc push için güvenli ortam değişkenleri döner.

    DVC S3 backend'i (boto3) AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY'e bakar.
    'dvc' kullanıcısının parola Airflow'da Variable olarak tutulur.

    Secret arama sırası:
      1. DVC_SECRET ortam değişkeni
      2. Airflow Variable 'dvc_secret'

    .dvc/config içindeki remote URL ve endpointurl'ü okur;
    parola hiçbir zaman config'e yazılmaz.
    """
    secret = os.environ.get("DVC_SECRET")
    if not secret:
        try:
            from airflow.models import Variable  # type: ignore[import]
            secret = Variable.get("dvc_secret", default_var=None)
        except Exception:
            pass
    if not secret:
        raise RuntimeError(
            "DVC secret bulunamadı. "
            "DVC_SECRET ortam değişkeni veya "
            "'dvc_secret' Airflow Variable'ı tanımlı değil."
        )

    env = os.environ.copy()
    env["AWS_ACCESS_KEY_ID"]     = "dvc"
    env["AWS_SECRET_ACCESS_KEY"] = secret
    return env
