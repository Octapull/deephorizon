"""MinIO helpers used by remote training jobs.

The client is cached per process so every DataLoader worker owns its own boto3
client. Object listing uses an S3 paginator; a single ``list_objects_v2`` call
would silently cap the training set at 1,000 keys.
"""

from __future__ import annotations

import io
import os
from typing import Any

import boto3
import numpy as np
from dotenv import load_dotenv

load_dotenv()

_CLIENTS_BY_PID: dict[int, Any] = {}


def get_s3_client():
    endpoint = os.getenv("MINIO_ENDPOINT")
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    missing = [
        name
        for name, value in (
            ("MINIO_ENDPOINT", endpoint),
            ("MINIO_ACCESS_KEY", access_key),
            ("MINIO_SECRET_KEY", secret_key),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing MinIO environment variables: {', '.join(missing)}")

    pid = os.getpid()
    client = _CLIENTS_BY_PID.get(pid)
    if client is None:
        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        _CLIENTS_BY_PID[pid] = client
    return client


def list_files_in_minio(bucket_name: str, prefix_path: str) -> list[str]:
    client = get_s3_client()
    paginator = client.get_paginator("list_objects_v2")
    file_keys: list[str] = []

    try:
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix_path):
            file_keys.extend(
                obj["Key"]
                for obj in page.get("Contents", [])
                if obj["Key"].endswith(".npy")
            )
    except Exception as exc:
        raise RuntimeError(
            f"Could not list MinIO objects in {bucket_name}/{prefix_path}"
        ) from exc

    return sorted(file_keys)


def load_npy_from_minio(bucket_name: str, file_key: str) -> np.ndarray:
    client = get_s3_client()
    try:
        response = client.get_object(Bucket=bucket_name, Key=file_key)
        body = response["Body"]
        try:
            payload = body.read()
        finally:
            body.close()
        return np.load(io.BytesIO(payload), allow_pickle=False)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load MinIO object {bucket_name}/{file_key}"
        ) from exc
