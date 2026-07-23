from services.ml.minio_loader import list_files_in_minio, load_npy_from_minio
import numpy as np

BUCKET = "datasets"
PREFIXES = ["training-512/v1/clean/", "training-512/v1/degraded/"]

for prefix in PREFIXES:
    print(f"\n=== {BUCKET}/{prefix} ===")
    files = list_files_in_minio(BUCKET, prefix)
    print(f"Toplam dosya: {len(files)}")
    for i, key in enumerate(files[:3]):
        data = load_npy_from_minio(BUCKET, key)
        if data is None:
            print(f"  {key}  -> YUKLENEMEDI")
            continue
        mn = float(np.min(data))
        mx = float(np.max(data))
        ok = (mn >= 0.0) and (mx <= 1.0)
        status = "OK 0-1" if ok else "HATA"
        print(f"  {key}  shape={data.shape}  min={mn:.6f}  max={mx:.6f}  -> {status}")
