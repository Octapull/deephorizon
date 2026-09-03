# Egitim imaji — services/ml/training/*.py'yi GPU'lu bir Kubernetes Job icinde calistirir.
#
# NEDEN AYRI IMAJ: egitim kodu veriyi MinIO'dan cekiyor (boto3), Hydra ile
# konfigure ediliyor ve her kosuyu MLflow'a logluyor. Bunlarin hicbiri stok
# pytorch imajinda yok.
#
# BUILD (sunucuda, repo kokunden):
#   docker build -f infra/docker/training.Dockerfile -t localhost:32000/deephorizon-training:<etiket> .
#   docker push localhost:32000/deephorizon-training:<etiket>
#
# Etiket olarak `latest` KULLANMA — Kubernetes ayni etiketi yeniden cekmez,
# guncelleme sessizce uygulanmaz. Tarih ver: 2026-08-07 gibi.
#
# Kullanim rehberi: docs/runbooks/ML-TRAINING.md

# Torch + CUDA + cuDNN hazir gelir. Torch'u pip ile kurmak GB'larca indirme ve
# CUDA surum eslesmesi riski demek. L40S (Ada Lovelace, sm_89) cu124 ile desteklenir.
#
# Bu imajda Python 3.11 var, repo koku 3.13 istiyor (pyproject.toml). Fark
# bilincli: 3.13 alt siniri ehtim icin konuldu (veri squad'i) ve egitim kodu
# ehtim kullanmiyor. Egitim tarafinda 3.13'e ihtiyac duyan bir sey cikarsa
# nvidia/cuda taban imajina gecilir ve torch cu124 wheel'i elle kurulur.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# --- Python bagimliliklari ---
# torch/torchvision zaten imajda ve requirements'taki alt sinirlari karsiliyor;
# pip onlari yeniden indirmez. Geri kalan (mlflow, hydra, boto3, metrikler) gelir.
COPY requirements/ /app/requirements/
RUN pip install --no-cache-dir -r requirements/ml.txt \
 && rm -rf /app/requirements

# --- ONNX export bagimliliklari (Faz 5 Görev 41) ---
# Restormer ve diger modellerin ONNX export'u icin gerekli.
# ml.txt'ye eklemek yerine burada kuruyoruz — boylece ml.txt sadece egitim
# bagimliliklarini icerir, export ayri bir concern.
RUN pip install --no-cache-dir onnx>=1.15.0 onnxruntime>=1.17.0

# --- Dogrulama: eksik bir sey varsa build burada dussun, egitimin 3. saatinde degil ---
RUN python -c "\
import torch, torchvision, torchmetrics, mlflow, hydra, omegaconf, boto3, lpips, skimage, cv2, numpy; \
import onnx, onnxruntime; \
print('torch', torch.__version__, '| cuda', torch.version.cuda, '| numpy', numpy.__version__); \
print('onnx', onnx.__version__, '| onnxruntime', onnxruntime.__version__); \
assert torch.version.cuda, 'CUDA destegi olmayan torch kuruldu'"

# --- Uygulama kodu ---
# Yalnizca services/ml kopyalanir; services/api (Go) ve services/frontend (Node)
# egitimle ilgisiz ve imaji buyutur. Build context filtresi:
# infra/docker/training.Dockerfile.dockerignore
COPY services/ml/ /app/services/ml/

# --- Calisma kullanicisi ---
# GPU erisimi root gerektirmez. Hydra cikti klasorunu (outputs/<tarih>) calisma
# dizinine yazar; bu yuzden WORKDIR yazilabilir ve trainer'a ait olmali.
RUN useradd --create-home trainer \
 && install -d -o trainer -g trainer /workspace
USER trainer
WORKDIR /workspace

# ENTRYPOINT — CMD DEGIL. Kubernetes'te manifest'e yazilan `args` CMD'nin
# YERINE gecer, ENTRYPOINT'in ise SONUNA eklenir.
#
# Bu imaj birden fazla egitim scripti destekler:
#   - services.ml.training.train          (U-Net baseline)
#   - services.ml.training.train_esrgan   (ESRGAN)
#   - services.ml.training.train_restormer (Restormer — Faz 5)
#
# Hangi script'in calisacagi TRAINING_SCRIPT env variable ile secilir.
# Kubernetes manifest'inde:
#   env:
#     - name: TRAINING_SCRIPT
#       value: "services.ml.training.train_restormer"
#
# Imajda kabuk acmak gerekirse ENTRYPOINT'i atla:
#   docker run --rm --entrypoint bash <imaj>
#   kubectl run dbg --image=<imaj> --command -- bash
ENTRYPOINT ["sh", "-c", "python -m ${TRAINING_SCRIPT:-services.ml.training.train}"]
