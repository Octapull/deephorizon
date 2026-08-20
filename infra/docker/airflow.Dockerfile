# Airflow worker imaji — DAG task'larinin ihtiyac duydugu araclar.
#
# Stok apache/airflow imajinda ehtim/mc/dvc/great-expectations yok; DAG'lar
# gorunur ama task'lar "command not found" ile duser. Bu imaj o boslugu doldurur.
#
# BUILD (sunucuda, repo kokunden):
#   docker build -f infra/docker/airflow.Dockerfile -t localhost:32000/deephorizon-airflow:<etiket> .
#   docker push localhost:32000/deephorizon-airflow:<etiket>
#
# Sonra infra/k8s/airflow/values.yaml -> images.airflow.repository/tag guncellenir
# ve YENIDEN RENDER alinir (values.yaml basindaki komut).
#
# Etiket olarak `latest` KULLANMA — Kubernetes ayni etiketi yeniden cekmez,
# guncelleme sessizce uygulanmaz. Tarih/surum ver: 2026-07-20 gibi.

FROM apache/airflow:3.2.0

# --- Sistem araclari (root) ---
USER root

# mc = MinIO client. DAG'lar `mc mirror` ile MinIO'ya yukluyor.
RUN curl -fsSL https://dl.min.io/client/mc/release/linux-amd64/mc \
      -o /usr/local/bin/mc \
 && chmod +x /usr/local/bin/mc \
 && mc --version

# --- Python bagimliliklari (airflow kullanicisi) ---
# Airflow imajinda pip her zaman airflow kullanicisiyla calistirilir; root ile
# kurulum imajin izin yapisini bozar.
USER airflow

# --chown sart: COPY varsayilan olarak root sahipliginde kopyalar, airflow
# kullanicisi sonrasinda silemez ("Permission denied").
COPY --chown=airflow:0 requirements/ /tmp/requirements/

# Constraint dosyasi Airflow'un KENDI bagimliliklarini sabit tutar; ehtim gibi
# listede olmayan paketler serbest cozulur. Onsuz pip, Airflow'un pinlerini
# ezip calisan kurulumu bozabilir.
RUN pip install --no-cache-dir -r /tmp/requirements/data.txt \
      --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.2.0/constraints-3.13.txt" \
 && rm -rf /tmp/requirements

# --- Dogrulama: eksik bir sey varsa build burada dussun, calisma aninda degil ---
RUN python -c "import ehtim, astropy, numpy, scipy, skimage, cv2, h5py; print('python paketleri OK')" \
 && dvc --version \
 && mc --version \
 && python -c "import great_expectations; print('great-expectations OK')"
