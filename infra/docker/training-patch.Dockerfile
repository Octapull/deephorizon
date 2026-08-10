FROM registry.container-registry.svc.cluster.local:5000/deephorizon-training:2026-08-07b

WORKDIR /app
COPY services/ml /app/services/ml

ENTRYPOINT ["python", "-m", "services.ml.training.train"]
