apiVersion: batch/v1
kind: Job
metadata:
  name: pix2pix-100ep-20260824
  namespace: deephorizon-ml
  labels:
    app: ml-training
    model: pix2pix
    dataset: training-512-v1
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 432000
  ttlSecondsAfterFinished: 604800
  template:
    metadata:
      labels:
        app: ml-training
        model: pix2pix
    spec:
      restartPolicy: Never
      automountServiceAccountToken: false
      containers:
        - name: trainer
          image: localhost:32000/deephorizon-training:2026-08-24
          imagePullPolicy: IfNotPresent
          command:
            - "python"
            - "-m"
            - "services.ml.training.gan_train"
          args:
            - "model=pix2pix"
            - "model.in_channels=1"
            - "model.out_channels=1"
            - "model.generator.dropout=0.5"
            - "model.generator.use_tanh=true"
            - "model.discriminator.in_channels=2"
            - "model.discriminator.base_channels=64"
            - "model.discriminator.max_channels=512"
            - "model.discriminator.n_layers=3"
            - "model.discriminator.use_sigmoid=true"
            - "training.epochs=100"
            - "training.batch_size=8"
            - "training.learning_rate=2.0e-4"
            - "training.scheduler.name=reduce_on_plateau"
            - "training.amp=true"
            - "training.amp_dtype=bfloat16"
            - "training.grad_accum_steps=1"
            - "training.early_stopping.enabled=true"
            - "training.early_stopping.min_epochs=40"
            - "training.early_stopping.patience=15"
            - "training.early_stopping.min_delta=1.0e-5"
            - "loss.name=combined"
            - "data.use_minio=true"
            - "data.bucket_name=datasets"
            - "data.minio_prefix=training-512/v1"
            - "data.augment=true"
            - "data.num_workers=4"
            - "data.pin_memory=true"
            - "device=cuda"
            - "paths.output_dir=/app/runs/pix2pix-100ep-20260824"
            - "hydra.run.dir=/app/runs/pix2pix-100ep-20260824/hydra"
            - "mlflow.experiment_name=pix2pix"
            - "mlflow.run_name=pix2pix-100ep-20260824"
          env:
            - name: MINIO_ENDPOINT
              value: http://minio.deephorizon-data.svc:9000
            - name: MINIO_ACCESS_KEY
              valueFrom:
                secretKeyRef:
                  name: minio-ml
                  key: access_key
            - name: MINIO_SECRET_KEY
              valueFrom:
                secretKeyRef:
                  name: minio-ml
                  key: secret_key
            - name: MLFLOW_TRACKING_URI
              value: http://mlflow.deephorizon-ml.svc:5000
            - name: PYTHONUNBUFFERED
              value: "1"
            - name: PYTHONPATH
              value: /app
          resources:
            requests:
              cpu: "8"
              memory: 32Gi
              nvidia.com/gpu: 1
            limits:
              cpu: "16"
              memory: 64Gi
              nvidia.com/gpu: 1
          volumeMounts:
            - name: outputs
              mountPath: /app/runs
            - name: dshm
              mountPath: /dev/shm
      volumes:
        - name: outputs
          persistentVolumeClaim:
            claimName: training-outputs-pvc
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 32Gi            kubectl get pod -n deephorizon-ml -w -l model=pix2pix"""Pix2Pix generator — U-Net wrapper for conditional image generation.

The generator is an encoder-decoder with skip connections (U-Net) that
maps a degraded input image to a clean output image. Dropout is applied
in the decoder (between upconv and concat) to introduce stochasticity,
following Isola et al. (2017).

Reference:
    Isola, P., Zhu, J.-Y., Zhou, T., Efros, A. A. (2017).
    "Image-to-Image Translation with Conditional Adversarial Networks."
    CVPR.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from services.ml.models.unet import UNet


class Pix2PixGenerator(nn.Module):
    """U-Net based generator for Pix2Pix.

    Wraps the deterministic ``UNet`` and adds optional dropout layers in
    the decoder path. The output is passed through ``Tanh`` to constrain
    pixel values to ``[-1, 1]`` — the standard Pix2Pix output range.

    Args:
        in_channels: Number of input channels (degraded image).
        out_channels: Number of output channels (generated image).
        dropout: Dropout probability in the decoder. ``0.0`` disables it.
        use_tanh: If ``True``, apply ``Tanh`` to the output. Pix2Pix
            training expects ``[-1, 1]``; inference pipelines that work
            in ``[0, 1]`` may set this to ``False``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        dropout: float = 0.5,
        use_tanh: bool = True,
    ) -> None:
        super().__init__()

        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.use_tanh = use_tanh

        # Core U-Net (encoder-decoder with skip connections)
        self.unet = UNet(in_channels=in_channels, out_channels=out_channels)

        # Dropout applied after each upconv (decoder path).
        # Pix2Pix paper applies dropout only in decoder; we mirror that.
        self._dropout_layers = nn.ModuleList()
        if dropout > 0.0:
            for _ in range(len(self.unet.ups)):  # decoder level sayısı kadar
                self._dropout_layers.append(nn.Dropout2d(p=dropout))

        # Output activation — Tanh constrains to [-1, 1].
        self._tanh = nn.Tanh() if use_tanh else nn.Identity()

    @staticmethod
    def _make_input_block(in_channels: int, out_channels: int) -> nn.Module:
        """Build a DoubleConv block with custom input channels."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through U-Net with decoder dropout.

        Args:
            x: Input tensor of shape ``(B, in_channels, H, W)``.

        Returns:
            Generated image of shape ``(B, out_channels, H, W)``,
            optionally passed through ``Tanh``.
        """
        # Encoder path — her seviyede skip connection sakla
        skips = []
        h = x
        for i, (encoder, pool) in enumerate(zip(self.unet.encoders, self.unet.pools)):
            h = encoder(h)
            skips.append(h)
            h = pool(h)

        # Bottleneck
        h = self.unet.bottleneck(h)

        # Decoder path with optional dropout
        for i, (up, decoder, skip) in enumerate(
            zip(self.unet.ups, self.unet.decoders, reversed(skips))
        ):
            h = up(h)
            if self._dropout_layers and i < len(self._dropout_layers):
                h = self._dropout_layers[i](h)
            h = torch.cat([h, skip], dim=1)
            h = decoder(h)

        h = self.unet.output(h)
        h = self._tanh(h)
        return h
