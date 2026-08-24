import torch
import torch.nn as nn


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: list[int] | None = None,
    ):
        """U-Net encoder-decoder with skip connections.

        Args:
            in_channels: Number of input channels (default: 1 for grayscale).
            out_channels: Number of output channels (default: 1).
            features: Channel widths at each encoder level. Length determines
                depth. Default: [64, 128, 256, 512] (4-level U-Net).
        """
        super().__init__()

        if features is None:
            features = [64, 128, 256, 512]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_channels = in_channels
        for feat in features:
            self.encoders.append(DoubleConv(prev_channels, feat))
            self.pools.append(nn.MaxPool2d(2))
            prev_channels = feat

        # Bottleneck (2x son feature)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        reversed_features = list(reversed(features))
        for i in range(len(reversed_features)):
            in_feat = reversed_features[i] * 2  # bottleneck veya onceki decoder
            out_feat = reversed_features[i]
            self.ups.append(
                nn.ConvTranspose2d(in_feat, out_feat, kernel_size=2, stride=2)
            )
            # Skip connection: concat(out_feat, encoder[i]) → DoubleConv
            self.decoders.append(DoubleConv(out_feat * 2, out_feat))

        # Output projection
        self.output = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # Backward-compat aliases (Pix2Pix/ESRGAN generator'leri bunlara erişiyor)
        # enc1, enc2, ... → encoder blokları
        # pool1, pool2, ... → pooling katmanları
        # up1, up2, ... → upsampling katmanları
        # dec1, dec2, ... → decoder blokları
        for i, (enc, pool, up, dec) in enumerate(
            zip(self.encoders, self.pools, reversed(self.ups), reversed(self.decoders))
        ):
            setattr(self, f"enc{i + 1}", enc)
            setattr(self, f"pool{i + 1}", pool)
        for i, (up, dec) in enumerate(zip(reversed(self.ups), reversed(self.decoders))):
            setattr(self, f"up{i + 1}", up)
            setattr(self, f"dec{i + 1}", dec)

    def forward(self, x):
        # Encoder: her seviyede feature map'i sakla (skip connection için)
        skips = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder: skip connection'ları ters sırayla kullan
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        # Output projection
        x = self.output(x)
        return x
