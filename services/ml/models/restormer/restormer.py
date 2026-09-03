"""Restormer — Efficient Transformer for High-Resolution Image Restoration.

Restormer (Zamir et al., 2022), yüksek çözünürlüklü görüntü restorasyon
için tasarlanmış bir **transformer tabanlı** encoder-decoder modelidir.
Standart ViT'in aksine, attention'ı uzamsal boyut (H×W) yerine **kanal
boyutu (C)** üzerinden hesaplayarak O(N²) → O(C²) maliyet düşüşü sağlar.

## Mimari

Restormer, U-Net benzeri 4-level encoder-decoder yapısındadır:

```
Input (B, 1, H, W)
    │
    ├─── Shallow feature extraction (Conv 3×3) ──→ feat_0
    │
    ├─── Encoder Level 1 (dim=48)  ──→ feat_1 (downsample 2×)
    ├─── Encoder Level 2 (dim=96)  ──→ feat_2 (downsample 4×)
    ├─── Encoder Level 3 (dim=192) ──→ feat_3 (downsample 8×)
    ├─── Encoder Level 4 (dim=384) ──→ feat_4 (downsample 16×)
    │
    ├─── Decoder Level 3 (dim=192) ──→ feat_3' (upsample 8×)
    ├─── Decoder Level 2 (dim=96)  ──→ feat_2' (upsample 4×)
    ├─── Decoder Level 1 (dim=48)  ──→ feat_1' (upsample 2×)
    │
    ├─── Refinement (Conv 3×3)
    │
Output (B, 1, H, W)
```

Her seviyede:
- **MDTA** (Multi-Dconv Head Transposed Self-Attention): Uzun-mesafe
  bağımlılıkları yakalar.
- **GDFN** (Gated-Dconv Feed-forward Network): Yerel komşuluk + gating
  ile bilgi seçimi yapar.

Skip connection'lar **concat yerine element-wise addition** ile birleştirilir
(kanal sayısı korunur, hesaplama ucuzlar).

## Neden Restormer?

- **CNN'ler (U-Net, ESRGAN)**: Yerel bilgiyi iyi yakalar, ama uzun-mesafe
  bağımlılıkları zayıf.
- **ViT (standart transformer)**: Uzun-mesafe bağımlılıkları iyi yakalar,
  ama yüksek çözünürlükte O(N²) maliyet çok pahalı.
- **Restormer**: Her ikisinin avantajlarını birleştirir — uzun-mesafe
  bağımlılık + düşük hesaplama maliyeti.

Referans:
    Zamir, S. W., Arora, A., Khan, S., Hayat, M., Khan, F. S., Yang, M.-H.,
    Shao, L. (2022). "Restormer: Efficient Transformer for High-Resolution
    Image Restoration." CVPR.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from services.ml.models.restormer.gdfn import GDFN
from services.ml.models.restormer.mdta import MDTA


class TransformerBlock(nn.Module):
    """Bir transformer bloğu — MDTA (attention) + GDFN (FFN).

    Pre-norm mimari kullanır (LayerNorm önce uygulanır, residual sonra).
    Bu, derin ağlarda eğitim kararlılığını artırır.

    Args:
        dim: Feature kanal sayısı.
        num_heads: MDTA'daki head sayısı.
        expansion_factor: GDFN'deki genişletme oranı.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        expansion_factor: float = 2.66,
    ) -> None:
        super().__init__()

        # 1) Attention bloğu: LayerNorm → MDTA → residual
        self.norm1 = nn.GroupNorm(1, dim)  # GroupNorm(1) ≡ LayerNorm
        self.attn = MDTA(dim=dim, num_heads=num_heads)

        # 2) FFN bloğu: LayerNorm → GDFN → residual
        self.norm2 = nn.GroupNorm(1, dim)
        self.ffn = GDFN(dim=dim, expansion_factor=expansion_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — pre-norm transformer bloğu.

        Args:
            x: Giriş tensor, shape ``(B, dim, H, W)``.

        Returns:
            Transformer bloğu çıktısı, shape ``(B, dim, H, W)``.
        """
        # Attention bloğu (residual)
        x = x + self.attn(self.norm1(x))
        # FFN bloğu (residual)
        x = x + self.ffn(self.norm2(x))
        return x


class Downsample(nn.Module):
    """2× downsample — PixelUnshuffle ile kanal artırıp uzamsal boyutu yarıya indir.

    PixelUnshuffle, PixelShuffle'ın tersidir. (B, C, H, W) → (B, C*4, H/2, W/2)
    dönüşümü yapar. Bu, bilgi kaybı olmadan downsample yapmanın standart
    yoludur (strided conv'dan farklı olarak tüm pikseller korunur).

    Args:
        dim: Giriş kanal sayısı.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        # PixelUnshuffle kanal sayısını 4× yapar (2×2 bloktan 4 piksel)
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — 2× downsample.

        Args:
            x: Giriş tensor, shape ``(B, dim, H, W)``.

        Returns:
            Çıkış tensor, shape ``(B, dim//2, H/2, W/2)``.
        """
        return self.body(x)


class Upsample(nn.Module):
    """2× upsample — PixelShuffle ile uzamsal boyutu iki katına çıkar.

    PixelShuffle, (B, C*4, H, W) → (B, C, 2H, 2W) dönüşümü yapar.
    Sub-pixel convolution olarak da bilinir.

    Args:
        dim: Giriş kanal sayısı.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        # Conv ile kanal sayısını 4× yap, sonra PixelShuffle ile uzamsal büyüt
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — 2× upsample.

        Args:
            x: Giriş tensor, shape ``(B, dim, H, W)``.

        Returns:
            Çıkış tensor, shape ``(B, dim, 2H, 2W)``.
        """
        return self.body(x)


class Restormer(nn.Module):
    """Restormer — 4-level encoder-decoder transformer.

    Args:
        in_channels: Giriş kanal sayısı (default 1, grayscale).
        out_channels: Çıkış kanal sayısı (default 1, grayscale).
        dim: İlk seviye (level 1) kanal sayısı (default 48).
            Sonraki seviyeler 2× artar: 48, 96, 192, 384.
        num_blocks: Her seviyedeki transformer blok sayısı (default [4, 6, 6, 8]).
            Restormer makalesinde bu sıralama kullanılır — derin seviyelerde
            daha fazla blok (daha karmaşık özellikler için).
        num_heads: Her seviyedeki attention head sayısı (default [1, 2, 4, 8]).
            Kanal sayısıyla orantılı — daha fazla kanal = daha fazla head.
        expansion_factor: GDFN genişletme oranı (default 2.66).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        dim: int = 48,
        num_blocks: list[int] | None = None,
        num_heads: list[int] | None = None,
        expansion_factor: float = 2.66,
    ) -> None:
        super().__init__()

        if num_blocks is None:
            num_blocks = [4, 6, 6, 8]
        if num_heads is None:
            num_heads = [1, 2, 4, 8]

        if len(num_blocks) != 4 or len(num_heads) != 4:
            raise ValueError(
                f"num_blocks and num_heads must have length 4, "
                f"got {len(num_blocks)} and {len(num_heads)}"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.expansion_factor = expansion_factor

        # Kanal boyutları: [dim, dim*2, dim*4, dim*8]
        dims = [dim, dim * 2, dim * 4, dim * 8]

        # ─── Shallow feature extraction ───
        # İlk conv: in_channels → dim (3×3, padding=1)
        # Bu, ham piksel değerlerini feature uzayına taşır.
        self.shallow = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)

        # ─── Encoder ───
        # 4 seviye, her biri: [TransformerBlock × N] + Downsample
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downsamples = nn.ModuleList()
        for i in range(4):
            # Her seviyede N transformer bloğu
            blocks = nn.Sequential(
                *[
                    TransformerBlock(
                        dim=dims[i],
                        num_heads=num_heads[i],
                        expansion_factor=expansion_factor,
                    )
                    for _ in range(num_blocks[i])
                ]
            )
            self.encoder_blocks.append(blocks)
            # Son seviyede downsample yok (zaten en derin noktadayız)
            if i < 3:
                self.encoder_downsamples.append(Downsample(dims[i]))

        # ─── Bottleneck ───
        # En derin seviyede ekstra transformer blokları (Restormer makalesi)
        self.bottleneck = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dims[3],
                    num_heads=num_heads[3],
                    expansion_factor=expansion_factor,
                )
                for _ in range(num_blocks[3])
            ]
        )

        # ─── Decoder ───
        # 4 seviye, her biri: Upsample + [TransformerBlock × N]
        # Skip connection'lar encoder'dan gelen feature map'lerle toplanır.
        #
        # Sıralama: Bottleneck (dims[3]=384) → Seviye 3 (dims[2]=192) →
        #           Seviye 2 (dims[1]=96) → Seviye 1 (dims[0]=48) → Çıkış
        #
        # Her upsample bloğu, **kaynak seviyenin** kanal sayısını alır:
        # - decoder_upsamples[0]: dims[3]=384 → dims[2]=192 (bottleneck'ten seviye 3'e)
        # - decoder_upsamples[1]: dims[2]=192 → dims[1]=96  (seviye 3'ten 2'ye)
        # - decoder_upsamples[2]: dims[1]=96  → dims[0]=48  (seviye 2'den 1'e)
        self.decoder_upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for i in range(4):
            # İlk seviyede (i=0) upsample yok — bottleneck'ten direkt skip+block
            if i > 0:
                # Upsample, **kaynak seviye** (3-i+1 = 4-i) kanal sayısını kullanır
                # i=1 → kaynak seviye 3 (dims[3]=384)
                # i=2 → kaynak seviye 2 (dims[2]=192)
                # i=3 → kaynak seviye 1 (dims[1]=96)
                src_level = 4 - i
                self.decoder_upsamples.append(Upsample(dims[src_level]))
            # Decoder blokları — seviye 3'ten başla (en derin), 0'a doğru git
            level = 3 - i
            blocks = nn.Sequential(
                *[
                    TransformerBlock(
                        dim=dims[level],
                        num_heads=num_heads[level],
                        expansion_factor=expansion_factor,
                    )
                    for _ in range(num_blocks[level])
                ]
            )
            self.decoder_blocks.append(blocks)

        # ─── Refinement + Output ───
        # Son conv: dim → out_channels (3×3, padding=1)
        # Refinement, decoder çıktısını son kez işler.
        self.refinement = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — 4-level encoder-decoder.

        Args:
            x: Giriş tensor, shape ``(B, in_channels, H, W)``.

        Returns:
            Çıkış tensor, shape ``(B, out_channels, H, W)``.
        """
        # 1) Shallow feature extraction
        feat = self.shallow(x)

        # 2) Encoder — her seviyede feature map'i sakla (skip connection için)
        skips = []
        for i in range(4):
            feat = self.encoder_blocks[i](feat)
            skips.append(feat)
            if i < 3:
                feat = self.encoder_downsamples[i](feat)

        # 3) Bottleneck
        feat = self.bottleneck(feat)

        # 4) Decoder — skip connection'ları ters sırayla ekle
        for i in range(4):
            if i > 0:
                feat = self.decoder_upsamples[i - 1](feat)
            # Skip connection: encoder feature map'i ile topla (concat değil!)
            level = 3 - i
            feat = feat + skips[level]
            feat = self.decoder_blocks[i](feat)

        # 5) Refinement + Output
        feat = self.refinement(feat)
        out = self.output(feat)
        return out
