"""Multi-Dconv Head Transposed Self-Attention (MDTA) — Restormer'ın kalbi.

Restormer (Zamir et al., 2022), standart ViT'in uzamsal self-attention'ını
**kanal bazlı (channel-wise)** self-attention ile değiştirir. Bu, hesaplama
maliyetini O(N²)'den O(C²)'ye düşürür (N=H×W token, C=kanal sayısı).

MDTA'nın iki temel bileşeni vardır:

1. **Depth-wise convolution (Dconv)**: Q, K, V üretilmeden önce her kanala
   ayrı bir 3×3 dwconv uygulanır. Bu, **yerel bağlam** (local context) yakalar
   ve saf attention'ın "tüm konumlar eşit ağırlıkta" sorununu giderir.

2. **Transposed self-attention**: Feature map (B, C, H, W) → (B, H×W, C)
   reshape edilir. Attention **C token üzerinden** hesaplanır, yani her
   uzamsal konum için tüm kanallar arasındaki ilişki öğrenilir.

Matematiksel olarak:

    Q = Dconv_q(x),  K = Dconv_k(x),  V = Dconv_v(x)
    Q, K, V ∈ R^{B × C × H × W}
    Q', K', V' = reshape to (B × H × W, C, 1)  # her konum bir C-vektörü
    Attention = softmax(Q'ᵀ K' / √d) V'
    out = reshape back to (B, C, H, W)

Bu sayede:
- Uzamsal boyut (H×W) büyük olsa bile attention maliyeti sabit kalır
- Her piksel konumu için **kanal-arası korelasyon** öğrenilir
- Dconv sayesinde **yerel komşuluk** bilgisi korunur

Referans:
    Zamir, S. W., Arora, A., Khan, S., Hayat, M., Khan, F. S., Yang, M.-H.,
    Shao, L. (2022). "Restormer: Efficient Transformer for High-Resolution
    Image Restoration." CVPR.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DconvProjection(nn.Module):
    """Q, K, V için depth-wise conv + 1×1 conv projeksiyonu.

    Standart attention'da Q/K/V sadece 1×1 conv ile üretilir. MDTA'da ise
    önce bir **depth-wise 3×3 conv** uygulanır — bu yerel komşuluk bilgisini
    yakalar, sonra 1×1 conv ile kanal sayısı ayarlanır.

    Args:
        dim: Giriş/çıkış kanal sayısı.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        # Depth-wise conv: her kanal için ayrı 3×3 filtre (groups=dim)
        # Bu, standart conv'dan C kat daha ucuz ve yerel bilgiyi yakalar.
        self.dconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        # 1×1 conv: kanal sayısını ayarla (bias=True → öğrenilebilir offset)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — dwconv + 1×1 proj.

        Args:
            x: Giriş tensor, shape ``(B, dim, H, W)``.

        Returns:
            Projeksiyon çıktısı, shape ``(B, dim, H, W)``.
        """
        x = self.dconv(x)
        x = self.proj(x)
        return x


class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Self-Attention — Restormer'ın attention bloğu.

    Standart multi-head self-attention'dan iki temel farkı vardır:

    1. **Dconv ön-projeksiyon**: Q, K, V üretilmeden önce dwconv uygulanır
       (yerel bağlam yakalanır).
    2. **Transposed attention**: Attention uzamsal boyut (H×W) yerine
       **kanal boyutu (C)** üzerinden hesaplanır. Maliyet O(C²·HW) olur,
       standart attention'ın O((HW)²·C) maliyetinden çok daha düşüktür.

    Args:
        dim: Giriş/çıkış kanal sayısı.
        num_heads: Attention head sayısı. dim, num_heads'e bölünebilir olmalı.
        bias: Q/K/V projeksiyonlarında bias kullanılsın mı.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )

        self.dim = dim
        self.num_heads = num_heads
        # Her head'in kanal boyutu — toplam dim = num_heads × head_dim
        self.head_dim = dim // num_heads
        # Ölçekleme faktörü — softmax öncesi QKᵀ büyüklüğünü kontrol eder
        self.scale = self.head_dim**-0.5

        # Q, K, V projeksiyonları — her biri dwconv + 1×1 conv
        self.q_proj = _DconvProjection(dim)
        self.k_proj = _DconvProjection(dim)
        self.v_proj = _DconvProjection(dim)

        # Çıkış projeksiyonu — 1×1 conv + dwconv (simetri için)
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — transposed self-attention.

        Args:
            x: Giriş tensor, shape ``(B, dim, H, W)``.

        Returns:
            Attention çıktısı, shape ``(B, dim, H, W)``.
        """
        B, C, H, W = x.shape

        # 1) Q, K, V üret — her biri (B, dim, H, W)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2) Multi-head reshape: (B, dim, H, W) → (B, num_heads, head_dim, H*W)
        #    Standart attention'dan farklı olarak head_dim kanal boyutu,
        #    H*W ise "token sayısı" gibi ele alınır.
        q = q.reshape(B, self.num_heads, self.head_dim, H * W)
        k = k.reshape(B, self.num_heads, self.head_dim, H * W)
        v = v.reshape(B, self.num_heads, self.head_dim, H * W)

        # 3) Transposed attention: (B, num_heads, head_dim, H*W) →
        #    (B, num_heads, H*W, head_dim) — kanal boyutunu "token" yapar
        q = q.permute(0, 1, 3, 2)  # (B, heads, HW, head_dim)
        k = k.permute(0, 1, 3, 2)  # (B, heads, HW, head_dim)
        v = v.permute(0, 1, 3, 2)  # (B, heads, HW, head_dim)

        # 4) Attention skoru: QKᵀ → (B, heads, HW, HW)
        #    Maliyet: HW × HW × head_dim — ama head_dim küçük (dim/num_heads)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # 5) Attention × V → (B, heads, HW, head_dim)
        out = attn @ v

        # 6) Geri reshape: (B, heads, HW, head_dim) → (B, dim, H, W)
        out = out.permute(0, 1, 3, 2)  # (B, heads, head_dim, HW)
        out = out.reshape(B, C, H, W)

        # 7) Çıkış projeksiyonu
        out = self.out_proj(out)
        return out
