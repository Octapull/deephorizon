"""Gated-Dconv Feed-forward Network (GDFN) — Restormer'ın FFN bloğu.

Standart transformer FFN'i iki lineer katmandan oluşur:
    FFN(x) = Linear₂(GELU(Linear₁(x)))

Restormer'da bu yapı **gating mekanizması** ile zenginleştirilir:

    GDFN(x) = Linear₂(φ(Gate(x)) ⊙ Linear₁(x))

Burada:
- **Linear₁**: Genişletme (expansion) — kanal sayısını dim → dim*expansion_factor
- **Gate**: Aynı genişletilmiş boyutta, dwconv'lu paralel yol
- **φ**: GELU aktivasyonu (non-linearity)
- **⊙**: Element-wise çarpım (gating)
- **Linear₂**: Daraltma — dim*expansion_factor → dim

Gating'in rolü: Ağ, **hangi bilginin bir sonraki katmana aktarılacağını**
öğrenir. İki paralel yolun çarpımı, faydalı özelliklerin güçlendirilmesini
ve gereksiz bilginin bastırılmasını sağlar (denoising, derestoration gibi
görevlerde kritik).

**Neden dwconv?**
Gate yolundaki depth-wise conv, **yerel komşuluk** bilgisini yakalar.
Saf gating (dwconv olmadan) sadece kanal-arası karışım yapar; dwconv
eklemek uzamsal bilgiyi de gating'e dahil eder.

Referans:
    Zamir, S. W., et al. (2022). "Restormer: Efficient Transformer for
    High-Resolution Image Restoration." CVPR.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GDFN(nn.Module):
    """Gated-Dconv Feed-forward Network — Restormer'ın FFN bloğu.

    İki paralel yolun element-wise çarpımı ile gating yapar:
    - **Değer yolu** (value path): Linear → dwconv → activation
    - **Gate yolu** (gate path): Linear → dwconv (activation yok)

    Çıkış: Daraltma (Linear) ile orijinal kanal sayısına dönüş.

    Args:
        dim: Giriş/çıkış kanal sayısı.
        expansion_factor: Genişletme oranı (default 2.66, Restormer makalesi).
            Restormer makalesinde 2.66 kullanılır — bu, GLU (Gated Linear Unit)
            varyantlarında yaygın bir oran.
    """

    def __init__(
        self,
        dim: int,
        expansion_factor: float = 2.66,
    ) -> None:
        super().__init__()

        if expansion_factor <= 0:
            raise ValueError(f"expansion_factor must be > 0, got {expansion_factor}")

        self.dim = dim
        self.expansion_factor = expansion_factor

        # Genişletilmiş kanal sayısı — Restormer makalesinde 2.66× kullanılır
        # Bu oran, gating için iki paralel yola yetecek kadar alan bırakır.
        hidden_dim = int(dim * expansion_factor)

        # Değer yolu: genişletme + dwconv + GELU
        # 1×1 conv ile kanal sayısını dim → hidden_dim'e çıkar
        self.value_proj = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        # Depth-wise conv: yerel komşuluk bilgisi (groups=hidden_dim → her kanal ayrı)
        self.value_dconv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim
        )
        # GELU aktivasyonu — standart transformer FFN'deki ReLU yerine
        self.value_act = nn.GELU()

        # Gate yolu: genişletme + dwconv (aktivasyon yok)
        # Gate, değer yolunun çıktısıyla çarpılacak — aktivasyon burada
        # uygulanmaz çünkü gating zaten bir tür "yumuşak seçim" yapar.
        self.gate_proj = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.gate_dconv = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim
        )

        # Daraltma: hidden_dim → dim (1×1 conv)
        self.out_proj = nn.Conv2d(hidden_dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — gated feed-forward.

        Args:
            x: Giriş tensor, shape ``(B, dim, H, W)``.

        Returns:
            FFN çıktısı, shape ``(B, dim, H, W)``.
        """
        # 1) Değer yolu: x → Linear → dwconv → GELU
        value = self.value_proj(x)
        value = self.value_dconv(value)
        value = self.value_act(value)

        # 2) Gate yolu: x → Linear → dwconv (aktivasyon yok)
        gate = self.gate_proj(x)
        gate = self.gate_dconv(gate)

        # 3) Gating: element-wise çarpım
        #    Bu, ağın "hangi kanalların geçeceğine" karar vermesini sağlar.
        hidden = value * gate

        # 4) Daraltma: hidden_dim → dim
        out = self.out_proj(hidden)
        return out
