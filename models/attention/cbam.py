import torch # type: ignore
import torch.nn as nn # type: ignore
from .film import FiLM

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7, use_film=False, film_in_dim=12):
        super().__init__()
        assert isinstance(channels, int) and channels > 0, \
            f"❌ CBAM init error: channels must be int>0, got {channels}"
        self.channels = channels
        self.use_film = use_film

        #print(f"⚙️ [CBAM Init] channels={channels}, reduction={reduction}, "
        #      f"kernel_size={kernel_size}, use_film={use_film}")

        # Channel Attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Spatial Attention
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.Sigmoid()
        )

        # Optional FiLM
        if self.use_film:
            self.film = FiLM(film_in_dim, channels)

    def forward(self, x, skin_vec=None):
        # === Debug: Check shapes early ===
        #print(f"\n🚦 [CBAM Forward] Input shape: {x.shape}")
        assert x.dim() == 4, f"❌ Expected 4D input [B,C,H,W], got {x.shape}"
        B, C, H, W = x.shape
        assert C == self.channels, f"❌ Channel mismatch: got {C}, expected {self.channels}"

        if self.use_film:
            assert skin_vec is not None, "❌ use_film=True but skin_vec is None"
            assert skin_vec.shape[0] == B, \
                f"❌ Batch mismatch: skin_vec batch {skin_vec.shape[0]} vs input {B}"
            #print(f"🎛️ [CBAM] skin_vec shape: {skin_vec.shape}")

        # === Channel Attention ===
        ca = self.channel_attn(x)  # [B,C,1,1]
        #print(f"✅ [CBAM] Channel attention weights shape: {ca.shape}")
        ca = ca * x

        # === Optional FiLM ===
        if self.use_film:
            #print("🎚️ [CBAM] Applying FiLM modulation...")
            ca = self.film(ca, skin_vec)

        # === Spatial Attention ===
        avg_out = torch.mean(ca, dim=1, keepdim=True)  # [B,1,H,W]
        max_out, _ = torch.max(ca, dim=1, keepdim=True)  # [B,1,H,W]
        sa_input = torch.cat([avg_out, max_out], dim=1)  # [B,2,H,W]
        #print(f"📐 [CBAM] Spatial attention input shape: {sa_input.shape}")
        sa = self.spatial_attn(sa_input)  # [B,1,H,W]
        #print(f"✅ [CBAM] Spatial attention weights shape: {sa.shape}")

        out = sa * ca
        #print(f"🏁 [CBAM] Output shape: {out.shape}")
        return out
