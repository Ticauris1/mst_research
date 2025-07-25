import torch # type: ignore
import torch.nn as nn # type: ignore
from .film import FiLM

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7, use_film=False, film_in_dim=12):
        super().__init__()
        assert isinstance(channels, int) and channels > 0, f"❌ CBAM init error: channels={channels}"
        self.channels = channels
        self.use_film = use_film

        #print(f"⚙️ Initializing CBAM with channels={channels}, use_film={use_film}")

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
        B, C, H, W = x.shape
        #print(f"\n🚦 CBAM Forward Pass: input shape = {x.shape}")
        if C != self.channels:
            raise ValueError(f"❌ CBAM.forward(): input has {C} channels, expected {self.channels}")

        # Channel Attention
        ca = self.channel_attn(x)
        #print(f"✅ Channel attention output shape: {ca.shape}")
        ca = ca * x

        # Optional FiLM modulation
        if self.use_film and skin_vec is not None:
            #print(f"🎛️ Applying FiLM: skin_vec shape = {skin_vec.shape}")
            ca = self.film(ca, skin_vec)

        # Spatial Attention
        avg_out = torch.mean(ca, dim=1, keepdim=True)
        max_out, _ = torch.max(ca, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        #print(f"📐 Spatial attention input shape: {sa_input.shape}")
        sa = self.spatial_attn(sa_input)
        #print(f"✅ Spatial attention weights shape: {sa.shape}")

        return sa * ca