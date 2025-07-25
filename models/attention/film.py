import torch # type: ignore
import torch.nn as nn # type: ignore

class FiLM(nn.Module):
    def __init__(self, in_features, feature_map_channels):
        """
        Feature-wise Linear Modulation (FiLM) Layer.
        Args:
            in_features (int): Input size of skin_vec (e.g., 12)
            feature_map_channels (int): Number of channels in the input feature map
        """
        super().__init__()
        self.gamma_fc = nn.Linear(in_features, feature_map_channels)
        self.beta_fc = nn.Linear(in_features, feature_map_channels)

    def forward(self, x, cond):
        """
        Args:
            x: Feature map tensor of shape [B, C, H, W]
            cond: Conditioning vector (e.g., skin vector) of shape [B, in_features]
        Returns:
            FiLM-modulated feature map of shape [B, C, H, W]
        """
        gamma = self.gamma_fc(cond).unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        beta = self.beta_fc(cond).unsqueeze(2).unsqueeze(3)    # [B, C, 1, 1]
        return gamma * x + beta