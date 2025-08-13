import torch.nn as nn  # type: ignore

class TwoLayerClassifierHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=7):  # or 4 for FairFace4
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.head(x)