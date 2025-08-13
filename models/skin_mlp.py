import torch.nn as nn # type: ignore

class Skin_Multi_Layer_Perceptron(nn.Module):
    def __init__(self, input_dim=12, hidden_dim1=32, hidden_dim2=16, output_dim=8):
        super(Skin_Multi_Layer_Perceptron, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim2, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.mlp(x)
