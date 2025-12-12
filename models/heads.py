# models/heads.py

import torch.nn as nn

class LinearHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.classifier = nn.Linear(in_dim, out_dim)
        
    def forward(self, features):
        return self.classifier(features)