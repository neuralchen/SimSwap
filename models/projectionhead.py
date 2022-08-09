import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self, proj_dim=256):
        super(ProjectionHead, self).__init__()

        self.proj = nn.Sequential(
            nn.Linear(proj_dim, proj_dim), 
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim), 
        )

    def forward(self, x):
        return self.proj(x)