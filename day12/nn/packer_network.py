""" Neural network for packing presents with orientation masks """
import torch
from torch import nn


class MultiChannelPacker(nn.Module):
    """
    Super simple: Each orientation is a channel, network picks best
    """

    def __init__(self, grid_h, grid_w):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w

        # Simple CNN that processes grid + all orientations
        self.net = nn.Sequential(
            # Input channels: 1 (grid) + N (orientations)
            nn.Conv2d(1 + 8, 64, 3, padding=1),  # Assume max 8 orientations
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            # Output: placement scores + orientation scores
            nn.Conv2d(256, 128, 1),
            nn.ReLU(),
        )

        # Placement scores (where to put it)
        self.placement_out = nn.Conv2d(128, 1, 1)

        # Orientation scores (which orientation to use)
        self.orientation_out = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 8),
            nn.Softmax(dim=1)
        )

    def forward(self, grid, orientation_maps):
        """
        grid: (batch, 1, H, W)
        orientation_maps: (batch, N, H, W) - N pre-computed orientation maps
        """
        # Combine grid and all orientation maps
        x = torch.cat([grid, orientation_maps], dim=1)

        # Process
        features = self.net(x)

        # Get outputs
        placement_logits = self.placement_out(features)
        placement_scores = torch.sigmoid(placement_logits).squeeze(1)

        orientation_probs = self.orientation_out(features)

        return placement_scores, orientation_probs
