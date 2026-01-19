""" Policy implementation for use with my present packing environment. """
from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class FeatureExtractor:
    """ Holds all feature extractors """
    grid_encoder: nn.Sequential
    present_encoder: nn.Sequential
    present_count_encoder: nn.Sequential


@dataclass
class Heads:
    """ Holds all model heads """
    present_idx_head: nn.Linear
    x_head: nn.Linear
    y_head: nn.Linear
    rot_head: nn.Linear
    flip_head: nn.Linear


class PresentPlacementPolicy(nn.Module):
    """ Policy nn for PresentEnv with spatial awareness """

    def __init__(self, grid_size):
        """
        Args:
            grid_size: (height, width) of the grid
        """
        super().__init__()

        self.flatten = nn.Flatten()

        # Process grid
        grid_encoder = nn.Sequential(
            nn.Linear(grid_size[0]*grid_size[1]*6, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Process present
        present_encoder = nn.Sequential(
            nn.Linear(3*3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Process present_count
        present_count_encoder = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.extractors = FeatureExtractor(
            grid_encoder, present_encoder, present_count_encoder)

        combined_features = 128 + 128 + 32

        present_idx_head = nn.Linear(combined_features, 1)
        x_head = nn.Linear(combined_features, 1)
        y_head = nn.Linear(combined_features, 1)
        rot_head = nn.Linear(combined_features, 1)
        flip_head = nn.Linear(combined_features, 2)

        self.heads = Heads(present_idx_head, x_head,
                           y_head, rot_head, flip_head)

    def forward(self, grid, presents, present_count):
        """ Forward function for running of nn """
        grid_features = self.extractors.grid_encoder(self.flatten(grid))

        present_features = self.extractors.present_encoder(
            self.flatten(presents))

        present_count_features = self.extractors.present_encoder(
            self.flatten(present_count))

        all_features = torch.cat(
            [grid_features, present_features, present_count_features])

        present_idx = self.heads.present_idx_head(all_features)
        x = self.heads.x_head(all_features)
        y = self.heads.y_head(all_features)
        rot = self.heads.rot_head(all_features)
        flip = self.heads.flip_head(all_features)

        return present_idx, x, y, rot, flip
