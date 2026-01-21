""" Policy implementation for use with my present packing environment. """
from dataclasses import dataclass
from tensordict import TensorDict
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
    present_idx_head: nn.Sequential
    x_head: nn.Sequential
    y_head: nn.Sequential
    rot_head: nn.Sequential
    flip_head: nn.Sequential


class PresentPlacementPolicy(nn.Module):
    """ Policy nn for PresentEnv with spatial awareness """

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        # Process grid
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128)
        )

        # Process present
        self.present_encoder = nn.Sequential(
            nn.Linear(6 * 3 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Process present_count
        self.present_count_encoder = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        combined_features = 128 + 128 + 32

        present_idx_head = nn.Sequential(
            nn.Linear(combined_features, 1),
            nn.Softmax(-1)
        )
        x_head = nn.Sequential(
            nn.Linear(combined_features, 1),
            nn.Sigmoid()
        )
        y_head = nn.Sequential(
            nn.Linear(combined_features, 1),
            nn.Sigmoid()
        )
        rot_head = nn.Sequential(
            nn.Linear(combined_features, 1),
            nn.Softmax(-1)
        )
        flip_head = nn.Sequential(
            nn.Linear(combined_features, 2),
            nn.Sigmoid()
        )

        self.heads = Heads(present_idx_head, x_head,
                           y_head, rot_head, flip_head)

    def update_grid_size(self, grid_size):
        """ Updates grid encoder for new grid size """
        self.grid_encoder = nn.Sequential(
            nn.Linear(grid_size[0]*grid_size[1], 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

    def forward(self, tensordict):
        """ Forward function for running of nn """
        grid = tensordict.get("grid")
        presents = tensordict.get("presents")
        present_count = tensordict.get("present_count")

        # get grid dimensions for x y calculations
        h, w = grid.shape

        # add batch dimensions
        grid = grid.unsqueeze(0).unsqueeze(0)
        presents = presents.unsqueeze(0)
        present_count = present_count.unsqueeze(0)

        # get grid features first
        grid_features = self.grid_encoder(grid)

        # get features from
        present_features = self.present_encoder(
            presents.flatten(start_dim=1))

        # get present_count features
        present_count_features = self.present_count_encoder(
            present_count)

        all_features = torch.cat([
            grid_features,
            present_features,
            present_count_features
        ], dim=1)

        with torch.no_grad():
            present_idx_probs = self.heads.present_idx_head(all_features)
            x_norm = self.heads.x_head(all_features)
            y_norm = self.heads.y_head(all_features)
            rot_probs = self.heads.rot_head(all_features)
            flip_probs = self.heads.flip_head(all_features)

            # convert probabilities to predicted values
            present_idx = torch.multinomial(
                present_idx_probs, 1).to(torch.uint8)
            rot = torch.multinomial(rot_probs, 1).to(torch.uint8)

            max_x = w - 3
            max_y = h - 3

            # coordinates are continuous: scale norm values and round
            x = (x_norm * max_x).round().to(torch.int64)
            y = (y_norm * max_y).round().to(torch.int64)

            # flip is binary so just has 1 threshold
            flip = (flip_probs > 0.5).to(torch.uint8)

        return TensorDict({
            "present_idx": present_idx,
            "x": x,
            "y": y,
            "rot": rot,
            "flip": flip
        }, batch_size=present_idx.shape[0])
