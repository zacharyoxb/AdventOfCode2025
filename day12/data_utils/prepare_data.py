""" Gets orientations of shape """
import numpy as np
import torch


def get_unique_orientations(shape_np):
    """ Get all unique rotations/flips of a shape """
    unique = []
    seen = set()

    for rotation in [0, 1, 2, 3]:  # 0°, 90°, 180°, 270°
        for flip_h in [False, True]:
            for flip_v in [False, True]:
                oriented = np.rot90(shape_np, rotation)
                if flip_h:
                    oriented = np.fliplr(oriented)
                if flip_v:
                    oriented = np.flipud(oriented)

                # Check if unique
                key = oriented.tobytes()
                if key not in seen:
                    seen.add(key)
                    unique.append(oriented)

    return unique


def pad_shape_to_grid(shape_np, grid_h, grid_w):
    """Center a shape in the grid"""
    sh, sw = shape_np.shape
    pad_h = (grid_h - sh) // 2
    pad_w = (grid_w - sw) // 2

    padded = np.pad(shape_np,
                    ((pad_h, grid_h - sh - pad_h),
                     (pad_w, grid_w - sw - pad_w)),
                    mode='constant', constant_values=0)
    return padded


def prepare_input_tensors(grid_np, shape_np, grid_h, grid_w):
    """
    Prepare inputs for MultiChannelPacker
    Returns: grid_tensor, orientation_maps_tensor, orientation_list
    """
    # 1. Get all unique orientations
    orientation_list = get_unique_orientations(shape_np)

    # 2. Convert grid to tensor
    grid_tensor = torch.from_numpy(grid_np).float()
    grid_tensor = grid_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # 3. Create orientation maps (padded to grid size)
    orientation_maps = []
    for orient in orientation_list:
        padded = pad_shape_to_grid(orient, grid_h, grid_w)
        orient_tensor = torch.from_numpy(padded).float()
        orientation_maps.append(orient_tensor.unsqueeze(0))  # Add batch dim

    # 4. Stack all orientation maps
    orientation_maps_tensor = torch.stack(
        orientation_maps, dim=1)  # (1, N, H, W)

    return grid_tensor, orientation_maps_tensor, orientation_list
