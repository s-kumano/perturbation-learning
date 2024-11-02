from typing import Sequence, Tuple

import torch
from torch import Tensor


def get_square_meshgrid(
    resolution: int, 
    limits: Sequence[float], 
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    s = torch.linspace(limits[0], limits[1], resolution, device=device) # (resolution,)
    # (resolution, resolution), (resolution, resolution)
    x, y = torch.meshgrid(s, s, indexing='xy')
    '''
    .--->x
    |
    v
    y
    '''
    y = y.flip(0)
    '''
    y
    ^
    |
    .--->x
    '''
    return x, y


def get_meshgrid_vector(
    axis_vec_1: Tensor, # (d,)
    axis_vec_2: Tensor, # (d,)
    coefficient_x: Tensor, # (resolution_y, resolution_x)
    coefficient_y: Tensor, # (resolution_y, resolution_x)
) -> Tensor:
    meshgrid_vector = coefficient_x[..., None] * axis_vec_1[None, None] \
        + coefficient_y[..., None] * axis_vec_2[None, None]
    return meshgrid_vector # (resolution_y, resolution_x, d)