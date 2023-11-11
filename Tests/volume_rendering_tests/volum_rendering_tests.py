import torch
from typing import Optional, Tuple
import unittest

torch.set_default_dtype(torch.float32)
from volume_rendering import *


def get_rays_org(
        height: int,
        width: int,
        focal_length: float,
        c2w: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
  Find origin and direction of rays through every pixel and camera origin.
  """

    # Apply pinhole camera model to gather directions at each pixel
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32).to(c2w),
        torch.arange(height, dtype=torch.float32).to(c2w),
        indexing='ij')
    i, j = i.transpose(-1, -2), j.transpose(-1, -2)
    directions = torch.stack([(i - width * .5) / focal_length,
                              -(j - height * .5) / focal_length,
                              -torch.ones_like(i)
                              ], dim=-1)

    print(f'd: {directions}')

    # Apply camera pose to directions
    rays_d = torch.sum(directions[..., None, :] * c2w[:3, :3], dim=-1)

    # Origin is same for all directions (the optical center)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def test_rays():
    height = 4
    width = 5
    focal_length = 128
    c2w = torch.tensor([[1, 2, 3, 2], [1, 4, 2, 1], [2, 5, 3, 1], [0, 0, 0, 1]], dtype=torch.float32)
    # print(c2w)
    # print(c2w[:3, :3])

    org_ray_o, org_ray_d = get_rays_org(height, width, focal_length, c2w)
    my_ray_o, my_ray_d = get_rays(height, width, focal_length, c2w)

    # print(f'test if equal')
    # print(f'original origin: {org_ray_o.shape} mine: {my_ray_o.shape}')
    # print(f'original direction: {org_ray_d.shape} mine: {my_ray_d.shape}')

    # print('')
    # print(org_ray_d)
    # print('')
    # print(my_ray_d)

    assert torch.prod(torch.eq(org_ray_o, my_ray_o))
    assert torch.prod(torch.eq(org_ray_d, my_ray_d))

    print(f'Equal: {torch.eq(org_ray_o, my_ray_o)}')
