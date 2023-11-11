import torch


def get_rays(height: int, width: int, focal_length: float) -> torch.Tensor:
    r"""
    Find origin and direction of rays through every pixel and camera origin.
    """

    # Apply pinhole camera model to gather directions at each pixel
    # create grid
    i, j = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing='ij')

    print(f'i shape: {i.shape}')

    # x, y, z where z is 1 since we're defining a plane that's 1 away
    directions = torch.stack([
        (j - width * .5) / focal_length,
        -(i - height * .5) / focal_length,
        -torch.ones_like(i)
    ], dim=-1)

    return directions


def transform_rays_with_image_orientation(rays, c2w):
    print(f'directions: {rays.shape}')
    print(f'spliced directions: {rays[..., None, :].shape}')
    print(f'c2w: {c2w.shape} to {c2w[:3, :3].shape}')

    # Apply camera pose transformation to newly created pixel directions
    rays_d = torch.matmul(rays, c2w[:3, :3].T)

    # Origin is same for all directions (the optical center)
    rays_o = c2w[:3, -1].expand(rays_d.shape)  # translation part of spatial matrix
    return rays_o, rays_d


def get_meshgrid(height, width):
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32),
        torch.arange(height, dtype=torch.float32),
        indexing='ij')
    return i, j


class CameraToWorldSpatialTransformation:
    def __init__(self, spatial_matrix):
        self.spatial_matrix = spatial_matrix
        self.orientation = spatial_matrix[:3, :3]
        self.translation = spatial_matrix[:3, -1]

    def orient_batch_of_rays(self, rays):
        return torch.matmul(rays, self.orientation.T)

    def expand_translation_to_fit_ray_batches(self, rays):
        return self.translation.expand(rays.shape)
