import torch


class CameraToWorldSpatialTransformationManager:
    def __init__(self, spatial_matrix):
        self.spatial_matrix = spatial_matrix
        self.orientation = spatial_matrix[:3, :3]
        self.translation = spatial_matrix[:3, -1]

    # Apply camera pose transformation to newly created pixel directions
    def orient_batch_of_rays(self, rays):
        return torch.matmul(rays, self.orientation.T)

    # Origin is same for all directions (the optical center)
    def expand_translation_to_fit_ray_batches(self, rays):
        return self.translation.expand(rays.shape)


def get_rays(height, width, focal_length, c2w):
    camera_to_world = CameraToWorldSpatialTransformationManager(c2w)
    ray_directions = get_ray_directions(height, width, focal_length)

    rays_d = camera_to_world.orient_batch_of_rays(ray_directions)
    rays_o = camera_to_world.expand_translation_to_fit_ray_batches(ray_directions)
    return rays_o, rays_d


def get_ray_directions(height: int, width: int, focal_length: float) -> torch.Tensor:
    row_meshgrid, col_meshgrid = torch.meshgrid(torch.arange(height), torch.arange(width))
    directions = get_ray_directions_from_meshgrid(row_meshgrid, col_meshgrid, height, width, focal_length)
    print(f'my d: {directions}')
    return directions


def get_ray_directions_from_meshgrid(row_meshgrid, col_meshgrid, height, width, focal_length):
    directions = torch.stack([
        rescale_meshgrid_to_fit_pixels(col_meshgrid, width, focal_length),
        -1*rescale_meshgrid_to_fit_pixels(row_meshgrid, height, focal_length),
        -torch.ones_like(row_meshgrid)
    ], dim=-1)

    return directions


def rescale_meshgrid_to_fit_pixels(meshgrid, length, focal_length):
    return (meshgrid - (.5 * length)) / focal_length
