import torch


def get_rays(height: int, width: int, focal_length: float) -> torch.Tensor:
    row_meshgrid, col_meshgrid = get_meshgrid(height, width)
    directions = get_ray_directions_from_meshgrid(row_meshgrid, col_meshgrid, height, width, focal_length)
    return directions


def transform_rays_with_image_orientation(rays, camera_to_world_spatial_transformation):

    camera_to_world_obj = CameraToWorldSpatialTransformationManager(camera_to_world_spatial_transformation)

    rays_d = camera_to_world_obj.orient_batch_of_rays(rays)
    rays_o = camera_to_world_obj.expand_translation_to_fit_ray_batches(rays)

    return rays_o, rays_d


def get_meshgrid(height, width):
    row_meshgrid, col_meshgrid = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing='ij')
    return row_meshgrid, col_meshgrid


def get_ray_directions_from_meshgrid(row_meshgrid, col_meshgrid, height, width, focal_length):
    directions = torch.stack([
        rescale_meshgrid_to_fit_pixels(col_meshgrid, width, focal_length),
        rescale_meshgrid_to_fit_pixels(row_meshgrid, height, focal_length),
        -torch.ones_like(row_meshgrid)
    ], dim=-1)

    return directions


def rescale_meshgrid_to_fit_pixels(meshgrid, length, focal_length):
    return (meshgrid - (.5 * length)) / focal_length


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
