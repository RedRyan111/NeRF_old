import torch
from data_manager import DataManager
from volume_rendering import transform_rays_with_image_orientation, get_rays

# For repeatability
# seed = 3407
# torch.manual_seed(seed)
# np.random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

filename = 'tiny_nerf_data.npz'
data_manager = DataManager(filename)

# Gather as torch tensors
example_img_index = data_manager.get_example_index()
example_img = torch.from_numpy(data_manager.images[example_img_index]).to(device)
example_pose = torch.from_numpy(data_manager.poses[example_img_index]).to(device)

# Grab rays from sample image
with torch.no_grad():
    ray_direction = get_rays(
        data_manager.image_height,
        data_manager.image_height,
        data_manager.focal
        )

    ray_origin, ray_direction = transform_rays_with_image_orientation(ray_direction.to(device), example_pose)

print('Ray Origin')
print(ray_origin.shape)
print(ray_origin[data_manager.image_height // 2, data_manager.image_width // 2, :])
print('')

print('Ray Direction')
print(ray_direction.shape)
print(ray_direction[data_manager.image_height // 2, data_manager.image_width // 2, :])
print('')