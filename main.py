import torch
from matplotlib import pyplot as plt

from data_manager import DataManager
from example_images.positional_encoder import PositionalEncoder
from sampling import sample_stratified
from volume_rendering import transform_rays_with_image_orientation, get_rays
from typing import Optional, Tuple, List, Union, Callable

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

# Draw stratified samples from example
rays_o = ray_origin.view([-1, 3])
rays_d = ray_direction.view([-1, 3])
n_samples = 8
near, far = 2., 6.
perturb = True
inverse_depth = False
with torch.no_grad():
  pts, z_vals = sample_stratified(rays_o, rays_d, near, far, n_samples, perturb=perturb)

print('Input Points')
print('')
print('Distances Along Ray')
print(z_vals.shape)

y_vals = torch.zeros_like(z_vals)

_, z_vals_unperturbed = sample_stratified(rays_o, rays_d, near, far, n_samples, perturb=False)

plt.plot(z_vals_unperturbed[0].cpu().numpy(), 1 + y_vals[0].cpu().numpy(), 'b-o')
plt.plot(z_vals[0].cpu().numpy(), y_vals[0].cpu().numpy(), 'r-o')
plt.ylim([-1, 2])
plt.title('Stratified Sampling (blue) with Perturbation (red)')
ax = plt.gca()
ax.axes.yaxis.set_visible(False)
plt.grid(True)
plt.show()


# Create encoders for points and view directions
encoder = PositionalEncoder(10)
viewdirs_encoder = PositionalEncoder(4)

# Grab flattened points and view directions
pts_flattened = pts.reshape(-1, 3)
viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
flattened_viewdirs = viewdirs[:, None, ...].expand(pts.shape).reshape((-1, 3))

# Encode inputs
encoded_points = encoder(pts_flattened)
encoded_viewdirs = viewdirs_encoder(flattened_viewdirs)

print('Encoded Points')
print(encoded_points.shape)
print(torch.min(encoded_points), torch.max(encoded_points), torch.mean(encoded_points))
print('')

print(encoded_viewdirs.shape)
print('Encoded Viewdirs')
print(torch.min(encoded_viewdirs), torch.max(encoded_viewdirs), torch.mean(encoded_viewdirs))
print('')