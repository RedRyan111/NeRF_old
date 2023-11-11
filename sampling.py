import torch
from matplotlib import pyplot as plt
from typing import Optional, Tuple, List, Union, Callable


def sample_stratified(rays_o: torch.Tensor, rays_d: torch.Tensor, starting_bin_distance: float,
                      ending_bin_distance: float, n_samples: int,
                      perturb: Optional[bool] = True) -> torch.Tensor:

    rescaled_bin_edges = torch.linspace(starting_bin_distance, ending_bin_distance, n_samples, device=rays_o.device)
    expanded_rescaled_bin_edges = rescaled_bin_edges.expand((rays_o.shape[0], n_samples))

    bin_width = (ending_bin_distance - starting_bin_distance) / n_samples

    if perturb:
        uniform_sample = torch.rand((rays_d.shape[0], n_samples), device=rays_o.device)
        perturbations = uniform_sample * (bin_width / 2)
        expanded_rescaled_bin_edges = expanded_rescaled_bin_edges + perturbations

    return expanded_rescaled_bin_edges
