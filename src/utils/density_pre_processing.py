import torch

def create_1voxel_window(dims):
    axis = torch.arange(-1,1 + 1e-10, 1, dtype=torch.int32)
    window = torch.stack(torch.meshgrid([axis] * dims, indexing="ij"), dim=-1)
    window = window[(window != 0).any(dim=-1)] # remove the [0,0] offset
    return window.reshape(-1, dims)

def find_edge_indexes(density, level_set=0, epsilon= 1e-1):
    window = create_1voxel_window(len(density.shape))
    density = density.clone()
    density = density.clamp(level_set)
    level_set_indexes = ((density - level_set).abs() < epsilon).nonzero()
    level_set_indexes_plus_window = (level_set_indexes[:,None] + window[None])
    for dim in range(len(density.shape)):
        level_set_indexes_plus_window[..., dim] = level_set_indexes_plus_window[..., dim].clamp(0, density.shape[dim] - 1)
    level_set_indexes_plus_window_density = density[level_set_indexes_plus_window.unbind(dim=-1)]
    is_edge = (level_set_indexes_plus_window_density > level_set).any(dim=-1)
    edge_indexes = level_set_indexes[is_edge]
    edge_indexes = edge_indexes.unique(dim=0) 
    return edge_indexes

def density_pre_processing(density, level_set = 0.01, epsilon=1e-1, diff=0.5):
    dim = len(density.shape)
    window = create_1voxel_window(dim)
    edge_indexes = find_edge_indexes(density, level_set, epsilon)
    density = density.clone()
    density = density.clamp(level_set)

    current_value = level_set - diff
    while len(edge_indexes) > 0:
        density[edge_indexes.unbind(dim=-1)] = current_value
        current_value -= diff

        new_possible_indexes = (edge_indexes[:,None] + window[None])
        for i in range(dim):
            new_possible_indexes[...,i] = new_possible_indexes[...,i].clamp(0, density.shape[i] - 1)
        new_possible_indexes_density = density[new_possible_indexes.unbind(dim=-1)]
        edge_indexes = new_possible_indexes[new_possible_indexes_density == level_set]
        edge_indexes = edge_indexes.unique(dim=0)     

    return density
