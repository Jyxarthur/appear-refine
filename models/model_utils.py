import torch
import einops
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def spacetime_flatten(x):
    # input dimension: b t h w c (batch_size frames height width channels)
    # output dimension: b (t*h*w) c
    return torch.reshape(x, [-1, x.shape[1] * x.shape[2] * x.shape[3] , x.shape[-1]])

def spacetime_unflatten(x, frames, resolution):
    # input dimension: b (t*h*w) c  (batch_size frames*height*width channels)
    # output dimension: b t h w c 
    return einops.rearrange(x, 'b (t h w) g_c -> b t h w g_c', t = frames, h = resolution[0], w = resolution[1])

def attn_mask(frames, resolution):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vec = torch.unsqueeze(torch.unsqueeze(torch.arange(frames) + 1, 1), 2)
    vec = vec.expand(frames, resolution[0], resolution[1])
    vec = torch.reshape(vec, [vec.shape[0] * vec.shape[1] * vec.shape[2]])
    mask = (vec.reshape(1, -1) - vec.reshape(-1, 1)) == 0
    return mask.to(device)

def find_time_mask(qq, frames, res):
    vec0 = (torch.arange(frames)).unsqueeze(0).unsqueeze(0).repeat(1, qq, 1)
    vec1 = (torch.arange(frames)).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).repeat(1, 1, res[0], res[1])
    vec0 = einops.rearrange(vec0, 'b qq t -> (qq t) b')
    vec1 = einops.rearrange(vec1, 'b t h w -> b (t h w)')
    mask = (vec1 - vec0) == 0
    return mask

def find_crosstime_mask(qq, local_list, global_list, res):
    vec0 = local_list
    vec1 = global_list.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, res[0], res[1])
    vec0 = einops.rearrange(vec0, 'b qq t -> (qq t) b')
    vec1 = einops.rearrange(vec1, 'b q tt h w -> b (q tt h w)')
    mask = (vec1 - vec0) == 0
    return mask

    
def build_query_grid(num_query):
    # input: q,t (number of queries, number of frames)
    # output dimension: 1 q 1"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ranges = [np.linspace(-1., 1., num=num_query)]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [num_query, -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(grid).to(device)


def build_querytime_grid(num_query, frames):
    # input: q,t (number of queries, number of frames)
    # output dimension: 1 q t 2"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ranges = [np.linspace(-1., 1., num=num_query)] + [np.linspace(-1., 1., num=frames)] 
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [num_query, frames, -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(grid).to(device)

def build_space_grid(resolution):
    # input: h,w (height, width)
    # output dimension: 1 1 1 h w 2"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ranges = [np.linspace(-1., 1., num=res) for res in resolution] 
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = np.expand_dims(grid, axis=0)
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(grid).to(device)

def build_spacetime_grid(frames, resolution):
    # input: t,h,w (number of frames, height, width)
    # output dimension: 1 t h w 3"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ranges = [np.linspace(-1., 1., num=frames)] + [np.linspace(-1., 1., num=res) for res in resolution] 
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [frames, resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(grid).to(device)


class SoftQuerySparseInfo(nn.Module):
    # input dimension: b q c (batch_size query channels)
    # output dimension: b q c
    def __init__(self, hidden_size):
        super(SoftQuerySparseInfo, self).__init__()
        self.proj = nn.Linear(1, hidden_size)
        self.grid = None

    def forward(self, num_query, q):
        grid = build_query_grid(num_query) # 1 q 1
        self.grid = grid[:, 0:q]
        return self.proj(self.grid)

class SoftPositionTimeEmbed(nn.Module):
    # input dimension: b t h w c (batch_size frames height width channels)
    # output dimension: b t h w c
    def __init__(self, hidden_size):
        super(SoftPositionTimeEmbed, self).__init__()
        self.proj = nn.Linear(3, hidden_size)
        self.grid = None 

    def forward(self, inputs, frames, resolution):
        self.grid = build_spacetime_grid(frames, resolution)
        return inputs + self.proj(self.grid)

class SoftPositionEmbed(nn.Module):
    # input dimension: b q t h w c (batch_size frames height width channels)
    # output dimension: b q t h w c
    def __init__(self, hidden_size):
        super(SoftPositionEmbed, self).__init__()
        self.proj = nn.Linear(2, hidden_size)
        self.grid = None

    def forward(self, inputs, resolution):
        self.grid = build_space_grid(resolution)
        return inputs + self.proj(self.grid)

class SoftPositionEmbed2(nn.Module):
    # input dimension: b t h w c (batch_size frames height width channels)
    # output dimension: b t h w c
    def __init__(self, hidden_size):
        super(SoftPositionEmbed2, self).__init__()
        self.proj = nn.Linear(2, hidden_size)
        self.grid = None

    def forward(self, inputs, resolution):
        self.grid = build_space_grid(resolution)[0]
        return inputs + self.proj(self.grid)

class SoftQueryTimeEmbed(nn.Module):
    # input dimension: b q t c (batch_size query frames height width channels)
    # output dimension: b q t c
    def __init__(self, hidden_size):
        super(SoftQueryTimeEmbed, self).__init__()
        self.proj = nn.Linear(2, hidden_size)
        self.grid = None

    def forward(self, inputs, num_query, frames):
        self.grid = build_querytime_grid(num_query, frames)
        return inputs + self.proj(self.grid)

class SoftTimeEmbed(nn.Module):
   #input dimension: b q t c (batch_size frames channels)
   #output dimension: b q t c
    def __init__(self, hidden_size):
        super(SoftTimeEmbed, self).__init__()
        self.proj = nn.Linear(1, hidden_size)
        self.grid = build_time_grid(5)

    def forward(self, inputs, frames):
        if self.grid.size()[1] != frames:
            self.grid = build_time_grid(frames)
        b = inputs.size()[0]
        inputs = einops.rearrange(inputs, 'b q t c -> (b q) t c')
        inputs = inputs + self.proj(self.grid)
        inputs = einops.rearrange(inputs, '(b q) t c -> b q t c', b = b)
        return inputs
