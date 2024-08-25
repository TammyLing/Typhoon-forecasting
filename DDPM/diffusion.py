import math
import copy
from pathlib import Path
import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

import sys
sys.path.append('/vol/bitbucket/zl1823/Typhoon-forecasting/pyphoon2')

from attend import Attend
import matplotlib.pyplot as plt

# from denoising_diffusion_pytorch.version import __version__

import os
import xarray as xr
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def set_seed(seed):
    '''
    Set the seed for reproducibility.
    
    Args:
    - seed: the seed value
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    set_seed(42 + worker_id)

def random_flip_and_rotate(tensors, seed):
    '''
    Randomly flip and rotate the tensors.
    
    Args:
    - tensors: a list of tensors
    - seed: the seed value
    
    Returns:
    - the transformed tensors
    '''
    random.seed(seed)
    if random.random() > 0.5:
        tensors = [torch.flip(tensor, dims=[2]) for tensor in tensors]
    if random.random() > 0.5:
        tensors = [torch.flip(tensor, dims=[1]) for tensor in tensors]
    rotations = random.choice([0, 1, 2, 3])
    tensors = [torch.rot90(tensor, k=rotations, dims=[1, 2]) for tensor in tensors]
    return tensors

def add_noise(tensors, noise_level=0.01):
    '''
    Add random noise to the tensors.
    
    Args:
    - tensors: a list of tensors
    - noise_level: the noise level
    
    Returns:
    - the noisy tensors
    '''
    noisy_tensors = [tensor + noise_level * torch.randn_like(tensor) for tensor in tensors]
    return noisy_tensors

def apply_smoothing(tensors, kernel_size=3):
    '''
    Apply Gaussian smoothing to the tensors.
    
    Args:
    - tensors: a list of tensors
    - kernel_size: the kernel size
    
    Returns:
    - the smoothed tensors
    '''
    smoothed_tensors = [T.GaussianBlur(kernel_size=kernel_size)(tensor) for tensor in tensors]
    return smoothed_tensors

def adjust_contrast(tensors, contrast_factor=0.1):
    '''
    Adjust the contrast of the tensors.
    
    Args:
    - tensors: a list of tensors
    - contrast_factor: the contrast factor
    
    Returns:
    - the contrast-adjusted tensors
    '''
    contrast_tensors = []
    for tensor in tensors:
        channels = tensor.split(1, dim=0)
        adjusted_channels = [T.functional.adjust_contrast(ch, contrast_factor=random.uniform(1-contrast_factor, 1+contrast_factor)) for ch in channels]
        # Combine the channels back into a single tensor
        contrast_tensor = torch.cat(adjusted_channels, dim=0)
        contrast_tensors.append(contrast_tensor)
    return contrast_tensors

def conservative_augmentation(tensors, seed):
    '''
    Apply conservative augmentation to the tensors.
    
    Args:
    - tensors: a list of tensors
    - seed: the seed value
    
    Returns:
    - the augmented
    '''
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Add random noise
    if random.random() > 0.5:
        tensors = add_noise(tensors, noise_level=0.01)
    
    # Apply smoothing
    if random.random() > 0.5:
        tensors = apply_smoothing(tensors, kernel_size=3)
    
    # Adjust contrast
    if random.random() > 0.5:
        tensors = adjust_contrast(tensors, contrast_factor=0.1)
    
    return tensors

class TyphoonDataset(Dataset):
    '''
    A custom dataset class for the typhoon data.
    '''
    def __init__(self, dataset, transform=None, augment=False, seed=42):
        super().__init__()
        self.dataset = dataset

        # get the image data and the target data
        self.image_data = self.dataset['image_data'].values
        self.u10 = self.dataset['u10'].values
        self.v10 = self.dataset['v10'].values
        self.sp = self.dataset['sp'].values
        self.t2m = self.dataset['t2m'].values

        self.transform = transform
        self.augment = augment
        self.seed = seed

        # Resize data to 64x64
        self.image_data_resized = np.array([np.array(Image.fromarray(img).resize((64, 64), Image.BICUBIC)) for img in self.image_data])
        self.u10_resized = np.array([np.array(Image.fromarray(u).resize((64, 64), Image.BICUBIC)) for u in self.u10])
        self.v10_resized = np.array([np.array(Image.fromarray(v).resize((64, 64), Image.BICUBIC)) for v in self.v10])
        self.sp_resized = np.array([np.array(Image.fromarray(sp).resize((64, 64), Image.BICUBIC)) for sp in self.sp])
        self.t2m_resized = np.array([np.array(Image.fromarray(t).resize((64, 64), Image.BICUBIC)) for t in self.t2m])

        # calculate min and max values for normalization
        self.sp_min = np.min(self.sp_resized)
        self.sp_max = np.max(self.sp_resized)
        self.t2m_min = np.min(self.t2m_resized)
        self.t2m_max = np.max(self.t2m_resized)

        self.image_min = np.min(self.image_data_resized)
        self.image_max = np.max(self.image_data_resized)
        self.u10_min = np.min(self.u10_resized)
        self.u10_max = np.max(self.u10_resized)
        self.v10_min = np.min(self.v10_resized)
        self.v10_max = np.max(self.v10_resized)

    def __len__(self):
        return self.image_data.shape[0]

    def __getitem__(self, index):
        # get the resized image and target data
        img = self.image_data_resized[index]

        # Normalize img to the same range as img_64
        img_normalized = (img - self.image_min) / (self.image_max - self.image_min)
        img_normalized = torch.tensor(img_normalized, dtype=torch.float32).unsqueeze(0)

        # get the resized image (already 64x64)
        img_64 = self.image_data_resized[index]
        img_64 = (img_64 - self.image_min) / (self.image_max - self.image_min)
        img_64 = torch.tensor(img_64, dtype=torch.float32).unsqueeze(0)

        # normalize the target data
        u10 = self.u10_resized[index]
        u10 = (u10 - self.u10_min) / (self.u10_max - self.u10_min)
        u10 = torch.tensor(u10, dtype=torch.float32).unsqueeze(0)

        v10 = self.v10_resized[index]
        v10 = (v10 - self.v10_min) / (self.v10_max - self.v10_min)
        v10 = torch.tensor(v10, dtype=torch.float32).unsqueeze(0)

        sp = self.sp_resized[index]
        sp = (sp - self.sp_min) / (self.sp_max - self.sp_min)
        sp = torch.tensor(sp, dtype=torch.float32).unsqueeze(0)

        t2m = self.t2m_resized[index]
        t2m = (t2m - self.t2m_min) / (self.t2m_max - self.t2m_min)
        t2m = torch.tensor(t2m, dtype=torch.float32).unsqueeze(0)

        original_u10 = self.u10[index].astype(np.float32)
        original_u10 = (original_u10 - self.u10_min) / (self.u10_max - self.u10_min)
        original_u10 = torch.tensor(original_u10, dtype=torch.float32).unsqueeze(0)

        original_v10 = self.v10[index].astype(np.float32)
        original_v10 = (original_v10 - self.v10_min) / (self.v10_max - self.v10_min)
        original_v10 = torch.tensor(original_v10, dtype=torch.float32).unsqueeze(0)

        original_sp = self.sp[index].astype(np.float32)
        original_sp = (original_sp - self.sp_min) / (self.sp_max - self.sp_min)
        original_sp = torch.tensor(original_sp, dtype=torch.float32).unsqueeze(0)

        original_t2m = self.t2m[index].astype(np.float32)
        original_t2m = (original_t2m - self.t2m_min) / (self.t2m_max - self.t2m_min)
        original_t2m = torch.tensor(original_t2m, dtype=torch.float32).unsqueeze(0)

        # copy the image to all channels
        img_64 = torch.cat((img_64, img_64, img_64, img_64), dim=0)

        # data augmentation
        if self.augment:
            tensors = [img_normalized, img_64, u10, v10, sp, t2m, original_u10, original_v10, original_sp, original_t2m]
            tensors = conservative_augmentation(tensors, self.seed + index)
            img_normalized, img_64, u10, v10, sp, t2m, original_u10, original_v10, original_sp, original_t2m = tensors

        return img_normalized, img_64, u10, v10, sp, t2m, original_u10, original_v10, original_sp, original_t2m

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions
def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules
def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * self.scale

# sinusoidal positional embeds
class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(Module):
    def __init__(self, dim, dim_out, dropout = 0.3):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, dropout = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

class Unet(Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 4,
        self_condition = False,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        dropout = 0.3,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = None,    # defaults to full attention only for inner most layer
        flash_attn = False
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        # prepare blocks

        FullAttention = partial(Attention, flash = flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim = time_dim, dropout = dropout)

        # layers

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond = None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_v',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = False,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not hasattr(model, 'random_or_learned_sinusoidal_cond') or not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        assert isinstance(image_size, (tuple, list)) and len(image_size) == 2, 'image size must be a integer or a tuple/list of two integers'
        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond = None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps = False):
        batch, device = shape[0], self.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def ddim_sample(self, shape, return_all_timesteps = False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(self, batch_size = 16, return_all_timesteps = False):
        (h, w), channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, h, w), return_all_timesteps = return_all_timesteps)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img
    
    @torch.inference_mode()
    def p_sample_loop_from_dt(self, input_img, start_step=None, return_all_timesteps=False):
        batch, device = input_img.shape[0], input_img.device

        # initialize the image
        img = input_img.to(device)
        imgs = [img]

        x_start = None

        # if start_step is None, start from the 0.8 of the total timesteps
        if start_step is None:
            start_step = int(self.num_timesteps * 0.8)

        t_noise = torch.tensor([start_step] * batch, device=device, dtype=torch.int64)
        noise = torch.randn_like(img)
        img = self.q_sample(img, t_noise, noise)

        # reverse diffusion process
        for t in tqdm(reversed(range(0, start_step + 1)), desc='sampling loop time step', total=start_step + 1):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)

        ret = self.unnormalize(ret)
        return ret


    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None, offset_noise_strength = None):
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
        
        if model_out.shape != target.shape:
            target = target.squeeze(1)

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)


# trainer class

class Trainer:
    def __init__(
        self,
        diffusion_model,
        train_dataloader,
        test_dataloader,
        dataset,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results3',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        convert_image_to = None,
        calculate_fid = True,
        inception_block_idx = 2048,
        max_grad_norm = 1.,
        num_fid_samples = 50000,
        save_best_and_latest_only = False
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling

        # default convert_image_to depending on channels

        if not exists(convert_image_to):
            convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (train_batch_size * gradient_accumulate_every) >= 8, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 8 or above'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        # dataset and dataloader
        train_dl = train_dataloader
        train_dl = self.accelerator.prepare(train_dl)
        self.train_dl = cycle(train_dl)

        test_dl = test_dataloader
        self.test_dl = self.accelerator.prepare(test_dl)

        self.dataset = dataset

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # FID-score computation

        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            from fid_evaluation import FIDEvaluation

            if not is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming."\
                    "Consider using DDIM sampling to save time."
                )

            self.fid_scorer = FIDEvaluation(
                batch_size=self.batch_size,
                dl=self.train_dl,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir=results_folder,
                device=self.device,
                num_fid_samples=num_fid_samples,
                inception_block_idx=inception_block_idx
            )

        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10 # infinite
            self.best_mse_score = 1e10 

        self.save_best_and_latest_only = save_best_and_latest_only

        self.inference_counter = 1

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                self.model.train()

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.train_dl)
                    data = [d.to(device) for d in data]
                    
                    # get the data
                    img, img_64, u10, v10, sp, t2m, original_u10, original_v10, original_sp, original_t2m = data
                    
                    # concatenate u10, v10, sp, t2m to form a four-channel input
                    input_tensor = torch.cat((u10, v10, sp, t2m), dim=1)

                    with self.accelerator.autocast():
                        loss = self.model(input_tensor)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()

                        u10_dir = os.path.join(self.results_folder, 'U10')
                        v10_dir = os.path.join(self.results_folder, 'V10')
                        sp_dir = os.path.join(self.results_folder, 'Pressure')
                        t2m_dir = os.path.join(self.results_folder, 'Temperature')

                        os.makedirs(u10_dir, exist_ok=True)
                        os.makedirs(v10_dir, exist_ok=True)
                        os.makedirs(sp_dir, exist_ok=True)
                        os.makedirs(t2m_dir, exist_ok=True)

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        all_images = torch.cat(all_images_list, dim = 0)

                        image = all_images[0]
                        u10_image = image[0, :, :].unsqueeze(0)  
                        v10_image = image[1, :, :].unsqueeze(0)  
                        sp_image = image[2, :, :].unsqueeze(0)  
                        t2m_image = image[3, :, :].unsqueeze(0)  

                        utils.save_image(u10_image, os.path.join(u10_dir, f'sample-{milestone}.png'))
                        utils.save_image(v10_image, os.path.join(v10_dir, f'sample-{milestone}.png'))
                        utils.save_image(sp_image, os.path.join(sp_dir, f'sample-{milestone}.png'))
                        utils.save_image(t2m_image, os.path.join(t2m_dir, f'sample-{milestone}.png'))

                        self.save(milestone)
                        
                        test_loss = self.inference(f'model-{milestone}')
                        print(f"Test Loss: {test_loss:.4f}")


                pbar.update(1)

        accelerator.print('training complete')

    def evaluate(self):
        loss_fn = nn.MSELoss()
        total_loss = 0
        num_batches = 0
        with torch.inference_mode():
            for data in self.val_dl:
                img, img_64, u10, v10, sp, t2m, original_u10, original_v10, original_sp, original_t2m = data
                    
                input_tensor = img_64

                batches = num_to_groups(self.num_samples, self.batch_size)
                all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                all_images = torch.cat(all_images_list, dim = 0)

                for i in range(input_tensor.size(0)):
                    # pred_img, _ = self.ema.ema_model.p_sample(noisy_input[i].unsqueeze(0), t[i].item())
                    pred_img = all_images[i].squeeze(0)
                    u10_i = u10[i].squeeze(0)
                    v10_i = v10[i].squeeze(0)
                    sp_i = sp[i].squeeze(0)
                    t2m_i = t2m[i].squeeze(0)
                    
                    loss = 0
                    loss += loss_fn(pred_img[0, :, :], u10_i).item()
                    loss += loss_fn(pred_img[1, :, :], v10_i).item()
                    loss += loss_fn(pred_img[2, :, :], sp_i).item()
                    loss += loss_fn(pred_img[3, :, :], t2m_i).item()
                    
                    total_loss += loss
                    num_batches += 1

        average_loss = total_loss / num_batches
        print("Evaluation loss:", average_loss)
        sys.stdout.flush()
        return average_loss

    def inference(self, model_name=None):
        self.model.eval()

        if model_name:
            results_folder = os.path.join(self.results_folder, 'test', model_name)
        else:
            results_folder = os.path.join(self.results_folder, 'test')
        os.makedirs(results_folder, exist_ok=True)

        total_loss = 0
        num_batches = 0
        loss_fn = nn.MSELoss()

        with torch.inference_mode():
            for batch_idx, data in enumerate(self.test_dl):
                if batch_idx > 0:
                    break

                img, img_64, u10, v10, sp, t2m, original_u10, original_v10, original_sp, original_t2m = data
                targets = [u10, v10, sp, t2m]
                targets = [t.to(self.device) for t in targets]

                input_tensor = img_64.to(self.device)
              
                all_images = self.ema.ema_model.p_sample_loop_from_dt(input_tensor)

                for i in range(input_tensor.size(0)):
                    pred_img = all_images[i]

                    for j, target_name in enumerate(['U10', 'V10', 'Pressure', 'Temperature']):
                        prediction = pred_img[j].cpu().numpy().squeeze()
                        true_value = targets[j][i].cpu().numpy().squeeze()

                        
                        prediction_tensor = torch.tensor(prediction).to(self.device)
                        true_value_tensor = torch.tensor(true_value).to(self.device)
                        loss = loss_fn(prediction_tensor, true_value_tensor)
                        total_loss += loss.item()
                        num_batches += 1


                        input_image = img_64[i][j].cpu().numpy().squeeze()
                        input_image = input_image * (self.dataset.u10_max - self.dataset.u10_min) + self.dataset.u10_min


                        if target_name == 'U10':
                            prediction = prediction * (self.dataset.u10_max - self.dataset.u10_min) + self.dataset.u10_min
                            true_value = true_value * (self.dataset.u10_max - self.dataset.u10_min) + self.dataset.u10_min
                        elif target_name == 'V10':
                            prediction = prediction * (self.dataset.v10_max - self.dataset.v10_min) + self.dataset.v10_min
                            true_value = true_value * (self.dataset.v10_max - self.dataset.v10_min) + self.dataset.v10_min
                        elif target_name == 'Pressure':
                            prediction = prediction * (self.dataset.sp_max - self.dataset.sp_min) + self.dataset.sp_min
                            true_value = true_value * (self.dataset.sp_max - self.dataset.sp_min) + self.dataset.sp_min
                        else:
                            prediction = prediction * (self.dataset.t2m_max - self.dataset.t2m_min) + self.dataset.t2m_min
                            true_value = true_value * (self.dataset.t2m_max - self.dataset.t2m_min) + self.dataset.t2m_min

                        prediction_resized = np.array(Image.fromarray(prediction).resize((512, 512), Image.BICUBIC))

                        lon_start, lon_end = 116.0794, 126.0794
                        lat_start, lat_end = 18.9037, 28.9037
                        lon = np.linspace(lon_start, lon_end, true_value.shape[0])
                        lat = np.linspace(lat_start, lat_end, true_value.shape[1])
                        lon, lat = np.meshgrid(lon, lat)

                        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
                        ax = axs[0]
                        input_plot = ax.imshow(img_64[i][j].cpu().numpy(), extent=[lon_start, lon_end, lat_start, lat_end], cmap='jet', aspect='equal')
                        ax.set_title('Input Image')
                        fig.colorbar(input_plot, ax=ax)
                        ax.set_xlabel('Longitude')
                        ax.set_ylabel('Latitude')
                        
                        ax = axs[1]
                        true_plot = ax.imshow(true_value, extent=[lon_start, lon_end, lat_start, lat_end], cmap='jet', aspect='equal')
                        ax.set_title('True Value')
                        fig.colorbar(true_plot, ax=ax)

                        ax = axs[2]
                        pred_plot = ax.imshow(prediction, extent=[lon_start, lon_end, lat_start, lat_end], cmap='jet', aspect='equal')
                        ax.set_title('Predicted Value')
                        fig.colorbar(pred_plot, ax=ax)

                        plt.tight_layout()
                        plt.savefig(os.path.join(results_folder, f'diffusion_{target_name}_{i}.png'))
                        plt.close(fig)
        average_loss = total_loss / num_batches
        return average_loss


def load_all_zarr_files(directory):
    """
    Load all zarr files in the specified directory and combine them into a single xarray dataset.
    
    Args:
    - directory: The directory that contains the zarr files.
    
    Returns:
    - A combined xarray dataset.
    """
    datasets = []
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            if dir.endswith(".zarr"):
                zarr_path = os.path.join(root, dir)
                print(f"Found zarr directory: {zarr_path}")
                sys.stdout.flush()
                ds = xr.open_zarr(zarr_path)
                datasets.append(ds)
    if not datasets:
        raise ValueError("No zarr files found in the specified directory。")
    combined_dataset = xr.concat(datasets, dim="time")
    return combined_dataset


if __name__ == '__main__':
    # Load the dataset
    set_seed(42)  
    base_directory = os.path.join('/vol', 'bitbucket', 'zl1823', 'Typhoon-forecasting', 'dataset', 'pre_processed_dataset', 'ERA5_without_nan_taiwan')
    combined_dataset = load_all_zarr_files(base_directory)

    # split 80% of the data for training and 20% for testing
    dataset = TyphoonDataset(combined_dataset)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size 
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    augmentation_ratio = 0.5
    augmented_size = int(train_size * augmentation_ratio)
    train_indices = train_dataset.indices
    train_combined_data = combined_dataset.isel(time=train_indices)
    augmented_dataset = TyphoonDataset(train_combined_data, augment=True)
    augmented_dataset = random_split(augmented_dataset, [augmented_size, len(augmented_dataset) - augmented_size], generator=torch.Generator().manual_seed(42))[0]

    # concatenate the original training dataset and the augmented dataset
    train_dataset = ConcatDataset([train_dataset, augmented_dataset])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, worker_init_fn=worker_init_fn)

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the loss function
    losses = [nn.MSELoss() for _ in range(4)]

    # Initialize the diffusion model
    model = GaussianDiffusion(
        Unet(dim=64, channels=4),
        image_size=(64, 64)
    )

    # Initialize the trainer
    trainer = Trainer(
        diffusion_model=model,
        train_dataloader=train_loader,
        test_dataloader = test_loader,
        dataset = dataset,
        train_batch_size=8,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        save_and_sample_every=2000,
        num_samples=25,
        results_folder='./results3',
        amp=True,
        mixed_precision_type='fp16',
        split_batches=True,
        calculate_fid=True,
        max_grad_norm=1.,
        save_best_and_latest_only=False
    )

    # Train the model
    trainer.train()

    # # Evaluate the model
    # print(f"Evaluating model:")
    # trainer.model.load_state_dict(torch.load('diffusion_model.pth'))

    # visualize_diffusion(test_loader, trainer.model, device, dataset, num_timesteps=1000)

    print('Script finished running')