
"""
final_puzzle_pieces.py
Converted from finalPuzzlePieces.ipynb
"""

# =============================================================
# Imports
# =============================================================
import math
import numpy as np
import os
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch_geometric
import matplotlib.pyplot as plt

from abc import abstractmethod
from typing import Callable, Optional, Tuple
from ase import Atoms
from ase.neighborlist import neighbor_list
from e3nn import o3
from e3nn.io import CartesianTensor
from e3nn.nn import FullyConnectedNet
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

EPOCHS = 300

def make_lr_lambda(warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda

def ddp_setup(rank: int, world_size: int):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=torch.device(f"cuda:{rank}"))

def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0

# =============================================================
# Device
# =============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device Used: {device}")


# =============================================================
# Forward (noising) processes
# =============================================================
class AtomDDIMForward(nn.Module):
    def __init__(self,
                 alpha_min: float = 0.001,
                 alpha_max: float = 1.0,
                 T: int = 1000,
                 num_classes: int = 89,
                 prior=None):
        super().__init__()
        self.T = T
        self.num_classes = num_classes
        self.register_buffer("alpha",
                             torch.linspace(start=alpha_max, end=alpha_min, steps=T))
        if prior is None:
            prior = torch.full((num_classes,), 1 / num_classes)
        self.register_buffer("mu", prior.view(1, -1))

    def forward(self, batch):
        t = torch.randint(1, self.T, (1,))
        a = self.alpha[t].view(1, 1)
        x0 = batch.x.float()
        x_t = a * x0 + (1 - a) * self.mu
        return x_t, t, batch.x


class CoordinateDDIMForward(nn.Module):
    def __init__(self, sigma_min=1e-3, sigma_max=1.0, T=1000, density_scaled=True, device=None):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.density_scaled = density_scaled
        self.T = T
        self.device = device

    def sigma(self, t):
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def forward(self, batch):
        t = torch.randint(low=1, high=self.T, size=(1,))
        t_frac = t.float() / self.T
        sigma_t = self.sigma(t_frac).to(self.device)
        if self.density_scaled:
            n = torch.bincount(batch.batch)[batch.batch]
            sigma_t = sigma_t / n.float().pow(1 / 3)
        eps = torch.randn_like(batch.pos)
        x_t = (batch.pos + eps * sigma_t.view(-1, 1)) % 1.0
        return x_t, t, eps


def compute_lattice_constants(dataset):
    dets, ns = [], []
    for batch in tqdm(dataset):
        det = torch.linalg.det(batch.cell).abs()
        n = torch.bincount(batch.batch).float()
        dets.append(det)
        ns.append(n)
    det = torch.cat(dets)
    n = torch.cat(ns)
    c_const = 1.0 / (n / det).mean()
    v_const = torch.var(det / n)
    return c_const, v_const


class LatticeForwardDDIM(nn.Module):
    def __init__(self, alpha_min=0.001, alpha_max=1.0, T=1000):
        super().__init__()
        self.T = T
        self.register_buffer("alpha_bar",
                             torch.cumprod(torch.linspace(start=alpha_max, end=alpha_min, steps=self.T), 0))

    def mu_n(self, c, n):
        return (c * n).pow(1 / 3)

    def sigma_n(self, v, n):
        return (v * n).pow(1 / 3)

    def forward(self, batch, c_const, v_const):
        t = torch.randint(1, self.T, (1,))
        a = self.alpha_bar[t].view(-1, 1, 1)
        n = torch.bincount(batch.batch)
        mu = self.mu_n(c_const, n).view(-1, 1, 1)
        sig = self.sigma_n(v_const, n).view(-1, 1, 1)
        eye = torch.eye(3, device=batch.cell.device).unsqueeze(0)
        eps = torch.randn_like(batch.cell)
        L_t = (torch.sqrt(a) * batch.cell
               + (1 - torch.sqrt(a)) * mu * eye
               + (1 - a) * sig * eps)
        return L_t, t, eps


# =============================================================
# Radial basis + graph builder
# =============================================================
class GaussianRBF(nn.Module):
    def __init__(self, radial_embed_size: int = 64, cutoff: float = 7.0,
                 start: float = 0.0, trainable: bool = False):
        super().__init__()
        radial_embed_size = int(radial_embed_size)
        start = float(start)
        cutoff = float(cutoff)
        centers = torch.linspace(start, cutoff, radial_embed_size)
        width = (centers[1] - centers[0]).item()
        self.gamma = 1.0 / (2 * width ** 2)
        if trainable:
            self.centers = nn.Parameter(centers)
        else:
            self.register_buffer("centers", centers)

    def forward(self, edge_len):
        d = edge_len.unsqueeze(-1)
        return torch.exp(-self.gamma * (d - self.centers) ** 2)


def make_graph_builder(cutoff=5.0, irreps_edge="1x0e+1x1o+1x2e", radial_fn=None):
    irreps_edge = o3.Irreps(irreps_edge)

    def _build(pos, cell, node_batch, atom_z, fractional=True):
        device = pos.device
        if node_batch is None:
            node_batch = torch.zeros(pos.shape[0], dtype=torch.long, device=device)

        all_i, all_j, all_vec, all_len = [], [], [], []
        offset = 0
        for b in range(int(node_batch.max().item()) + 1):
            mask = node_batch == b
            pos_b = pos[mask].detach().cpu().numpy()
            cell_b = cell[b].detach().cpu().numpy()
            z_b = atom_z[mask].detach().cpu().numpy()

            if fractional:
                atoms = Atoms(numbers=z_b, scaled_positions=pos_b % 1.0,
                              cell=cell_b, pbc=True)
            else:
                atoms = Atoms(numbers=z_b, positions=pos_b,
                              cell=cell_b, pbc=True)
            
            cell_det = abs(np.linalg.det(cell_b))
            widths = []
            for k in range(3):
                a1 = cell_b[(k + 1) % 3]
                a2 = cell_b[(k + 2) % 3]
                face_area = np.linalg.norm(np.cross(a1, a2))
                widths.append(cell_det / (face_area + 1e-12))
            min_width = min(widths)
            
            bad_cell = ((not np.isfinite(cell_b).all()) or cell_det < 1e-3 or min_width < cutoff)
            
            if bad_cell:
                offset += int(mask.sum().item())
                continue
                
            try:
                i, j, D, d = neighbor_list("ijDd", atoms, cutoff=cutoff)
            except (ValueError, np.linalg.LinAlgError):
                offset += int(mask.sum().item())
                continue

            all_i.append(torch.as_tensor(i, dtype=torch.long) + offset)
            all_j.append(torch.as_tensor(j, dtype=torch.long) + offset)
            all_vec.append(torch.as_tensor(np.asarray(D), dtype=torch.float32))
            all_len.append(torch.as_tensor(np.asarray(d), dtype=torch.float32))
            offset += int(mask.sum().item())
            
        if len(all_i) == 0:
            empty_i = torch.empty(0, dtype=torch.long, device=device)
            edge_index = torch.stack([empty_i, empty_i])
            edge_vec = torch.zeros(0, 3, device=device)
            edge_len = torch.zeros(0, device=device)
        else:
            edge_index = torch.stack([torch.cat(all_i), torch.cat(all_j)]).to(device)
            edge_vec = torch.cat(all_vec).to(device)
            edge_len = torch.cat(all_len).to(device)

        edge_embed = o3.spherical_harmonics(
            irreps_edge, edge_vec, normalize=True, normalization="component"
        )
        if radial_fn is not None:
            edge_feat = radial_fn(edge_len)   # [E, radial_embed_size]
        else:
            edge_feat = edge_len.unsqueeze(-1)  # [E, 1] fallback

        return edge_index, edge_feat, edge_embed

    return _build


graph_builder = make_graph_builder(
    radial_fn=GaussianRBF(radial_embed_size=64, cutoff=5.0, start=0.0, trainable=False).to(device)
)


# =============================================================
# Helper functions for the equivariant network
# =============================================================
def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def scatter_sum(src: torch.Tensor,
                index: torch.Tensor,
                dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def scatter_add(src: torch.Tensor,
                index: torch.Tensor,
                dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    return scatter_sum(src, index, dim, out, dim_size)


def scatter_softmax(src: torch.Tensor,
                    index: torch.Tensor,
                    dim: int = -1,
                    dim_size: Optional[int] = None) -> torch.Tensor:
    if not torch.is_floating_point(src):
        raise ValueError('`scatter_softmax` can only be computed over tensors '
                         'with floating point data types.')

    index = broadcast(index, src, dim)
    max_value_per_index = scatter_max(src, index, dim=dim, dim_size=dim_size)[0]
    max_per_src_element = max_value_per_index.gather(dim, index)
    recentered_scores = src - max_per_src_element
    recentered_scores_exp = recentered_scores.exp_()
    sum_per_index = scatter_sum(recentered_scores_exp, index, dim, dim_size=dim_size)
    normalizing_constants = sum_per_index.gather(dim, index)
    return recentered_scores_exp.div(normalizing_constants)


def scatter_max(src, index, dim=-1, out=None, dim_size=None):
    index = broadcast(index, src, dim)
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0
    size = list(src.size())
    size[dim] = dim_size
    out = torch.full(size, float('-inf'), dtype=src.dtype, device=src.device)
    out = out.scatter_reduce(dim, index, src, reduce="amax", include_self=True)
    return out, None


class TimestepBlock(nn.Module):
    """Any module where forward() takes timestep embeddings as a second argument."""
    @abstractmethod
    def forward(self, x, emb):
        """Apply the module to `x` given `emb` timestep embeddings."""


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """A sequential module that passes timestep embeddings to children that support it."""
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class EGCN(nn.Module):
    def __init__(self,
                 irreps_in,
                 irreps_out,
                 irreps_edge,
                 radial_embed_size,
                 num_radial_layer,
                 radial_hidden_size,
                 use_sc=True,
                 irrep_normalization='component',
                 path_normalization='element'):
        super(EGCN, self).__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_edge = o3.Irreps(irreps_edge)
        self.radial_embed_size = radial_embed_size
        self.num_radial_layer = num_radial_layer
        self.radial_hidden_size = radial_hidden_size
        self.use_sc = use_sc

        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_in, self.irreps_edge, self.irreps_out,
            internal_weights=False, shared_weights=False,
            irrep_normalization=irrep_normalization,
            path_normalization=path_normalization
        )

        self.fc = FullyConnectedNet(
            [radial_embed_size] + num_radial_layer * [radial_hidden_size] + [self.tp.weight_numel],
            torch.nn.functional.silu
        )
        self.sc = None
        if self.use_sc:
            self.sc = o3.Linear(self.irreps_in, self.irreps_out)

    def forward(self, feature, edge_index, edge_feat, edge_embed, dim_size=None):
        src, dst = edge_index
        weight = self.fc(edge_feat)
        out = self.tp(feature[src], edge_embed, weight=weight)
        out = scatter_sum(out, dst, dim=0, dim_size=feature.shape[0])
        if self.use_sc:
            out += self.sc(feature)
        return out


class NormActivation(nn.Module):
    def __init__(self, irreps_in, act_scalars=torch.nn.functional.silu, act_vectors=torch.sigmoid):
        super(NormActivation, self).__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.scalar_irreps = self.irreps_in[0:1]
        self.vector_irreps = self.irreps_in[1:]
        self.act_scalars = act_scalars
        self.act_vectors = act_vectors
        self.scalar_idx = self.irreps_in[0].mul

        inner_out = o3.Irreps([(mul, (0, 1)) for mul, _ in self.vector_irreps])

        self.inner_prod = o3.TensorProduct(
            self.vector_irreps, self.vector_irreps, inner_out, [
                (i, i, i, 'uuu', False) for i in range(len(self.vector_irreps))
            ]
        )
        self.mul = o3.ElementwiseTensorProduct(inner_out, self.vector_irreps)

    def forward(self, features):
        scalars = self.act_scalars(features[..., :self.scalar_idx])
        vectors = features[..., self.scalar_idx:]
        norm = torch.sqrt(self.inner_prod(vectors, vectors) + 1e-8)
        act = self.act_vectors(norm)
        vectors_out = self.mul(act, vectors)
        return torch.cat([scalars, vectors_out], dim=-1)


class ResidualBlock(TimestepBlock):
    def __init__(self,
                 irreps_in,
                 irreps_out,
                 irreps_edge,
                 time_channels,
                 radial_embed_size,
                 num_radial_layer,
                 radial_hidden_size,
                 use_sc=True,
                 irrep_normalization='component',
                 path_normalization='element'):
        super().__init__()

        self.irreps_in = o3.Irreps(irreps_in)

        self.norm_1 = NormActivation(irreps_in=irreps_in)
        self.conv_1 = EGCN(irreps_in=irreps_in,
                           irreps_out=irreps_in,
                           irreps_edge=irreps_edge,
                           radial_embed_size=radial_embed_size,
                           num_radial_layer=num_radial_layer,
                           radial_hidden_size=radial_hidden_size,
                           use_sc=use_sc,
                           irrep_normalization=irrep_normalization,
                           path_normalization=path_normalization)

        # --- Equivariant timestep injection -------------------------------
        # Produce ONE scalar gain per irrep instance (sum of multiplicities).
        # Scaling an irrep by a scalar preserves equivariance.
        self.num_irrep_scalars = sum(mul for mul, _ in self.irreps_in)
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, self.num_irrep_scalars)
        )
        # ------------------------------------------------------------------

        self.norm_2 = NormActivation(irreps_in=irreps_out)
        self.conv_2 = EGCN(irreps_in=irreps_in,
                           irreps_out=irreps_out,
                           irreps_edge=irreps_edge,
                           radial_embed_size=radial_embed_size,
                           num_radial_layer=num_radial_layer,
                           radial_hidden_size=radial_hidden_size,
                           use_sc=use_sc,
                           irrep_normalization=irrep_normalization,
                           path_normalization=path_normalization)

        if o3.Irreps(irreps_in) != o3.Irreps(irreps_out):
            self.shortcut = EGCN(irreps_in=irreps_in,
                                 irreps_out=irreps_out,
                                 irreps_edge=irreps_edge,
                                 radial_embed_size=radial_embed_size,
                                 num_radial_layer=num_radial_layer,
                                 radial_hidden_size=radial_hidden_size,
                                 use_sc=use_sc,
                                 irrep_normalization=irrep_normalization,
                                 path_normalization=path_normalization)
        else:
            self.shortcut = nn.Identity()

    def _broadcast_gains(self, gains):
        """
        Expand per-irrep scalar gains [N, num_irrep_scalars] to the full
        feature dimension [N, irreps_in.dim] by repeating each gain across
        the (2l+1) components of its irrep.
        """
        repeats = []
        for mul, ir in self.irreps_in:
            # each of the `mul` instances spans (2l+1) components
            repeats.extend([ir.dim] * mul)
        repeats = torch.tensor(repeats, device=gains.device)
        return torch.repeat_interleave(gains, repeats, dim=-1)

    def forward(self, feature, edge_index, edge_feat, edge_embed, t, dim_size=None):
        """
        `feature` : [N, irreps_in.dim] node features (steerable)
        `t`       : [N, time_channels] (or [batch, time_channels] gathered to nodes)
        """
        # print(f"Start of ResidualBlock")
        h = self.conv_1(feature, edge_index, edge_feat, edge_embed, dim_size=dim_size)
        h = self.norm_1(h)

        # --- Equivariant timestep modulation (FiLM-style scaling) ---------
        gains = self.time_emb(t)                 # [N, num_irrep_scalars]
        gains = self._broadcast_gains(gains)     # [N, irreps_in.dim]
        h = h * gains                            # scalar scaling -> equivariant
        # ------------------------------------------------------------------

        h = self.conv_2(h, edge_index, edge_feat, edge_embed, dim_size=dim_size)
        h = self.norm_2(h)

        if isinstance(self.shortcut, EGCN):
            sc = self.shortcut(feature, edge_index, edge_feat, edge_embed, dim_size=dim_size)
        else:
            sc = self.shortcut(feature)
        # print(f"End of ResidualBlock")

        return h + sc  # Residual block


class AttentionBlock(nn.Module):
    """
    E(3)-equivariant self-attention over graph nodes.

    - Q, K, V are produced as steerable (irreps) features via EGCN.
    - Attention logits are INVARIANT scalars built from per-irrep inner
      products of Q and K  ->  softmax weights are rotation-invariant.
    - Output is a scalar-weighted sum of the V irreps  ->  equivariant.
    """
    def __init__(self,
                 irreps_in,
                 irreps_edge,
                 radial_embed_size,
                 num_radial_layer,
                 radial_hidden_size,
                 num_heads=1,
                 use_sc=True,
                 irrep_normalization='component',
                 path_normalization='element'):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.num_heads = num_heads

        # feature dim must split evenly across heads
        assert self.irreps_in.dim % num_heads == 0

        self.norm = NormActivation(irreps_in=irreps_in)

        # Produce Q, K, V as irreps. We request 3x the input irreps so each
        # of Q/K/V keeps the same irrep structure as the input.
        irreps_qkv = (self.irreps_in * 3).sort().irreps.simplify()
        self.to_qkv = EGCN(irreps_in=irreps_in,
                           irreps_out=irreps_qkv,
                           irreps_edge=irreps_edge,
                           radial_embed_size=radial_embed_size,
                           num_radial_layer=num_radial_layer,
                           radial_hidden_size=radial_hidden_size,
                           use_sc=use_sc,
                           irrep_normalization=irrep_normalization,
                           path_normalization=path_normalization)

        self.proj = EGCN(irreps_in=irreps_in,
                         irreps_out=irreps_in,
                         irreps_edge=irreps_edge,
                         radial_embed_size=radial_embed_size,
                         num_radial_layer=num_radial_layer,
                         radial_hidden_size=radial_hidden_size,
                         use_sc=use_sc,
                         irrep_normalization=irrep_normalization,
                         path_normalization=path_normalization)

        # Pre-compute, for each head, the (start, dim, l) layout of irreps
        # so we can form per-irrep invariant inner products.
        self._build_irrep_layout()

    def _build_irrep_layout(self):
        """Record slice + (2l+1) size of every irrep instance (per head)."""
        layout = []            # list of (start, length) within one head
        head_dim = self.irreps_in.dim // self.num_heads
        offset = 0
        for mul, ir in self.irreps_in:
            for _ in range(mul):
                layout.append((offset, ir.dim))
                offset += ir.dim
        self.head_dim = head_dim
        self.irrep_layout = layout   # used at runtime

    def _invariant_dot(self, q, k):
        """
        q, k : [E, num_heads, head_dim]  (edge-wise query/key)
        Returns invariant logits [E, num_heads] by summing per-irrep
        inner products (inner product within an irrep is rotation-invariant).
        """
        logits = 0.0
        for (start, length) in self.irrep_layout:
            qs = q[..., start:start + length]
            ks = k[..., start:start + length]
            logits = logits + (qs * ks).sum(dim=-1)   # invariant scalar per irrep
        return logits   # [E, num_heads]

    def forward(self, feature, edge_index, edge_feat, edge_embed, dim_size=None):
        """
        feature : [N, irreps_in.dim]
        edge_index : [2, E]  (src, dst)
        Attention is computed over the graph's edges (neighbourhood attention).
        """
        # print(f"Start of AttentionBlock")
        N = feature.shape[0]
        h = self.norm(feature)

        qkv = self.to_qkv(h, edge_index, edge_feat, edge_embed, dim_size=dim_size)
        q, k, v = qkv.reshape(N, 3, self.irreps_in.dim).unbind(dim=1)  # each [N, dim]

        # split into heads
        q = q.reshape(N, self.num_heads, self.head_dim)
        k = k.reshape(N, self.num_heads, self.head_dim)
        v = v.reshape(N, self.num_heads, self.head_dim)

        src, dst = edge_index[0], edge_index[1]

        # invariant attention logits on each edge (dst attends to src)
        scale = 1. / math.sqrt(self.head_dim // max(1, self.num_heads))
        logits = self._invariant_dot(q[dst], k[src]) * scale   # [E, num_heads]

        # softmax over incoming edges per destination node (equivariant: scalars)
        attn = scatter_softmax(logits, dst, dim=0)             # [E, num_heads]

        # scalar-weighted aggregation of value irreps -> equivariant
        msg = v[src] * attn.unsqueeze(-1)                      # [E, num_heads, head_dim]
        out = scatter_add(msg, dst, dim=0, dim_size=N)         # [N, num_heads, head_dim]
        out = out.reshape(N, self.irreps_in.dim)

        out = self.proj(out, edge_index, edge_feat, edge_embed, dim_size=dim_size)
        # print(f"End of AttentionBlock")
        return out + feature   # residual (shortcut)


class EUNet(nn.Module):
    def __init__(self,
                 irreps_in,
                 irreps_out,
                 irreps_model="32x0e+8x1o",
                 num_spherical=7,
                 num_radial=16,
                 radial_embed_size=64,
                 num_radial_layer=2,
                 radial_hidden_size=128,
                 irreps_mult=(1, 2, 2, 2),            # TODO: confirm original value
                 attention_levels=(2, 3),           # TODO: confirm original value
                 num_res_blocks=2,              # TODO: confirm original value
                 num_heads=1,
                 time_channels=None,
                 use_sc=True,
                 irrep_normalization='component',
                 path_normalization='element'):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_model = o3.Irreps(irreps_model)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_edge = (o3.Irreps.spherical_harmonics(num_spherical, p=1)).sort().irreps.simplify()
        self.num_res_blocks = num_res_blocks
        self.num_heads = num_heads

        time_channels = time_channels or 4 * self.irreps_model.count(o3.Irrep("0e"))
        self.time_channels = time_channels

        base_scalars = self.irreps_model.count(o3.Irrep("0e"))
        self.time_embed = nn.Sequential(
            nn.Linear(base_scalars, time_channels),
            nn.SiLU(),
            nn.Linear(time_channels, time_channels),
        )

        egcn_kwargs = dict(
            irreps_edge=(o3.Irreps.spherical_harmonics(num_spherical, p=1)).sort().irreps.simplify(),
            radial_embed_size=radial_embed_size,
            num_radial_layer=num_radial_layer,
            radial_hidden_size=radial_hidden_size,
            use_sc=use_sc,
            irrep_normalization=irrep_normalization,
            path_normalization=path_normalization,
        )
        res_kwargs = dict(time_channels=time_channels, **egcn_kwargs)
        attn_kwargs = dict(num_heads=num_heads, **egcn_kwargs)

        def scaled(irreps, m):
            return o3.Irreps([(mul * m, ir) for mul, ir in o3.Irreps(irreps)])

        # ---- stem ----
        self.input_conv = EGCN(irreps_in=self.irreps_in,
                               irreps_out=self.irreps_model,
                               **egcn_kwargs)

        # ---- down path ----
        self.down_blocks = nn.ModuleList()
        down_irreps = [self.irreps_model]
        cur = self.irreps_model
        for level, m in enumerate(irreps_mult):
            level_out = scaled(self.irreps_model, m)
            for _ in range(num_res_blocks):
                block = nn.ModuleList([ResidualBlock(irreps_in=cur,
                                                     irreps_out=level_out,
                                                     **res_kwargs)])
                cur = level_out
                if level in attention_levels:
                    block.append(AttentionBlock(irreps_in=cur, **attn_kwargs))
                self.down_blocks.append(block)
                down_irreps.append(cur)

        # ---- middle ----
        self.mid_res1 = ResidualBlock(irreps_in=cur, irreps_out=cur, **res_kwargs)
        self.mid_attn = AttentionBlock(irreps_in=cur, **attn_kwargs)
        self.mid_res2 = ResidualBlock(irreps_in=cur, irreps_out=cur, **res_kwargs)

        # ---- up path ----
        self.up_blocks = nn.ModuleList()
        for level, m in list(enumerate(irreps_mult))[::-1]:
            level_out = scaled(self.irreps_model, m)
            for i in range(num_res_blocks):
                skip = down_irreps.pop()
                in_irreps = (cur + skip).sort().irreps.simplify()
                block = nn.ModuleList([ResidualBlock(irreps_in=in_irreps,
                                                     irreps_out=level_out,
                                                     **res_kwargs)])
                cur = level_out
                if level in attention_levels:
                    block.append(AttentionBlock(irreps_in=cur, **attn_kwargs))
                self.up_blocks.append(block)

        # ---- output ----
        self.out_norm = NormActivation(irreps_in=cur)
        self.out_conv = EGCN(irreps_in=cur,
                             irreps_out=self.irreps_out,
                             **egcn_kwargs)

    def _run_block(self, block, feature, edge_index, edge_feat, edge_embed, emb, dim_size):
        """ResidualBlock (time-conditioned) then optional AttentionBlock."""
        feature = block[0](feature, edge_index, edge_feat, edge_embed, emb, dim_size=dim_size)
        if len(block) > 1:
            feature = block[1](feature, edge_index, edge_feat, edge_embed, dim_size=dim_size)
        return feature

    def forward(self, feature, edge_index, edge_feat, edge_embed,
                timesteps, node_batch=None, dim_size=None):
        base_scalars = self.irreps_model.count(o3.Irrep("0e"))
        t_sin = timestep_embedding(timesteps, base_scalars)
        emb = self.time_embed(t_sin)
        if node_batch is not None:
            emb = emb[node_batch]

        h = self.input_conv(feature, edge_index, edge_feat, edge_embed, dim_size=dim_size)
        hs = [h]

        for block in self.down_blocks:
            h = self._run_block(block, h, edge_index, edge_feat, edge_embed, emb, dim_size)
            hs.append(h)

        h = self.mid_res1(h, edge_index, edge_feat, edge_embed, emb, dim_size=dim_size)
        h = self.mid_attn(h, edge_index, edge_feat, edge_embed, dim_size=dim_size)
        h = self.mid_res2(h, edge_index, edge_feat, edge_embed, emb, dim_size=dim_size)

        for block in self.up_blocks:
            skip = hs.pop()
            h = torch.cat([h, skip], dim=-1)
            h = self._run_block(block, h, edge_index, edge_feat, edge_embed, emb, dim_size)

        h = self.out_norm(h)
        h = self.out_conv(h, edge_index, edge_feat, edge_embed, dim_size=dim_size)
        return h


# =============================================================
# Dataset
# =============================================================
def row_to_atoms(row):
    positions = np.stack(row.positions).astype(np.float32)
    cell = np.stack(row.cell).astype(np.float32)
    numbers = np.asarray(row.atomic_numbers, dtype=np.int32)

    cell_inverse = np.linalg.inv(cell)
    frac_coor = np.matmul(cell_inverse, positions.T)

    return Atoms(
        numbers=numbers,
        positions=frac_coor.T,
        cell=cell,
        pbc=True
    )


class AlexMP20(Dataset):
    def __init__(self, atoms_list: list, num_atoms: int = 89):
        super().__init__(root=None, transform=None)
        self.atoms_list = atoms_list
        self.num_atoms = num_atoms

    def len(self):
        return len(self.atoms_list)

    def get_one_hot_encode(self, atomic_numbers: np.ndarray):
        return F.one_hot(
            torch.tensor(atomic_numbers, dtype=torch.long),
            num_classes=self.num_atoms
        )

    def get(self, idx):
        selected_compound = self.atoms_list[idx]
        one_hot_encoded = self.get_one_hot_encode(selected_compound.numbers)
        data = Data(x=one_hot_encoded,
                    z=torch.tensor(selected_compound.numbers, dtype=torch.long),
                    pos=torch.tensor(selected_compound.get_positions(), dtype=torch.float32),
                    cell=torch.tensor(np.array(selected_compound.cell), dtype=torch.float32).unsqueeze(0)
                    )
        return data


# =============================================================
# Training loop
# =============================================================
import statistics
import os
import torch.distributed as dist


def train(atom_model: nn.Module, coord_model: nn.Module, lattice_model: nn.Module,
          atom_fwd: nn.Module, coord_fwd: nn.Module, lattice_fwd: nn.Module,
          loader: torch_geometric.loader.DataLoader, graph_builder: Callable,
          c_const: float, v_const: float, latt_feat_mean: torch.Tensor, latt_feat_std: torch.Tensor,
          sampler: torch.utils.data.distributed.DistributedSampler,
          epochs: int = 100, lr: float = 3e-5, atom_lr: float = 1e-5,
          device: torch.device = device,
          log_every: int = 50, warmup_steps: int = 5,
          skip_thresh: float = 50.0, latt_skip_thresh: float = 100.0,
          atom_clip: float = 0.5, other_clip: float = 1.0,
          logit_clamp: float = 30.0,
          ckpt_every: int = 1000, ckpt_dir: str = "."):

    # Atom branch gets its own (lower) LR — it's the most volatile.
    opt_atom = torch.optim.AdamW(atom_model.parameters(), lr=atom_lr)
    opt_coord = torch.optim.AdamW(coord_model.parameters(), lr=lr)
    opt_latt = torch.optim.AdamW(lattice_model.parameters(), lr=lr)

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * epochs
    scheduler_func = make_lr_lambda(warmup_steps, total_steps)

    atom_scheduler = LambdaLR(optimizer=opt_atom, lr_lambda=scheduler_func)
    coord_scheduler = LambdaLR(optimizer=opt_coord, lr_lambda=scheduler_func)
    latt_scheduler = LambdaLR(optimizer=opt_latt, lr_lambda=scheduler_func)

    mse = nn.SmoothL1Loss(beta=1.0)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = []  # (epoch, med_total, med_atom, med_coord, med_latt)

    def branch_loss_is_safe(loss_i, thresh):
        """Collective safety check. Under DDP, gradients are averaged across
        ranks during backward(), so a single bad rank poisons EVERY replica's
        gradient. We must therefore skip the step on ALL ranks together: if
        ANY rank's loss is non-finite or above threshold, all ranks skip."""
        safe = torch.isfinite(loss_i) and (loss_i.item() < thresh)
        flag = torch.tensor([1.0 if safe else 0.0], device=device)
        if dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)   # 0 if any rank unsafe
        return flag.item() > 0.5

    def guarded_step(is_safe, model_i, opt_i, clip):
        if is_safe:
            torch.nn.utils.clip_grad_norm_(model_i.parameters(), clip)
            opt_i.step()
            return True
        else:
            opt_i.zero_grad(set_to_none=True)
            return False

    lattice_ct = CartesianTensor("ij")
    rtp = lattice_ct.reduced_tensor_products().to(device)

    stop_training = False

    for epoch in tqdm(range(epochs)):
        if stop_training:
            break
        if sampler is not None:
            sampler.set_epoch(epoch)

        atom_model.train(); coord_model.train(); lattice_model.train()

        running = {"total": 0.0, "atom": 0.0, "coord": 0.0, "latt": 0.0}
        losses = {"total": [], "atom": [], "coord": [], "latt": []}
        skipped = {"atom": 0, "coord": 0, "latt": 0}
        n_batches = 0

        progress_bar = tqdm(
            enumerate(loader),
            desc=f"Training Epoch {epoch}",
            total=len(loader)
        )

        if is_main_process():
            print("c_const:", c_const, "v_const:", v_const)

        for step, batch in progress_bar:
            batch = batch.to(device)

            # ---- graph from the CLEAN structure (training uses true edges) ----
            edge_index, edge_feat, edge_embed = graph_builder(
                batch.pos, batch.cell, batch.batch, batch.z)

            # ---- sample a noised version of each modality ----
            atom_t, t_a, atom_target = atom_fwd(batch)
            coord_t, t_c, coord_target = coord_fwd(batch)
            latt_t, t_l, latt_target = lattice_fwd(batch, c_const, v_const)

            Bgraphs = int(batch.batch.max()) + 1
            tvec = lambda t: torch.full((Bgraphs,), int(t), device=device, dtype=torch.long)

            # ---- predictions ----
            atom_logits = atom_model(atom_t, edge_index, edge_feat, edge_embed,
                                     tvec(t_a), node_batch=batch.batch)
            # Bound the logits so softmax/exp can't overflow and the correct-class
            # logit can't grow without limit (the slow-divergence source).
            atom_logits = atom_logits.clamp(-logit_clamp, logit_clamp)

            coord_pred = coord_model(coord_t, edge_index, edge_feat, edge_embed,
                                     tvec(t_c), node_batch=batch.batch)

            latt_t_flat = lattice_ct.from_cartesian(latt_t, rtp=rtp)
            latt_t_flat = (latt_t_flat - latt_feat_mean) / latt_feat_std
            latt_t_flat = latt_t_flat.clamp(-5.0, 5.0)
            latt_feat_nodes = latt_t_flat[batch.batch]

            eps_pred_nodes = lattice_model(latt_feat_nodes, edge_index, edge_feat, edge_embed,
                                           tvec(t_l), node_batch=batch.batch)

            node_counts = torch.bincount(batch.batch, minlength=Bgraphs).clamp(min=1)
            eps_pred_sum = scatter_sum(eps_pred_nodes, batch.batch, dim=0, dim_size=Bgraphs)
            eps_pred_flat = eps_pred_sum / node_counts.unsqueeze(-1).to(eps_pred_sum.dtype)
            eps_pred = lattice_ct.to_cartesian(eps_pred_flat, rtp=rtp)

            # ---- losses (one per modality) ----
            loss_atom = ce(atom_logits, atom_target.float())   # classification
            loss_coord = mse(coord_pred, coord_target)
            loss_latt = mse(eps_pred, latt_target)             # epsilon-(smooth L1)

            loss = loss_atom + loss_coord + loss_latt

            # ---- single backward (DDP averages grads here), then guarded steps ----
            for o in (opt_atom, opt_coord, opt_latt):
                o.zero_grad(set_to_none=True)

            loss.backward()

            # Collective safety checks (all ranks agree before stepping).
            safe_atom = branch_loss_is_safe(loss_atom, skip_thresh)
            safe_coord = branch_loss_is_safe(loss_coord, skip_thresh)
            safe_latt = branch_loss_is_safe(loss_latt, latt_skip_thresh)

            stepped_atom = guarded_step(safe_atom, atom_model, opt_atom, atom_clip)
            stepped_coord = guarded_step(safe_coord, coord_model, opt_coord, other_clip)
            stepped_latt = guarded_step(safe_latt, lattice_model, opt_latt, other_clip)

            if not stepped_atom:
                skipped["atom"] += 1
            if not stepped_coord:
                skipped["coord"] += 1
            if not stepped_latt:
                skipped["latt"] += 1

            # Step all schedulers once per iteration so their LR schedules stay
            # aligned with total_steps regardless of per-branch skips.
            atom_scheduler.step()
            coord_scheduler.step()
            latt_scheduler.step()

            # ---- hard abort if the model has gone non-finite (unrecoverable) ----
            # Check collectively so all ranks break together.
            finite = torch.isfinite(loss).float().to(device).view(1)
            if dist.is_initialized():
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if finite.item() < 0.5:
                if is_main_process():
                    print(f"[epoch {epoch+1} step {step}] non-finite loss detected "
                          f"on at least one rank — stopping to avoid NaN propagation.")
                stop_training = True
                break

            # ---- accumulate (means + per-branch lists for medians) ----
            running["total"] += loss.item()
            running["atom"] += loss_atom.item()
            running["coord"] += loss_coord.item()
            running["latt"] += loss_latt.item()
            losses["total"].append(loss.item())
            losses["atom"].append(loss_atom.item())
            losses["coord"].append(loss_coord.item())
            losses["latt"].append(loss_latt.item())
            n_batches += 1

            # ---- periodic checkpoint (rank 0 only) ----
            if ckpt_every and is_main_process() and (step + 1) % ckpt_every == 0:
                def _sd(m):   # unwrap DDP if needed
                    return m.module.state_dict() if hasattr(m, "module") else m.state_dict()
                torch.save(
                    {"epoch": epoch + 1, "step": step + 1,
                     "atom": _sd(atom_model),
                     "coord": _sd(coord_model),
                     "lattice": _sd(lattice_model),
                     "c_const": c_const, "v_const": v_const,
                     "latt_feat_mean": latt_feat_mean.cpu(), "latt_feat_std": latt_feat_std.cpu()},
                    os.path.join(ckpt_dir, f"ckpt_e{epoch+1}_s{step+1}.pt"),
                )

            # ---- mid-epoch logging ----
            if log_every and (step + 1) % log_every == 0 and is_main_process():
                print(f"  epoch {epoch+1} | batch {step+1} | "
                      f"loss {loss.item():.4f} "
                      f"(atom {loss_atom.item():.4f}, "
                      f"coord {loss_coord.item():.4f}, "
                      f"latt {loss_latt.item():.4f})")

        # ---- per-epoch summaries: report MEDIANS (robust to outliers) ----
        med = {k: statistics.median(v) if v else 0.0 for k, v in losses.items()}
        avg = {k: v / max(n_batches, 1) for k, v in running.items()}

        history.append((epoch + 1, med["total"], med["atom"], med["coord"], med["latt"]))
        if is_main_process():
            print(f"[epoch {epoch+1:3d}/{epochs}] (median) "
                  f"total {med['total']:.4f} | "
                  f"atom {med['atom']:.4f} | "
                  f"coord {med['coord']:.4f} | "
                  f"latt {med['latt']:.4f}")
            print(f"              (mean)   "
                  f"total {avg['total']:.4f} | "
                  f"atom {avg['atom']:.4f} | "
                  f"coord {avg['coord']:.4f} | "
                  f"latt {avg['latt']:.4f}")
            print(f"              skipped  "
                  f"atom {skipped['atom']} | "
                  f"coord {skipped['coord']} | "
                  f"latt {skipped['latt']}  (of {n_batches} batches)")

    return history

# =============================================================
# NEW: Loss curve plotting
# =============================================================
def plot_loss_curves(history, save_path=None, logy=False, show=True):
    """
    Plot training loss curves from the `history` returned by `train()`.

    Parameters
    ----------
    history : list of tuples
        Each tuple is (epoch, total, atom, coord, latt).
    save_path : str, optional
        If given, the figure is saved to this path.
    logy : bool
        Use a logarithmic y-axis (useful when losses span orders of magnitude).
    show : bool
        Whether to call plt.show().
    """
    if not history:
        print("History is empty — nothing to plot.")
        return

    epochs = [h[0] for h in history]
    total = [h[1] for h in history]
    atom = [h[2] for h in history]
    coord = [h[3] for h in history]
    latt = [h[4] for h in history]

    # Each entry: (data, label, color, marker)
    series = [
        (total, "Total", "black", None),
        (atom, "Atom (CE)", "C0", "o"),
        (coord, "Coordinate (MSE)", "C1", "s"),
        (latt, "Lattice (MSE)", "C2", "^"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    for ax, (data, label, color, marker) in zip(axes, series):
        ax.plot(
            epochs,
            data,
            label=label,
            color=color,
            marker=marker,
            markersize=3,
            linewidth=2.5 if marker is None else 1.5,
        )

        if logy:
            ax.set_yscale("log")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Training Loss Curves")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved loss curve to {save_path}")

    if show:
        plt.show()


def main(rank: int, world: int):
    if world > 1:
        ddp_setup(rank=rank, world_size=world)

    device = torch.device(f"cuda:{rank}")          # rank-specific, NOT plain "cuda"

    print(f"[rank {rank}] device={device}, current_cuda={torch.cuda.current_device()}")

    # ---- data ----
    splits = {'train': 'train.parquet', 'val': 'val.parquet', 'test': 'test.parquet'}
    df = pd.read_parquet("hf://datasets/OMatG/Alex-MP-20/" + splits["train"])

    rng = range(int(len(df)))
    atoms_list = [row_to_atoms(df.iloc[i]) for i in (tqdm(rng) if rank == 0 else rng)]
    train_dataset = AlexMP20(atoms_list=atoms_list)

    sampler = DistributedSampler(train_dataset,
                                 num_replicas=world,     # from world_size
                                 rank=rank,              # this process's rank (0..world-1)
                                 shuffle=True,
                                 drop_last=True)
    dataloader = DataLoader(train_dataset,
                            batch_size=12,                # per-GPU batch size
                            sampler=sampler,             # do NOT also pass shuffle=True
                            num_workers=2,
                            pin_memory=True,
                            persistent_workers=True)

    # ---- forward (noising) processes ----
    atom_fwd = AtomDDIMForward().to(device)
    coor_fwd = CoordinateDDIMForward(device=device).to(device)
    latt_fwd = LatticeForwardDDIM().to(device)

    # ---- per-rank graph builder ----
    local_graph_builder = make_graph_builder(
        radial_fn=GaussianRBF(radial_embed_size=64, cutoff=7.0,
                              start=0.0, trainable=False).to(device)
    )

    # ---- models ----
    atom_model = EUNet(irreps_in="89x0e", num_spherical=2, num_radial=16,
                       irreps_out="89x0e", irreps_model="32x0e+8x1o",
                       radial_embed_size=64, num_radial_layer=2,
                       radial_hidden_size=64).to(device)
    coor_model = EUNet(irreps_in="1x1o", num_spherical=2, num_radial=16,
                       irreps_out="1x1o", irreps_model="32x0e+8x1o",
                       radial_embed_size=64, num_radial_layer=2,
                       radial_hidden_size=64).to(device)
    latt_model = EUNet(irreps_in="1x0e+1x1e+1x2e", num_spherical=2, num_radial=16,
                       irreps_out="1x0e+1x1e+1x2e", irreps_model="32x0e+8x1o",
                       radial_embed_size=64, num_radial_layer=2,
                       radial_hidden_size=64).to(device)
    if world > 1:
        atom_model = DDP(atom_model, device_ids=[rank], find_unused_parameters=True)
        coor_model = DDP(coor_model, device_ids=[rank], find_unused_parameters=True)
        latt_model = DDP(latt_model, device_ids=[rank], find_unused_parameters=True)

    # ---- lattice constants: compute over FULL dataset on rank 0, then broadcast ----
    if rank == 0:
        # separate plain loader (no DistributedSampler) so we see the whole dataset
        const_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
        c_const, v_const = compute_lattice_constants(const_loader)
        consts = torch.tensor([float(c_const), float(v_const)], device=device)

        
        lattice_ct = CartesianTensor("ij")
        rtp = lattice_ct.reduced_tensor_products().to(device)
        feats = []
        for b in const_loader:
            b = b.to(device)
            lt, _ , _ = latt_fwd(b, float(c_const), float(v_const))
            feats.append(lattice_ct.from_cartesian(lt, rtp=rtp).detach().cpu())
        feats = torch.cat(feats, 0)
        latt_feat_mean = feats.mean(0)
        latt_feat_std = feats.std(0).clamp(min=1e-6)

        consts = torch.cat([
            torch.tensor([float(c_const), float(v_const)]),
            latt_feat_mean, latt_feat_std
        ]).to(device)

    else:
        consts = torch.empty(20, device=device)
    if world > 1:
        dist.broadcast(consts, src=0)

    c_const, v_const = consts[0].item(), consts[1].item()
    latt_feat_mean = consts[2:11].to(device)
    latt_feat_std = consts[11:20].to(device)

    # ---- train ----
    history = train(atom_model=atom_model,
                    coord_model=coor_model,
                    lattice_model=latt_model,
                    atom_fwd=atom_fwd,
                    coord_fwd=coor_fwd,
                    lattice_fwd=latt_fwd,
                    loader=dataloader,
                    graph_builder=local_graph_builder,
                    latt_feat_mean=latt_feat_mean,
                    latt_feat_std=latt_feat_std,
                    c_const=c_const,
                    v_const=v_const,
                    epochs=EPOCHS,
                    device=device,
                    sampler=sampler,
                    ckpt_dir="/home/bandarua/scratch/projects/diffusion_mattergen/ckpt/cpkt_1")

    if is_main_process() and history:
        plot_loss_curves(history, save_path="loss_curves_3.png", show=False)
        torch.save(atom_model.state_dict(), "atom_model_3.pth")
        torch.save(coor_model.state_dict(), "coor_model_3.pth")
        torch.save(latt_model.state_dict(), "latt_model_3.pth")
    
    if world > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Number of GPUs: {world_size}")
    if world_size > 1:
        mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    #                            ^ trailing comma required — keep it a tuple
    else:
        main(rank=0, world=1)