"""Port of DiffCSP CSPNet (jiaor17/DiffCSP @ 7121d15, diffcsp/pl_modules/cspnet.py) to modern PyTorch.

Differences from the original:
  * no torch_scatter / torch_geometric / einops dependency (pure-torch scatter-mean, block-diag edges);
  * only the 'fc' edge style is implemented (the MP-20 CSP checkpoint uses edge_style='fc');
    'knn' raises NotImplementedError;
  * pred_type / pred_scalar / smooth heads kept for state-dict compatibility.
Module names and parameter shapes are identical to the original so the Lightning
checkpoint's `decoder.*` state dict loads with strict=True.
"""
import math

import torch
import torch.nn as nn

MAX_ATOMIC_NUM = 100


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Equivalent of torch_scatter.scatter(src, index, dim=0, reduce='mean', dim_size=dim_size)."""
    out = src.new_zeros((dim_size,) + src.shape[1:])
    out.index_add_(0, index, src)
    cnt = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    cnt.index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    cnt = cnt.clamp(min=1)
    return out / cnt.view((-1,) + (1,) * (src.dim() - 1))


class SinusoidsEmbedding(nn.Module):
    def __init__(self, n_frequencies=10, n_space=3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        # plain attribute in the original (not a buffer) -> not in the state dict
        self.frequencies = 2 * math.pi * torch.arange(self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :].to(x.device)
        emb = emb.reshape(-1, self.n_frequencies * self.n_space)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class CSPLayer(nn.Module):
    """Message passing layer for cspnet."""

    def __init__(self, hidden_dim=128, act_fn=nn.SiLU(), dis_emb=None, ln=False, ip=True):
        super().__init__()
        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.ip = True  # hard-coded True in the original
        if dis_emb is not None:
            self.dis_dim = dis_emb.dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 9 + self.dis_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.ln = ln
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def edge_model(self, node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff=None):
        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        if frac_diff is None:
            xi, xj = frac_coords[edge_index[0]], frac_coords[edge_index[1]]
            frac_diff = (xj - xi) % 1.0
        if self.dis_emb is not None:
            frac_diff = self.dis_emb(frac_diff)
        if self.ip:
            lattice_ips = lattices @ lattices.transpose(-1, -2)
        else:
            lattice_ips = lattices
        lattice_ips_flatten = lattice_ips.view(-1, 9)
        lattice_ips_flatten_edges = lattice_ips_flatten[edge2graph]
        edges_input = torch.cat([hi, hj, lattice_ips_flatten_edges, frac_diff], dim=1)
        return self.edge_mlp(edges_input)

    def node_model(self, node_features, edge_features, edge_index):
        agg = scatter_mean(edge_features, edge_index[0], dim_size=node_features.shape[0])
        agg = torch.cat([node_features, agg], dim=1)
        return self.node_mlp(agg)

    def forward(self, node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff=None):
        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)
        edge_features = self.edge_model(node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff)
        node_output = self.node_model(node_features, edge_features, edge_index)
        return node_input + node_output


class CSPNet(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        latent_dim=256,
        num_layers=4,
        max_atoms=100,
        act_fn="silu",
        dis_emb="sin",
        num_freqs=10,
        edge_style="fc",
        cutoff=6.0,
        max_neighbors=20,
        ln=False,
        ip=True,
        smooth=False,
        pred_type=False,
        pred_scalar=False,
    ):
        super().__init__()
        self.ip = ip
        self.smooth = smooth
        if self.smooth:
            self.node_embedding = nn.Linear(max_atoms, hidden_dim)
        else:
            self.node_embedding = nn.Embedding(max_atoms, hidden_dim)
        self.atom_latent_emb = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        if act_fn == "silu":
            self.act_fn = nn.SiLU()
        if dis_emb == "sin":
            self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs)
        elif dis_emb == "none":
            self.dis_emb = None
        for i in range(num_layers):
            self.add_module("csp_layer_%d" % i, CSPLayer(hidden_dim, self.act_fn, self.dis_emb, ln=ln, ip=ip))
        self.num_layers = num_layers
        self.coord_out = nn.Linear(hidden_dim, 3, bias=False)
        self.lattice_out = nn.Linear(hidden_dim, 9, bias=False)
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.pred_type = pred_type
        self.ln = ln
        self.edge_style = edge_style
        if self.ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)
        if self.pred_type:
            self.type_out = nn.Linear(hidden_dim, MAX_ATOMIC_NUM)
        self.pred_scalar = pred_scalar
        if self.pred_scalar:
            self.scalar_out = nn.Linear(hidden_dim, 1)

    def gen_edges(self, num_atoms, frac_coords, lattices, node2graph):
        if self.edge_style == "fc":
            # original: block_diag(ones(n,n)...) -> torch_geometric dense_to_sparse (row-major nonzero order)
            lis = [torch.ones(int(n), int(n), device=num_atoms.device) for n in num_atoms]
            fc_graph = torch.block_diag(*lis)
            fc_edges = fc_graph.nonzero().t().contiguous()
            return fc_edges, (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]) % 1.0
        raise NotImplementedError("only edge_style='fc' is ported (MP-20 CSP checkpoint uses 'fc')")

    def forward(self, t, atom_types, frac_coords, lattices, num_atoms, node2graph):
        edges, frac_diff = self.gen_edges(num_atoms, frac_coords, lattices, node2graph)
        edge2graph = node2graph[edges[0]]
        if self.smooth:
            node_features = self.node_embedding(atom_types)
        else:
            node_features = self.node_embedding(atom_types - 1)

        t_per_atom = t.repeat_interleave(num_atoms, dim=0)
        node_features = torch.cat([node_features, t_per_atom], dim=1)
        node_features = self.atom_latent_emb(node_features)

        for i in range(self.num_layers):
            node_features = self._modules["csp_layer_%d" % i](
                node_features, frac_coords, lattices, edges, edge2graph, frac_diff=frac_diff
            )

        if self.ln:
            node_features = self.final_layer_norm(node_features)

        coord_out = self.coord_out(node_features)
        graph_features = scatter_mean(node_features, node2graph, dim_size=int(num_atoms.shape[0]))

        if self.pred_scalar:
            return self.scalar_out(graph_features)

        lattice_out = self.lattice_out(graph_features)
        lattice_out = lattice_out.view(-1, 3, 3)
        if self.ip:
            lattice_out = torch.einsum("bij,bjk->bik", lattice_out, lattices)
        if self.pred_type:
            type_out = self.type_out(node_features)
            return lattice_out, coord_out, type_out
        return lattice_out, coord_out
