# =============================================================
# generate.py  —  reverse DDIM sampling from trained models
# =============================================================
import numpy as np
import torch
import torch.nn.functional as F
from ase import Atoms
from e3nn.io import CartesianTensor
from tqdm import tqdm
from SmokeTest.DiffusionEGCN_smoketest import EUNet, GaussianRBF, make_graph_builder, scatter_sum
from SmokeTest.DiffusionEGCN_smoketest import AtomDDIMForward, CoordinateDDIMForward, LatticeForwardDDIM

# Assumes the following are importable from your training module:
#   EUNet, GaussianRBF, make_graph_builder, scatter_sum,
#   AtomDDIMForward, CoordinateDDIMForward, LatticeForwardDDIM
# from final_puzzle_pieces import (...)

device = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================
# The reverse sampler (user-supplied DDIM equations)
# =============================================================
@torch.no_grad()
def sample_structures(atom_model, coord_model, lattice_model,
                      atom_fwd, coord_fwd, lattice_fwd,
                      graph_builder,
                      c_const, v_const,
                      latt_feat_mean, latt_feat_std,
                      num_atoms_per_sample,
                      device=device,
                      num_steps=None,
                      rebuild_graph_every=1,
                      atom_sigma_fn=None,
                      logit_clamp=30.0):
    atom_model.eval(); coord_model.eval(); lattice_model.eval()

    T = atom_fwd.T
    num_steps = num_steps or T
    ts = torch.linspace(T - 1, 1, num_steps).round().long().tolist()

    if atom_sigma_fn is None:
        atom_sigma_fn = lambda t_int: 0.0

    B = len(num_atoms_per_sample)
    N = int(sum(num_atoms_per_sample))
    K = atom_fwd.num_classes

    node_batch = torch.repeat_interleave(
        torch.arange(B, device=device),
        torch.tensor(num_atoms_per_sample, device=device)
    )
    node_counts = torch.bincount(node_batch, minlength=B).clamp(min=1)
    n_per = torch.tensor(num_atoms_per_sample, device=device)
    tvec = lambda t: torch.full((B,), int(t), device=device, dtype=torch.long)

    lattice_ct = CartesianTensor("ij")
    rtp = lattice_ct.reduced_tensor_products().to(device)
    eye = torch.eye(3, device=device).unsqueeze(0)

    def alpha_at(sched, t_prev_int):
        return sched[0].item() if t_prev_int <= 0 else sched[t_prev_int].item()

    # ---- priors (t = T) ----
    x_atom = atom_fwd.mu.expand(N, K).clone()
    x_pos = torch.rand(N, 3, device=device)
    assert torch.isfinite(x_pos).all(), "non-finite fractional positions"
    x_pos = x_pos % 1.0

    assert c_const > 0 and v_const > 0, f"bad constants: c={c_const}, v={v_const}"
    mu_n = lattice_fwd.mu_n(c_const, n_per).view(-1, 1, 1)
    sig_n = lattice_fwd.sigma_n(v_const, n_per).view(-1, 1, 1)
    L = mu_n * eye + 0.1 * sig_n * torch.randn(B, 3, 3, device=device)
    assert torch.isfinite(L).all(), "initial lattice prior is non-finite"
    one_k = atom_fwd.mu.view(1, K)

    cached_graph = None
    for i, t in enumerate(tqdm(ts, desc="sampling")):
        t_prev = ts[i + 1] if i < len(ts) - 1 else 0

        if (cached_graph is None) or (i % rebuild_graph_every == 0):
            z_hard = x_atom.argmax(dim=-1)
            cached_graph = graph_builder(x_pos, L, node_batch, z_hard, fractional=True)
        edge_index, edge_feat, edge_embed = cached_graph

        # ---- ATOMS ----
        a_t = atom_fwd.alpha[t].item()
        a_prev = alpha_at(atom_fwd.alpha, t_prev)
        sig_t = float(atom_sigma_fn(int(t)))
        logits = atom_model(x_atom, edge_index, edge_feat, edge_embed,
                            tvec(t), node_batch=node_batch).clamp(-logit_clamp, logit_clamp)
        if i == 0:
            print("logit mean/std:", logits.mean().item(), logits.std().item())
            print("top-5 classes:", logits[0].topk(5).indices.tolist())
            print("softmax max prob:", F.softmax(logits, -1).max().item())
        x0_atom = F.softmax(logits, dim=-1)
        x_atom = (sig_t * x_atom
                  + (a_prev - sig_t * a_t) * x0_atom
                  + ((1 - a_prev) - (1 - a_t) * sig_t) * one_k)

        # ---- COORDINATES ----
        sig_c_t = coord_fwd.sigma(torch.tensor(float(t) / T, device=device))
        sig_c_prev = coord_fwd.sigma(torch.tensor(float(t_prev) / T, device=device))
        if coord_fwd.density_scaled:
            scale = node_counts[node_batch].float().pow(1 / 3)
            sig_c_t = (sig_c_t / scale).view(-1, 1)
            sig_c_prev = (sig_c_prev / scale).view(-1, 1)
        else:
            sig_c_t = sig_c_t.view(1, 1); sig_c_prev = sig_c_prev.view(1, 1)
        eps_coord = coord_model(x_pos, edge_index, edge_feat, edge_embed,
                                tvec(t), node_batch=node_batch)
        x_pos = (((x_pos + sig_c_t * eps_coord) % 1.0) - sig_c_prev * eps_coord) % 1.0

        # ---- LATTICE ----
        ab_t = lattice_fwd.alpha_bar[t].view(1, 1, 1)
        ab_prev = (lattice_fwd.alpha_bar[t_prev].view(1, 1, 1)
                   if t_prev > 0 else torch.ones(1, 1, 1, device=device))
        L_feat = lattice_ct.from_cartesian(L, rtp=rtp)
        L_feat = ((L_feat - latt_feat_mean) / latt_feat_std).clamp(-5.0, 5.0)
        eps_nodes = lattice_model(L_feat[node_batch], edge_index, edge_feat, edge_embed,
                                  tvec(t), node_batch=node_batch)
        eps_sum = scatter_sum(eps_nodes, node_batch, dim=0, dim_size=B)
        eps_flat = eps_sum / node_counts.unsqueeze(-1).to(eps_sum.dtype)
        eps_L = lattice_ct.to_cartesian(eps_flat, rtp=rtp)
        x0_core = (L
                   - (1 - torch.sqrt(ab_t)) * mu_n * eye
                   - torch.sqrt(1 - ab_t) * (sig_n ** 2) * eps_L) / torch.sqrt(ab_t).clamp(min=1e-6)
        L = (torch.sqrt(ab_prev) * x0_core
             + (1 - torch.sqrt(ab_prev)) * mu_n * eye
             + torch.sqrt(1 - ab_prev) * sig_n * eps_L)
        if i % 20 == 0:
            print(f"step {i}: det(L)=", torch.linalg.det(L).abs().mean().item())
        # after computing the new L in the loop
        det = torch.linalg.det(L).abs()                       # [B]
        target_vol = (c_const * n_per.float())                # [B]
        scale = (target_vol / det.clamp(min=1e-6)).pow(1/3).view(-1, 1, 1).clamp(0.5, 2.0)
        L = L * scale
        
        if not torch.isfinite(L).all():
            L = torch.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0)
            # optional: keep determinant positive / cell non-degenerate


    z_final = x_atom.argmax(dim=-1)
    results, offset = [], 0
    for b, n in enumerate(num_atoms_per_sample):
        sl = slice(offset, offset + n)
        results.append({
            "z": z_final[sl].detach().cpu(),
            "frac_pos": (x_pos[sl] % 1.0).detach().cpu(),
            "cell": L[b].detach().cpu(),
        })
        offset += n
    return results


# =============================================================
# Model construction (must match training config exactly)
# =============================================================
def build_models(device):
    atom_model = EUNet(irreps_in="89x0e", num_spherical=2, num_radial=16,
                       irreps_out="89x0e", irreps_model="32x0e+8x1o",
                       radial_embed_size=64, num_radial_layer=2,
                       radial_hidden_size=64).to(device)
    coord_model = EUNet(irreps_in="1x1o", num_spherical=2, num_radial=16,
                        irreps_out="1x1o", irreps_model="32x0e+8x1o",
                        radial_embed_size=64, num_radial_layer=2,
                        radial_hidden_size=64).to(device)
    lattice_model = EUNet(irreps_in="1x0e+1x1e+1x2e", num_spherical=2, num_radial=16,
                          irreps_out="1x0e+1x1e+1x2e", irreps_model="32x0e+8x1o",
                          radial_embed_size=64, num_radial_layer=2,
                          radial_hidden_size=64).to(device)
    return atom_model, coord_model, lattice_model


def load_checkpoint(ckpt_path, atom_model, coord_model, lattice_model, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    atom_model.load_state_dict(ckpt["atom"])
    coord_model.load_state_dict(ckpt["coord"])
    lattice_model.load_state_dict(ckpt["lattice"])
    print(f"Loaded checkpoint {ckpt_path} "
          f"(epoch {ckpt.get('epoch','?')}, step {ckpt.get('step','?')})")


# =============================================================
# Post-processing & validation
# =============================================================
def to_ase(sample):
    frac = sample["frac_pos"].numpy()
    cell = sample["cell"].numpy()
    cart = frac @ cell                       # fractional -> Cartesian
    return Atoms(numbers=sample["z"].numpy(), positions=cart, cell=cell, pbc=True)


def validate(sample, min_dist=0.5):
    """Basic physical sanity checks. Returns (ok, messages)."""
    msgs = []
    cell = sample["cell"].numpy()
    vol = abs(np.linalg.det(cell))
    n = len(sample["z"])
    ok = True

    if vol < 1e-3:
        ok = False; msgs.append(f"degenerate cell (vol={vol:.3e})")
    else:
        msgs.append(f"volume {vol:.1f} A^3 ({vol/n:.1f} A^3/atom)")

    z = sample["z"].numpy()
    if z.min() < 1 or z.max() > 88:
        ok = False; msgs.append(f"species out of range [{z.min()}, {z.max()}]")

    # minimum interatomic distance (with PBC via ASE)
    try:
        atoms = to_ase(sample)
        d = atoms.get_all_distances(mic=True)
        iu = np.triu_indices(n, k=1)
        if n > 1 and d[iu].min() < min_dist:
            ok = False; msgs.append(f"atoms too close ({d[iu].min():.2f} A)")
    except Exception as e:
        msgs.append(f"distance check failed: {e}")

    return ok, msgs


# =============================================================
# Driver
# =============================================================
def generate(ckpt_path,
             num_atoms_per_sample=(12, 8, 16, 10),
             num_steps=200,
             rebuild_graph_every=5,
             device=device):
    num_atoms_per_sample = list(num_atoms_per_sample)

    # 1) models + checkpoint
    atom_model, coord_model, lattice_model = build_models(device)
    load_checkpoint(ckpt_path, atom_model, coord_model, lattice_model, device)
    
    for name, m in [('atom', atom_model), ("coord", coord_model), ("lattice", lattice_model)]:
        bad = [n for n, p in m.named_parameters() if not torch.isfinite(p).all()]
        print(f"{name} non-finite params:", bad[:3] if bad else "(all finite)")

    # 2) forward processes (schedules must match training)
    atom_fwd = AtomDDIMForward().to(device)
    coord_fwd = CoordinateDDIMForward(device=device).to(device)
    lattice_fwd = LatticeForwardDDIM().to(device)
    
    print("alpha_bar[first t]:", lattice_fwd.alpha_bar[998].item())
    print("alpha_bar[last t]:", lattice_fwd.alpha_bar[1].item())
    print("1/sqrt(alpha_bar[1]):", 1.0 / (lattice_fwd.alpha_bar[1].item() ** 0.5))

    # 3) graph builder (same cutoff / RBF as training)
    graph_builder = make_graph_builder(
        radial_fn=GaussianRBF(radial_embed_size=64, cutoff=7.0,
                              start=0.0, trainable=False).to(device)
    )

    # 4) constants + lattice-feature stats.
    #    These MUST be the same values used in training. Easiest is to save
    #    them into the checkpoint; here we assume they are stored there, with
    #    a fallback to recomputing is not possible at inference, so load them.
    ckpt = torch.load(ckpt_path, map_location=device)
    c_const = ckpt["c_const"]; v_const = ckpt["v_const"]
    latt_feat_mean = ckpt["latt_feat_mean"].to(device)
    latt_feat_std = ckpt["latt_feat_std"].to(device)
    
    # ===== CHECK B: does the atom model predict real species on CLEAN data? =====
    # (run this as a separate snippet; needs a real structure from your dataset)
    from SmokeTest.DiffusionEGCN_smoketest import row_to_atoms, AlexMP20
    import pandas as pd
    from torch_geometric.loader import DataLoader

    df = pd.read_parquet("hf://datasets/OMatG/Alex-MP-20/train.parquet")
    atoms_list = [row_to_atoms(df.iloc[i]) for i in range(4)]
    ds = AlexMP20(atoms_list=atoms_list)
    batch = next(iter(DataLoader(ds, batch_size=4))).to(device)

    ei, ef, ee = graph_builder(batch.pos, batch.cell, batch.batch, batch.z)
    tvec = lambda t: torch.full((int(batch.batch.max())+1,), t, device=device, dtype=torch.long)
    # feed the CLEAN one-hot at a small timestep
    logits = atom_model(batch.x.float(), ei, ef, ee, tvec(1), node_batch=batch.batch)
    print("predicted:", logits.argmax(-1)[:20].tolist())
    print("true     :", batch.z[:20].tolist())
# ============================================================================


    # 5) sample
    samples = sample_structures(
        atom_model, coord_model, lattice_model,
        atom_fwd, coord_fwd, lattice_fwd,
        graph_builder,
        c_const, v_const,
        latt_feat_mean, latt_feat_std,
        num_atoms_per_sample=num_atoms_per_sample,
        num_steps=num_steps,
        rebuild_graph_every=rebuild_graph_every,
        device=device,
    )

    # 6) convert + validate
    atoms_out = []
    for i, s in enumerate(samples):
        ok, msgs = validate(s)
        status = "OK " if ok else "BAD"
        print(f"[{status}] sample {i}: n={len(s['z'])} | " + " | ".join(msgs))
        atoms_out.append(to_ase(s))

    return samples, atoms_out


if __name__ == "__main__":
    samples, atoms_out = generate(
        ckpt_path="/home/bandarua/scratch/projects/diffusion_mattergen/ckpt/cpkt_1/ckpt_e100_s3000.pt",       # your saved checkpoint
        num_atoms_per_sample=[12, 8, 16, 10, 20, 8, 24],
        num_steps=200,
        rebuild_graph_every=5,
    )

    # Inspect the first structure
    a = atoms_out[0]
    print("\nFirst generated structure:")
    print("  numbers:", a.get_atomic_numbers())
    print("  cell:\n", np.round(a.cell.array, 3))
    print("  frac positions:\n", np.round(a.get_scaled_positions(), 3))

    # To save as CIF locally (I cannot produce a file for you):
    from ase.io import write
    print(f"Length of the generated structures: {len(atoms_out)}")
    write("generated.cif", atoms_out)
    print("\nTo save: ase.io.write('generated.cif', atoms_out)")
