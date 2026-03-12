"""
horizon.py — Apparent Horizon Finder & Thermodynamics
======================================================

Finds the apparent horizon (AH) of the BBH system by locating where
the lapse function A(v, x, z) = 0 in the bulk.

For our characteristic metric:
    ds² = (1/z²)[-A dv² + Σ²(e^B dx² + e^{-B} dy²) + 2 dv dz + 2V dv dx]

The apparent horizon is the outermost marginally outer trapped surface
(MOTS), which in this gauge corresponds to A(v, x, z_AH) = 0.

Once the horizon is located, we compute:
    - Horizon area:  Area = ∫ Σ²|_{z=z_AH} dx dy
    - BH entropy:    S = Area / (4 G_N)
    - BH temperature: T = κ/(2π) where κ is the surface gravity

Merger detection: two separate A=0 surfaces → one connected surface
"""

import torch
from ads_config import BBHConfig


# ====================================================================== #
#  HORIZON FINDER  (MOTS: Θ⁺ = d₊Σ/Σ = 0, with A=0 fallback)            #
# ====================================================================== #
def find_apparent_horizon(metric_siren, encoder, boundary_input,
                          reconstructor, cfg: type = BBHConfig,
                          v_value: float = None,
                          n_x: int = 64, n_u: int = 256):
    """
    Find the apparent horizon via MOTS condition: Θ⁺ = d₊Σ/Σ = 0.

    The outgoing null expansion is:
        Θ⁺ = d₊Σ / Σ  where  d₊ = ∂_v + (A/2)∂_u

    We evaluate on a dense (x, u) grid at fixed v, compute Σ_v and Σ_u
    via finite differences, and find where Θ⁺ crosses zero.
    Falls back to A=0 scan if MOTS detection fails (early training).

    Returns
    -------
    horizon_data : dict with keys:
        found, z_AH, u_AH, x_vals, Sigma_AH, B_AH, area, entropy, n_components
    """
    device = cfg.DEVICE
    if v_value is None:
        v_value = cfg.V_RANGE[1]
    dv_fd = 0.005

    x_vals = torch.linspace(cfg.X_RANGE[0], cfg.X_RANGE[1], n_x, device=device)
    u_vals = torch.linspace(cfg.U_MIN, cfg.U_MAX, n_u, device=device)
    du = (cfg.U_MAX - cfg.U_MIN) / max(n_u - 1, 1)

    xx, uu = torch.meshgrid(x_vals, u_vals, indexing='ij')

    # Evaluate metric at v and v-dv for finite-difference Σ_v
    grids = {}
    for v_eval in [v_value, v_value - dv_fd]:
        vv = torch.full_like(xx, v_eval)
        coords = torch.stack([vv.flatten(), xx.flatten(), uu.flatten()], dim=-1)
        with torch.no_grad():
            z_lat = encoder(boundary_input).detach().expand(coords.shape[0], -1)
            met = reconstructor.reconstruct(metric_siren(coords, z_lat), coords)
        grids[v_eval] = {k: met[k].reshape(n_x, n_u) for k in ['Sigma', 'A', 'B']}

    S_now = grids[v_value]['Sigma']
    S_prev = grids[v_value - dv_fd]['Sigma']
    A_grid = grids[v_value]['A']
    B_grid = grids[v_value]['B']

    # Σ_v via backward finite difference
    S_v = (S_now - S_prev) / dv_fd

    # Σ_u via central finite difference along u
    S_u = torch.zeros_like(S_now)
    S_u[:, 1:-1] = (S_now[:, 2:] - S_now[:, :-2]) / (2 * du)
    S_u[:, 0] = (S_now[:, 1] - S_now[:, 0]) / du
    S_u[:, -1] = (S_now[:, -1] - S_now[:, -2]) / du

    # MOTS: Θ⁺ = (Σ_v + A/2 · Σ_u) / Σ = 0
    Theta_plus = (S_v + 0.5 * A_grid * S_u) / S_now.clamp(min=1e-8)

    z_AH = torch.full((n_x,), float('nan'), device=device)
    u_AH = torch.full((n_x,), float('nan'), device=device)
    Sigma_AH = torch.full((n_x,), float('nan'), device=device)
    B_AH = torch.full((n_x,), float('nan'), device=device)

    # Scan for outermost Θ⁺ = 0 crossing
    for i in range(n_x):
        for j in range(1, n_u):
            if Theta_plus[i, j-1] > 0 and Theta_plus[i, j] <= 0:
                frac = Theta_plus[i, j-1] / (Theta_plus[i, j-1] - Theta_plus[i, j] + 1e-12)
                u_h = u_vals[j-1] + frac * (u_vals[j] - u_vals[j-1])
                u_AH[i], z_AH[i] = u_h, torch.exp(u_h)
                Sigma_AH[i] = (1-frac) * S_now[i, j-1] + frac * S_now[i, j]
                B_AH[i] = (1-frac) * B_grid[i, j-1] + frac * B_grid[i, j]
                break

    found_mask = ~torch.isnan(z_AH)

    # Fallback: if MOTS finds nothing, try A=0
    if not found_mask.any():
        for i in range(n_x):
            neg = (A_grid[i] <= 0)
            if neg.any():
                j = neg.float().argmax().item()
                if j > 0:
                    f = A_grid[i, j-1] / (A_grid[i, j-1] - A_grid[i, j] + 1e-12)
                    u_h = u_vals[j-1] + f * (u_vals[j] - u_vals[j-1])
                    Sigma_AH[i] = (1-f) * S_now[i, j-1] + f * S_now[i, j]
                    B_AH[i] = (1-f) * B_grid[i, j-1] + f * B_grid[i, j]
                else:
                    u_h = u_vals[j]; Sigma_AH[i] = S_now[i, j]; B_AH[i] = B_grid[i, j]
                u_AH[i], z_AH[i] = u_h, torch.exp(u_h)
        found_mask = ~torch.isnan(z_AH)

    found = found_mask.any().item()
    if not found:
        return dict(found=False, z_AH=z_AH, u_AH=u_AH, x_vals=x_vals,
                    Sigma_AH=Sigma_AH, B_AH=B_AH,
                    area=torch.tensor(0.0, device=device),
                    entropy=torch.tensor(0.0, device=device), n_components=0)

    dx = (cfg.X_RANGE[1] - cfg.X_RANGE[0]) / max(n_x - 1, 1)
    L_y = cfg.X_RANGE[1] - cfg.X_RANGE[0]
    area = L_y * (Sigma_AH[found_mask] ** 2).sum() * dx
    entropy = area / (4.0 * cfg.NEWTON_G)
    n_components = _count_components(found_mask)

    return dict(found=True, z_AH=z_AH, u_AH=u_AH, x_vals=x_vals,
                Sigma_AH=Sigma_AH, B_AH=B_AH, area=area,
                entropy=entropy, n_components=n_components)


def _count_components(mask: torch.Tensor) -> int:
    """Count disconnected True regions in a 1D boolean mask."""
    if not mask.any():
        return 0
    edges = torch.diff(mask.float())
    # Rising edges = start of a component
    return max(1, int((edges > 0).sum().item()) + (1 if mask[0] else 0))


# ====================================================================== #
#  SURFACE GRAVITY (TEMPERATURE)                                           #
# ====================================================================== #
def compute_surface_gravity(metric_siren, encoder, boundary_input,
                            reconstructor, horizon_data: dict,
                            cfg: type = BBHConfig) -> torch.Tensor:
    """
    Surface gravity κ from the gradient of A at the horizon:

        κ = (1/2) |dA/dz|_{z=z_AH}|

    BH temperature: T = κ / (2π)

    Returns: T_BH scalar
    """
    device = cfg.DEVICE
    if not horizon_data['found']:
        return torch.tensor(0.0, device=device)

    found_mask = ~torch.isnan(horizon_data['u_AH'])
    if not found_mask.any():
        return torch.tensor(0.0, device=device)

    # Sample points right at the horizon
    v_val = cfg.V_RANGE[1]
    x_h = horizon_data['x_vals'][found_mask]
    u_h = horizon_data['u_AH'][found_mask]
    v_h = torch.full_like(x_h, v_val)

    coords = torch.stack([v_h, x_h, u_h], dim=-1).requires_grad_(True)

    with torch.no_grad():
        z_latent = encoder(boundary_input).detach()

    with torch.enable_grad():
        z_lat = z_latent.expand(coords.shape[0], -1)
        raw = metric_siren(coords, z_lat)
        metric = reconstructor.reconstruct(raw, coords)

        # dA/du at horizon
        A = metric['A']
        g = torch.autograd.grad(A.sum(), coords, create_graph=False)[0]
    dA_du = g[:, 2]  # d/du

    # dA/dz = e^{-u} dA/du (chain rule: du/dz = 1/z = e^{-u})
    dA_dz = torch.exp(-u_h) * dA_du

    kappa = 0.5 * dA_dz.abs().mean()
    T_BH = kappa / (2.0 * 3.14159265358979)

    return T_BH


# ====================================================================== #
#  HORIZON REGULARITY LOSS  (for training)                                 #
# ====================================================================== #
def horizon_regularity_loss(metric: dict, coords: torch.Tensor,
                            cfg: type = BBHConfig) -> torch.Tensor:
    """
    At the apparent horizon (A → 0), the metric must remain regular.
    This means Σ must be finite and positive, and B must be finite.

    We add a soft penalty: if A is small, penalize Σ < ε or |B| → ∞

    This prevents the network from creating coordinate singularities
    at the horizon.

    Returns: scalar loss
    """
    A = metric['A']
    S = metric['Sigma']

    # Weight: stronger where A is near zero (near horizon)
    horizon_weight = torch.exp(-A.abs() * 10.0)

    # Penalize Σ < 0.1 near the horizon
    sigma_reg = torch.relu(0.1 - S).pow(2)

    return (horizon_weight * sigma_reg).mean()


# ====================================================================== #
#  COVARIANT HRT EXTREMAL SURFACE                                          #
# ====================================================================== #
def hrt_entanglement_entropy(metric_siren, encoder, boundary_input,
                             reconstructor, x_boundary: float = 0.0,
                             v_value: float = None,
                             cfg: type = BBHConfig,
                             n_pts: int = 200,
                             n_shoot_z: int = 16,
                             n_shoot_v: int = 8,
                             half_width: float = 0.5,
                             horizon_data: dict = None) -> torch.Tensor:
    """
    Covariant HRT surface in full (v, x, z) spacetime.

    The RT surface is a 2-surface parameterized by (x, y) with both
    v(x) and z(x) varying. The induced area functional is:

      A = L_y ∫ √(h_xx · h_yy) dx

    where:
      h_xx = (1/z²)[Σ²e^B - A(v')² + 2Vv' + 2v'z']
      h_yy = Σ²e^{-B}/z²

    On a constant-v slice, h_xx = Σ²e^B/z² (independent of z(x)),
    so the v-variation is physically essential (z-direction is null).

    Shoots over (z_*, δv) parameterization and finds minimum area.
    Compares connected vs disconnected phases (RT phase transition).
    """
    import math
    device = cfg.DEVICE
    if v_value is None:
        v_value = cfg.V_RANGE[1]

    x_left = x_boundary - half_width
    x_right = x_boundary + half_width
    x_pts = torch.linspace(x_left, x_right, n_pts, device=device)
    dx_val = (x_right - x_left) / max(n_pts - 1, 1)
    L_y = cfg.X_RANGE[1] - cfg.X_RANGE[0]

    z_stars = torch.linspace(cfg.Z_MIN * 10, cfg.Z_MAX * 0.8, n_shoot_z, device=device)
    dv_offsets = torch.linspace(-0.15, 0.0, n_shoot_v, device=device)

    t = (x_pts - x_left) / (x_right - x_left + 1e-12)
    profile = torch.sin(math.pi * t)

    best_area = torch.tensor(float('inf'), device=device)

    for z_star in z_stars:
        for dv_off in dv_offsets:
            z_surf = cfg.Z_MIN + (z_star.item() - cfg.Z_MIN) * profile
            v_surf = v_value + dv_off.item() * profile
            u_surf = torch.log(z_surf.clamp(min=cfg.Z_MIN))

            coords = torch.stack([v_surf, x_pts, u_surf], dim=-1)
            with torch.no_grad():
                z_lat = encoder(boundary_input).detach().expand(n_pts, -1)
                met = reconstructor.reconstruct(metric_siren(coords, z_lat), coords)

            S = met['Sigma'].squeeze(-1)
            A_f = met['A'].squeeze(-1)
            B_f = met['B'].squeeze(-1)
            V_f = met['V_shift'].squeeze(-1)
            z_f = met['z'].squeeze(-1)

            dz_dx = torch.zeros_like(x_pts)
            dv_dx = torch.zeros_like(x_pts)
            if n_pts > 2:
                dz_dx[1:-1] = (z_surf[2:] - z_surf[:-2]) / (2 * dx_val)
                dz_dx[0] = (z_surf[1] - z_surf[0]) / dx_val
                dz_dx[-1] = (z_surf[-1] - z_surf[-2]) / dx_val
                dv_dx[1:-1] = (v_surf[2:] - v_surf[:-2]) / (2 * dx_val)
                dv_dx[0] = (v_surf[1] - v_surf[0]) / dx_val
                dv_dx[-1] = (v_surf[-1] - v_surf[-2]) / dx_val

            h_xx = (S**2 * torch.exp(B_f)
                    - A_f * dv_dx**2
                    + 2 * V_f * dv_dx
                    + 2 * dv_dx * dz_dx) / (z_f**2 + 1e-12)
            h_yy = S**2 * torch.exp(-B_f) / (z_f**2 + 1e-12)

            det_h = h_xx * h_yy
            valid = det_h > 0
            if not valid.any():
                continue

            integrand = torch.where(valid,
                                    torch.sqrt(det_h.clamp(min=1e-12)),
                                    torch.zeros_like(det_h))
            area = L_y * torch.trapezoid(integrand, dx=dx_val)
            if area > 0 and area < best_area:
                best_area = area

    # Disconnected phase: use horizon entropy if available
    if horizon_data is not None and horizon_data.get('found', False):
        disc_entropy = horizon_data['entropy']
    else:
        disc_area = torch.tensor(0.0, device=device)
        for x_bh in [cfg.BH_POSITION_1, cfg.BH_POSITION_2]:
            u_d = torch.linspace(cfg.U_MIN, cfg.U_MAX, n_pts, device=device)
            z_d = torch.exp(u_d)
            dv_d = -0.02 * torch.sin(math.pi * (u_d - cfg.U_MIN)/(cfg.U_MAX - cfg.U_MIN))
            v_d = v_value + dv_d
            coords_d = torch.stack([v_d, torch.full_like(u_d, x_bh), u_d], dim=-1)
            with torch.no_grad():
                z_lat_d = encoder(boundary_input).detach().expand(n_pts, -1)
                met_d = reconstructor.reconstruct(metric_siren(coords_d, z_lat_d), coords_d)
            S_d = met_d['Sigma'].squeeze(-1)
            disc_area += torch.trapezoid(
                S_d**2 / (z_d**2 + 1e-12),
                dx=(cfg.U_MAX - cfg.U_MIN) / max(n_pts - 1, 1))
        disc_entropy = disc_area / (4.0 * cfg.NEWTON_G)

    S_conn = best_area / (4.0 * cfg.NEWTON_G) if best_area < float('inf') else torch.tensor(float('inf'), device=device)
    return torch.min(S_conn, disc_entropy) if isinstance(disc_entropy, torch.Tensor) else S_conn
