"""
observables.py — Physical Observable Extraction from the Learned Metric
========================================================================

Extracts gauge-invariant physical quantities from the trained BBH metric:

1. Boundary stress-energy tensor <T_μν> from asymptotic falloff
2. Gravitational waveform h+ from the anisotropy B at the boundary
3. Quasinormal mode frequencies from late-time ringdown
4. Mass and angular momentum of the final BH
5. Energy conservation check (Ward identity)
"""

import torch
import numpy as np
from ads_config import BBHConfig


# ====================================================================== #
#  BOUNDARY STRESS-ENERGY TENSOR  (holographic dictionary)                 #
# ====================================================================== #
def extract_boundary_stress_tensor(metric_siren, encoder, boundary_input,
                                   reconstructor,
                                   cfg: type = BBHConfig,
                                   n_v: int = 100, n_x: int = 64):
    """
    Extract the boundary CFT stress-energy tensor <T_μν>(v, x) from
    the z³ coefficient of the metric functions.

    In the Fefferman-Graham / EF expansion:
        A(v,x,z) = 1 + z² + a₄(v,x) z³ + ...
        B(v,x,z) =          b₄(v,x) z³ + ...

    The holographic dictionary gives:
        <T_vv>(v,x)         ∝  a₄(v,x)     = δA_net(v,x,u_bnd)
        <T_xx - T_yy>(v,x)  ∝  b₄(v,x)     = δB_net(v,x,u_bnd)

    The raw SIREN deviations AT the boundary (z→0) are exactly the
    z³ coefficients, since we factored out z³ in MetricReconstructor.

    Parameters
    ----------
    metric_siren, encoder, boundary_input, reconstructor : models
    cfg : BBHConfig

    Returns
    -------
    dict with keys:
        v_grid   : (n_v,)
        x_grid   : (n_x,)
        T_vv     : (n_v, n_x) — energy density
        T_xx_yy  : (n_v, n_x) — pressure anisotropy (GW signal)
        T_vx     : (n_v, n_x) — momentum flux
        total_E  : (n_v,)     — integrated energy at each v
    """
    device = cfg.DEVICE

    v_grid = torch.linspace(cfg.V_RANGE[0], cfg.V_RANGE[1], n_v, device=device)
    x_grid = torch.linspace(cfg.X_RANGE[0], cfg.X_RANGE[1], n_x, device=device)

    T_vv = torch.zeros(n_v, n_x, device=device)
    T_xx_yy = torch.zeros(n_v, n_x, device=device)
    T_vx = torch.zeros(n_v, n_x, device=device)

    u_bnd = cfg.U_MIN  # Evaluate at the AdS boundary

    with torch.no_grad():
        z_latent = encoder(boundary_input).detach()

        for iv in range(n_v):
            v_val = v_grid[iv]
            coords = torch.stack([
                torch.full((n_x,), v_val, device=device),
                x_grid,
                torch.full((n_x,), u_bnd, device=device),
            ], dim=-1)

            z_lat = z_latent.expand(n_x, -1)
            raw = metric_siren(coords, z_lat)
            # raw[:, i] are the SIREN deviation outputs
            # At the boundary, these ARE the z³ coefficients

            # Holographic normalization factor:
            # In AdS4, <T_μν> = (3/(16πG)) · (coefficient at z³)
            norm = 3.0 / (16.0 * 3.14159265 * cfg.NEWTON_G)

            T_vv[iv]    = norm * raw[:, cfg.IDX_A].squeeze()
            T_xx_yy[iv] = norm * raw[:, cfg.IDX_B].squeeze()
            T_vx[iv]    = norm * raw[:, cfg.IDX_V].squeeze()

    # Integrated energy at each time slice
    dx = (cfg.X_RANGE[1] - cfg.X_RANGE[0]) / max(n_x - 1, 1)
    L_y = cfg.X_RANGE[1] - cfg.X_RANGE[0]
    total_E = L_y * T_vv.sum(dim=1) * dx

    return dict(
        v_grid=v_grid, x_grid=x_grid,
        T_vv=T_vv, T_xx_yy=T_xx_yy, T_vx=T_vx,
        total_E=total_E,
    )


# ====================================================================== #
#  GRAVITATIONAL WAVEFORM  h+                                              #
# ====================================================================== #
def extract_gravitational_waveform(metric_siren, encoder, boundary_input,
                                   reconstructor,
                                   cfg: type = BBHConfig,
                                   n_v: int = 500, x_obs: float = 0.0):
    """
    Extract the gravitational waveform from the boundary anisotropy.

    The anisotropy B encodes the tensor perturbation mode. At the
    boundary (z → 0), the z³ coefficient of B gives the transverse-
    traceless part of the boundary stress tensor:

        h+(v) ∝ <T_xx - T_yy>(v, x_obs) ∝ δB_net(v, x_obs, u_bnd)

    This is the gravitational wave signal as observed by the boundary CFT.

    Parameters
    ----------
    x_obs : observation point in x-coordinate (default: center)

    Returns
    -------
    dict with keys:
        v_times   : (n_v,)  — advanced time array
        h_plus    : (n_v,)  — waveform amplitude
        frequency : (n_v-1,) — instantaneous frequency (d(phase)/dv)
    """
    device = cfg.DEVICE
    v_times = torch.linspace(cfg.V_RANGE[0], cfg.V_RANGE[1], n_v, device=device)
    u_bnd = cfg.U_MIN

    coords = torch.stack([
        v_times,
        torch.full_like(v_times, x_obs),
        torch.full_like(v_times, u_bnd),
    ], dim=-1)

    with torch.no_grad():
        z_latent = encoder(boundary_input).detach()
        z_lat = z_latent.expand(n_v, -1)
        raw = metric_siren(coords, z_lat)

    norm = 3.0 / (16.0 * 3.14159265 * cfg.NEWTON_G)
    h_plus = norm * raw[:, cfg.IDX_B]

    # Instantaneous frequency via finite differences
    dv = (cfg.V_RANGE[1] - cfg.V_RANGE[0]) / max(n_v - 1, 1)
    dh = torch.diff(h_plus.squeeze())
    # Analytic signal for frequency estimation
    frequency = dh / (dv + 1e-12)

    return dict(
        v_times=v_times,
        h_plus=h_plus.squeeze(),
        frequency=frequency,
    )


# ====================================================================== #
#  QUASINORMAL MODE EXTRACTION                                             #
# ====================================================================== #
def extract_qnm_frequencies(waveform_data: dict,
                            cfg: type = BBHConfig,
                            n_modes: int = 3) -> dict:
    """
    Extract quasinormal mode (QNM) frequencies from the late-time
    gravitational waveform via Prony's method.

    QNMs are damped sinusoidal oscillations:
        h+(v) ≈ Σ_n  C_n exp(-α_n v) cos(ω_n v + φ_n)

    where ω_n are the QNM frequencies and α_n are the damping rates.

    Uses the Prony method: fit the signal to a sum of complex exponentials.

    Parameters
    ----------
    waveform_data : output of extract_gravitational_waveform
    n_modes : number of QNM modes to extract

    Returns
    -------
    dict with keys:
        omega_real : (n_modes,) — oscillation frequencies
        omega_imag : (n_modes,) — damping rates
        amplitudes : (n_modes,) — mode amplitudes
    """
    h = waveform_data['h_plus'].cpu().numpy()
    v = waveform_data['v_times'].cpu().numpy()

    # Use only the late-time portion (post-merger ringdown)
    n_total = len(h)
    start = int(0.6 * n_total)  # Last 40% of the signal
    h_ring = h[start:]
    v_ring = v[start:]
    dv = v_ring[1] - v_ring[0] if len(v_ring) > 1 else 1.0

    N = len(h_ring)
    if N < 2 * n_modes + 1:
        return dict(omega_real=np.zeros(n_modes),
                    omega_imag=np.zeros(n_modes),
                    amplitudes=np.zeros(n_modes))

    # Prony's method: Toeplitz matrix approach
    M = n_modes
    # Build data matrix
    H = np.zeros((N - M, M))
    for i in range(N - M):
        H[i] = h_ring[i:i + M]
    b = h_ring[M:N]

    # Least squares for characteristic polynomial coefficients
    try:
        coeffs, _, _, _ = np.linalg.lstsq(H, b, rcond=None)
        # Root finding
        char_poly = np.concatenate([np.array([1.0]), -coeffs[::-1]])
        roots = np.roots(char_poly)

        # Convert to frequencies: z_k = exp(s_k Δv), so s_k = ln(z_k)/Δv
        s_k = np.log(roots + 1e-30) / dv
        omega_real = np.abs(np.imag(s_k))
        omega_imag = -np.real(s_k)  # Damping rate (positive = decaying)
        amplitudes = np.abs(roots)

        # Sort by amplitude (most dominant first)
        order = np.argsort(-amplitudes)[:n_modes]
        return dict(
            omega_real=omega_real[order],
            omega_imag=omega_imag[order],
            amplitudes=amplitudes[order],
        )
    except Exception:
        return dict(omega_real=np.zeros(n_modes),
                    omega_imag=np.zeros(n_modes),
                    amplitudes=np.zeros(n_modes))


# ====================================================================== #
#  ENERGY CONSERVATION CHECK  (Ward identity)                              #
# ====================================================================== #
def check_energy_conservation(stress_tensor: dict,
                              cfg: type = BBHConfig) -> dict:
    """
    Check the Ward identity (energy conservation) of the boundary CFT:

        ∂_v <T^vv> + ∂_x <T^vx> = 0

    This should be approximately satisfied if the Einstein equations
    are well-solved.

    Returns
    -------
    dict with:
        violation : (n_v-1, n_x-1)  — Ward identity residual
        max_violation : scalar
        mean_violation : scalar
    """
    T_vv = stress_tensor['T_vv']     # (n_v, n_x)
    T_vx = stress_tensor['T_vx']     # (n_v, n_x)
    v_grid = stress_tensor['v_grid']
    x_grid = stress_tensor['x_grid']

    dv = (v_grid[-1] - v_grid[0]) / max(len(v_grid) - 1, 1)
    dx = (x_grid[-1] - x_grid[0]) / max(len(x_grid) - 1, 1)

    # Finite difference
    dT_vv_dv = (T_vv[1:, :] - T_vv[:-1, :]) / (dv + 1e-12)
    dT_vx_dx = (T_vx[:, 1:] - T_vx[:, :-1]) / (dx + 1e-12)

    # Trim to common grid
    n_v = min(dT_vv_dv.shape[0], dT_vx_dx.shape[0])
    n_x = min(dT_vv_dv.shape[1], dT_vx_dx.shape[1])

    violation = dT_vv_dv[:n_v, :n_x] + dT_vx_dx[:n_v, :n_x]

    return dict(
        violation=violation,
        max_violation=violation.abs().max().item(),
        mean_violation=violation.abs().mean().item(),
    )


# ====================================================================== #
#  FINAL BLACK HOLE MASS                                                   #
# ====================================================================== #
def compute_final_mass(stress_tensor: dict,
                       cfg: type = BBHConfig) -> torch.Tensor:
    """
    Compute the ADM-like mass of the final black hole from the total
    boundary energy at late times:

        M = ∫ <T_vv>(v_final, x) dx dy

    Returns: M_final scalar
    """
    # Use the last time slice
    T_vv_final = stress_tensor['T_vv'][-1]  # (n_x,)
    x_grid = stress_tensor['x_grid']
    dx = (x_grid[-1] - x_grid[0]) / max(len(x_grid) - 1, 1)
    L_y = cfg.X_RANGE[1] - cfg.X_RANGE[0]

    M_final = L_y * T_vv_final.sum() * dx
    return M_final
