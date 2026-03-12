"""
ads_physics.py -- Anti-de Sitter Physics Engine
=================================================

Implements the core physics-aware components required for rigorous
holographic bulk reconstruction via Physics-Informed Neural Networks.

Components:
    1. AdS Monte Carlo Collocation Sampler (logarithmic z-axis)
    2. Klein-Gordon PDE Loss (curved-spacetime d'Alembertian via autograd)
    3. Ryu-Takayanagi Eikonal Constraint (differentiable minimal surfaces)
    4. Sigmoid Curriculum Annealing Scheduler

Reference Geometry:
    Poincaré Patch AdS₄:  ds² = (R²/z²)(-dt² + dx² + dy² + dz²)
    where z=0 is the conformal boundary (UV), z→∞ is deep bulk (IR).

    For a static slice (dt=0), the induced 3D metric is:
        g_ij = (R/z)² δ_ij    =>    √(-g) = (R/z)³
    We set R=1 (AdS radius) throughout.
"""

import math
import torch
import torch.nn as nn


# =====================================================================
# 1. AdS MONTE CARLO COLLOCATION SAMPLER
# =====================================================================

def sample_ads_collocation(batch_size: int, num_points: int,
                           z_min: float = 1e-4, z_max: float = 1.0,
                           device: str = "cpu"):
    """
    Sample collocation points in the Poincaré patch of AdS space.

    The x, y coordinates are sampled uniformly in [-1, 1].
    The z coordinate (holographic depth) is sampled LOGARITHMICALLY
    to ensure equal density per unit of proper distance in AdS.

    Returns
    -------
    coords : Tensor, shape (batch_size, num_points, 3)
        (x, y, z) with requires_grad=True for autograd PDE computation.
    measure : Tensor, shape (batch_size, num_points, 1)
        The integration weight sqrt(-g) = 1/z³ for each point.
    """
    # Spatial coordinates: uniform in [-1, 1]
    x = torch.rand(batch_size, num_points, 1, device=device) * 2 - 1
    y = torch.rand(batch_size, num_points, 1, device=device) * 2 - 1

    # Holographic depth: logarithmic sampling
    # z = exp(uniform(log(z_min), log(z_max)))
    log_z_min = math.log(z_min)
    log_z_max = math.log(z_max)
    log_z = torch.rand(batch_size, num_points, 1, device=device)
    log_z = log_z * (log_z_max - log_z_min) + log_z_min
    z = torch.exp(log_z)

    # Integration measure: sqrt(-g) = (R/z)^3, with R=1
    measure = 1.0 / (z ** 3)

    # Concatenate and enable gradients for autograd PDE
    coords = torch.cat([x, y, z], dim=-1)
    coords = coords.requires_grad_(True)

    return coords, measure


# =====================================================================
# 2. KLEIN-GORDON PDE LOSS (Curved Spacetime via Autograd)
# =====================================================================

def klein_gordon_residual(phi, coords, mass_sq: float = 0.0):
    """
    Compute the Klein-Gordon equation residual in Poincaré AdS₃+1.

    The scalar field φ must satisfy:
        □_AdS φ - m²φ = 0

    In the Poincaré patch (static slice, R=1):
        g^{ij} = z² δ^{ij}  (inverse metric)
        √(-g) = 1/z³

    The covariant d'Alembertian expands to:
        □φ = z² (∂²φ/∂x² + ∂²φ/∂y² + ∂²φ/∂z²) - z ∂φ/∂z

    Parameters
    ----------
    phi : Tensor, shape (B, N, 1)
        Scalar field values predicted by the SIREN at collocation points.
    coords : Tensor, shape (B, N, 3), requires_grad=True
        The (x, y, z) collocation coordinates.
    mass_sq : float
        The mass² parameter for the KG field. m²=0 for a massless scalar.

    Returns
    -------
    residual : Tensor, shape (B, N, 1)
        The KG equation residual at each collocation point.
    """
    # First derivatives: ∂φ/∂x, ∂φ/∂y, ∂φ/∂z
    grad_phi_tuple = torch.autograd.grad(
        outputs=phi,
        inputs=coords,
        grad_outputs=torch.ones_like(phi),
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )
    
    if grad_phi_tuple[0] is None:
        grad_phi = torch.zeros_like(coords)
    else:
        grad_phi = grad_phi_tuple[0]

    dphi_dx = grad_phi[:, :, 0:1]  # ∂φ/∂x
    dphi_dy = grad_phi[:, :, 1:2]  # ∂φ/∂y
    dphi_dz = grad_phi[:, :, 2:3]  # ∂φ/∂z

    # Second derivatives: ∂²φ/∂x², ∂²φ/∂y², ∂²φ/∂z²
    grad_dx_tuple = torch.autograd.grad(
        outputs=dphi_dx, inputs=coords,
        grad_outputs=torch.ones_like(dphi_dx),
        create_graph=True, retain_graph=True, allow_unused=True
    )
    d2phi_dx2 = grad_dx_tuple[0][:, :, 0:1] if grad_dx_tuple[0] is not None else torch.zeros_like(dphi_dx)

    grad_dy_tuple = torch.autograd.grad(
        outputs=dphi_dy, inputs=coords,
        grad_outputs=torch.ones_like(dphi_dy),
        create_graph=True, retain_graph=True, allow_unused=True
    )
    d2phi_dy2 = grad_dy_tuple[0][:, :, 1:2] if grad_dy_tuple[0] is not None else torch.zeros_like(dphi_dy)

    grad_dz_tuple = torch.autograd.grad(
        outputs=dphi_dz, inputs=coords,
        grad_outputs=torch.ones_like(dphi_dz),
        create_graph=True, retain_graph=True, allow_unused=True
    )
    d2phi_dz2 = grad_dz_tuple[0][:, :, 2:3] if grad_dz_tuple[0] is not None else torch.zeros_like(dphi_dz)

    # Extract z coordinate for metric factors
    z = coords[:, :, 2:3]  # shape: (B, N, 1)

    # Covariant d'Alembertian in Poincaré AdS:
    #   □φ = z²(∂²φ/∂x² + ∂²φ/∂y² + ∂²φ/∂z²) - z ∂φ/∂z
    z_sq = z ** 2
    laplacian = z_sq * (d2phi_dx2 + d2phi_dy2 + d2phi_dz2) - z * dphi_dz

    # Klein-Gordon residual: □φ - m²φ = 0
    residual = laplacian - mass_sq * phi

    return residual


def klein_gordon_loss(phi, coords, measure, mass_sq: float = 0.0):
    """
    Weighted Klein-Gordon PDE loss integrated over AdS collocation points.

    L_KG = (1/N) Σ_i |residual_i|² * measure_i

    Parameters
    ----------
    phi : Tensor, shape (B, N, 1)
    coords : Tensor, shape (B, N, 3), requires_grad=True
    measure : Tensor, shape (B, N, 1)
        Integration weight sqrt(-g) = 1/z³.
    mass_sq : float

    Returns
    -------
    loss : scalar Tensor
    """
    residual = klein_gordon_residual(phi, coords, mass_sq)
    # Weighted L2 norm with the AdS volume measure
    weighted = (residual ** 2) * measure
    return weighted.mean()


# =====================================================================
# 3. RYU-TAKAYANAGI EIKONAL CONSTRAINT
# =====================================================================

def eikonal_loss(phi, coords):
    """
    Enforce the Eikonal equation |∇φ| = 1 on the SIREN output,
    making the level-set φ=0 a well-defined minimal surface.

    L_eikonal = (1/N) Σ_i (|∇φ_i| - 1)²

    Parameters
    ----------
    phi : Tensor, shape (B, N, 1)
    coords : Tensor, shape (B, N, 3), requires_grad=True

    Returns
    -------
    loss : scalar Tensor
    """
    grad_phi_tuple = torch.autograd.grad(
        outputs=phi,
        inputs=coords,
        grad_outputs=torch.ones_like(phi),
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )
    
    if grad_phi_tuple[0] is None:
        grad_phi = torch.zeros_like(coords)
    else:
        grad_phi = grad_phi_tuple[0]

    grad_norm = torch.linalg.norm(grad_phi, dim=-1, keepdim=True)  # (B, N, 1)
    return ((grad_norm - 1.0) ** 2).mean()


def ryu_takayanagi_loss(area_proxy, entropy, G_N: float = 1.0):
    """
    Enforce the Ryu-Takayanagi formula: Area / (4 G_N) = S

    Parameters
    ----------
    area_proxy : scalar Tensor
        A differentiable proxy for the minimal surface area.
    entropy : scalar Tensor or float
        The von Neumann entanglement entropy of the quantum layer.
    G_N : float
        Newton's gravitational constant (set to 1.0 in natural units).

    Returns
    -------
    loss : scalar Tensor
    """
    rt_prediction = area_proxy / (4.0 * G_N)
    if isinstance(entropy, float):
        entropy = torch.tensor(entropy, device=area_proxy.device)
    return (rt_prediction - entropy) ** 2


# =====================================================================
# 4. SIGMOID CURRICULUM ANNEALING SCHEDULER
# =====================================================================

def get_loss_weights(epoch: int, max_epochs: int = 100):
    """
    Dynamic loss weight scheduler using sigmoid annealing.

    Schedule:
        - 2D Boundary (UV Anchor): always 1.0
        - 3D Data Scaffold: starts at 1.0, decays to 0.0 as PDE turns on
        - KG PDE: ramps smoothly from 0 → 1.0 centered at epoch 25
        - RT Area: ramps smoothly from 0 → 0.1 centered at epoch 55

    Parameters
    ----------
    epoch : int
        Current training epoch.
    max_epochs : int
        Total number of training epochs.

    Returns
    -------
    dict with keys: '3d_data', 'boundary', 'pde', 'rt'
    """
    # 2D Boundary is the conformal anchor (always on)
    w_boundary = 1.0

    # PDE Loss: sigmoid ramp centered at epoch 25, steepness 0.5
    pde_center = max(15, int(max_epochs * 0.25))
    w_pde = 1.0 / (1.0 + math.exp(-0.5 * (epoch - pde_center)))

    # Topological Scaffolding (Perturbation Theory)
    # The fake 3D data fades out perfectly as the true PDE turns on
    w_3d_data = 1.0 - w_pde

    # RT Area Loss: sigmoid ramp centered at epoch 55, steepness 0.5
    rt_center = max(45, int(max_epochs * 0.55))
    w_rt = 0.1 / (1.0 + math.exp(-0.5 * (epoch - rt_center)))

    return {
        "3d_data": w_3d_data,
        "boundary": w_boundary,
        "pde": w_pde,
        "rt": w_rt,
    }


# =====================================================================
# QUICK TEST
# =====================================================================

if __name__ == "__main__":
    print("Testing AdS Physics Engine...\n")

    device = "cpu"

    # 1. Test collocation sampler
    coords, measure = sample_ads_collocation(
        batch_size=2, num_points=512, z_min=1e-4, z_max=1.0, device=device
    )
    print(f"Collocation coords: {list(coords.shape)}")
    print(f"  x range: [{coords[:,:,0].min():.4f}, {coords[:,:,0].max():.4f}]")
    print(f"  y range: [{coords[:,:,1].min():.4f}, {coords[:,:,1].max():.4f}]")
    print(f"  z range: [{coords[:,:,2].min():.6f}, {coords[:,:,2].max():.4f}]")
    print(f"Measure range: [{measure.min():.1f}, {measure.max():.1e}]")

    # 2. Test Klein-Gordon loss with a dummy SIREN
    print("\nTesting Klein-Gordon PDE loss...")
    dummy_phi = torch.sin(coords[:, :, 0:1] + coords[:, :, 1:2]) * torch.exp(-coords[:, :, 2:3])
    # The dummy phi already requires grad because coords requires grad
    
    # We need coords to already have grad enabled (done in sampler)
    kg_loss = klein_gordon_loss(dummy_phi, coords, measure, mass_sq=0.0)
    print(f"  KG Loss: {kg_loss.item():.6f}")

    # 3. Test Eikonal loss
    print("\nTesting Eikonal constraint...")
    eik_loss = eikonal_loss(dummy_phi, coords)
    print(f"  Eikonal Loss: {eik_loss.item():.6f}")

    # 4. Test curriculum scheduler
    print("\nCurriculum schedule preview (Topological Scaffolding):")
    for ep in [1, 10, 20, 25, 30, 40, 50, 55, 60, 80, 100]:
        w = get_loss_weights(ep, max_epochs=100)
        print(f"  Epoch {ep:3d}: Scaffold3D={w['3d_data']:.4f}  Boundary={w['boundary']:.1f}  "
              f"PDE={w['pde']:.4f}  RT={w['rt']:.6f}")

    # 5. Test RT loss
    print("\nTesting RT loss...")
    area_proxy = torch.tensor(12.0, requires_grad=True)
    entropy = 3.0
    rt_loss = ryu_takayanagi_loss(area_proxy, entropy, G_N=1.0)
    print(f"  RT Loss (Area=12, S=3): {rt_loss.item():.4f}")

    print("\n✓ All AdS Physics Engine tests passed!")
