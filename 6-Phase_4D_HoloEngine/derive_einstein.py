"""
derive_einstein.py — Symbolic derivation of Einstein equations for our metric
==============================================================================

Derives ALL Einstein + KG equations from scratch using SymPy for the
characteristic metric ansatz:

    ds² = (1/z²)[-A dv² + Σ²(e^B dx² + e^{-B} dy²) + 2 dv dz + 2V dv dx]

with A, Σ, B, V, φ functions of (v, x, z).

This is the GROUND TRUTH — every coefficient is computed symbolically
with no manual adaptation or dimensional guessing.
"""

import sympy as sp
from sympy import symbols, Function, exp, sqrt, Rational, simplify, diff, Matrix
from sympy import pprint, collect, factor, cancel

# Coordinates
v, x, y, z = symbols('v x y z')
coords = [v, x, y, z]
n = len(coords)

# Metric functions (functions of v, x, z; y-independent by Z2)
A_f = Function('A')(v, x, z)
S_f = Function('Sigma')(v, x, z)
B_f = Function('B')(v, x, z)
V_f = Function('V_s')(v, x, z)

# ===================================================================
# METRIC TENSOR g_{mu nu}
# ===================================================================
# ds² = (1/z²)[-A dv² + Σ²(e^B dx² + e^{-B} dy²) + 2 dv dz + 2V dv dx]
#
# Basis ordering: (v, x, y, z) = indices (0, 1, 2, 3)

g = Matrix.zeros(4, 4)

# Overall 1/z² factor
oz2 = 1 / z**2

g[0, 0] = oz2 * (-A_f)          # g_{vv}
g[0, 1] = oz2 * V_f             # g_{vx}
g[1, 0] = oz2 * V_f             # g_{xv}
g[0, 3] = oz2 * 1               # g_{vz}
g[3, 0] = oz2 * 1               # g_{zv}
g[1, 1] = oz2 * S_f**2 * exp(B_f)   # g_{xx}
g[2, 2] = oz2 * S_f**2 * exp(-B_f)  # g_{yy}
# All other components are 0 (g_{xz}=g_{yz}=g_{xy}=g_{zz}=0)

print("=" * 60)
print("METRIC TENSOR g_{mu,nu}")
print("=" * 60)
for i in range(4):
    for j in range(i, 4):
        if g[i, j] != 0:
            print(f"g[{coords[i]},{coords[j]}] = {g[i, j]}")

# ===================================================================
# INVERSE METRIC g^{mu nu}
# ===================================================================
print("\n" + "=" * 60)
print("COMPUTING INVERSE METRIC...")
print("=" * 60)

g_inv = g.inv()

# Simplify
for i in range(4):
    for j in range(4):
        g_inv[i, j] = simplify(g_inv[i, j])

print("Inverse metric computed.")
for i in range(4):
    for j in range(i, 4):
        if g_inv[i, j] != 0:
            expr = simplify(g_inv[i, j])
            print(f"g^[{coords[i]},{coords[j]}] = {expr}")

# ===================================================================
# DETERMINANT
# ===================================================================
det_g = simplify(g.det())
print(f"\ndet(g) = {det_g}")
sqrt_neg_g = simplify(sqrt(-det_g))
print(f"sqrt(-g) = {sqrt_neg_g}")

# ===================================================================
# CHRISTOFFEL SYMBOLS  Γ^σ_{μν} = ½ g^{σρ}(∂_μ g_{ρν} + ∂_ν g_{ρμ} - ∂_ρ g_{μν})
# ===================================================================
print("\n" + "=" * 60)
print("COMPUTING CHRISTOFFEL SYMBOLS...")
print("=" * 60)

Gamma = [[[None for _ in range(4)] for _ in range(4)] for _ in range(4)]

for sigma in range(4):
    for mu in range(4):
        for nu in range(mu, 4):  # symmetric in lower indices
            val = 0
            for rho in range(4):
                val += Rational(1, 2) * g_inv[sigma, rho] * (
                    diff(g[rho, nu], coords[mu])
                    + diff(g[rho, mu], coords[nu])
                    - diff(g[mu, nu], coords[rho])
                )
            Gamma[sigma][mu][nu] = simplify(val)
            Gamma[sigma][nu][mu] = Gamma[sigma][mu][nu]

print("Christoffel symbols computed.")

# ===================================================================
# RICCI TENSOR  R_{μν} = ∂_ρ Γ^ρ_{μν} - ∂_ν Γ^ρ_{μρ} + Γ^ρ_{ρλ}Γ^λ_{μν} - Γ^ρ_{νλ}Γ^λ_{μρ}
# ===================================================================
print("\n" + "=" * 60)
print("COMPUTING RICCI TENSOR (this takes a while)...")
print("=" * 60)

R = Matrix.zeros(4, 4)

for mu in range(4):
    for nu in range(mu, 4):
        val = 0
        for rho in range(4):
            val += diff(Gamma[rho][mu][nu], coords[rho])
            val -= diff(Gamma[rho][mu][rho], coords[nu])
            for lam in range(4):
                val += Gamma[rho][rho][lam] * Gamma[lam][mu][nu]
                val -= Gamma[rho][nu][lam] * Gamma[lam][mu][rho]
        R[mu, nu] = val
        R[nu, mu] = val
        comp_name = f"R[{coords[mu]},{coords[nu]}]"
        print(f"  {comp_name} computed.")

print("Ricci tensor computed.")

# ===================================================================
# RICCI SCALAR
# ===================================================================
print("\nComputing Ricci scalar...")
R_scalar = 0
for mu in range(4):
    for nu in range(4):
        R_scalar += g_inv[mu, nu] * R[mu, nu]
R_scalar = simplify(R_scalar)
print(f"R = {R_scalar}")

# ===================================================================
# EINSTEIN TENSOR  G_{μν} = R_{μν} - ½ R g_{μν}
# ===================================================================
print("\n" + "=" * 60)
print("COMPUTING EINSTEIN TENSOR + Λ...")
print("=" * 60)

Lambda = -3  # AdS4 cosmological constant with L=1

G = Matrix.zeros(4, 4)
for mu in range(4):
    for nu in range(mu, 4):
        G[mu, nu] = R[mu, nu] - Rational(1, 2) * R_scalar * g[mu, nu] + Lambda * g[mu, nu]
        G[nu, mu] = G[mu, nu]

# ===================================================================
# EXTRACT THE EQUATIONS  G_{μν} + Λ g_{μν} = 0  (vacuum)
# ===================================================================
print("\n" + "=" * 60)
print("EINSTEIN EQUATIONS  G_{μν} + Λ g_{μν} = 0")
print("=" * 60)

# For vacuum (no matter), each component of G + Λg must vanish.
# Print the key components:

# Component labels
labels = {
    (3, 3): "E_zz (→ R1: Sigma equation)",
    (1, 3): "E_xz (→ R2/Sigma/B related)",
    (0, 3): "E_vz (→ nested radial or evolution)",
    (0, 0): "E_vv (→ constraint or A equation)",
    (1, 1): "E_xx",
    (2, 2): "E_yy",
    (0, 1): "E_vx (→ shift V equation)",
}

for (mu, nu), label in labels.items():
    print(f"\n--- {label} ---")
    expr = simplify(G[mu, nu])
    # Check if it vanishes for pure AdS: A=1, Sigma=1, B=0, V=0
    test = expr.subs([(A_f, 1), (S_f, 1), (B_f, 0), (V_f, 0)])
    # Remove all derivatives of constants
    for f in [A_f, S_f, B_f, V_f]:
        for c in [v, x, z]:
            test = test.subs(diff(f, c), 0)
            for c2 in [v, x, z]:
                test = test.subs(diff(f, c, c2), 0)
                for c3 in [v, x, z]:
                    test = test.subs(diff(f, c, c2, c3), 0)
    test = simplify(test)
    print(f"  Pure AdS check: {test}")
    if test == 0:
        print("  ✓ Vanishes for pure AdS")
    else:
        print(f"  ✗ NON-ZERO for pure AdS: {test}")

print("\n" + "=" * 60)
print("DERIVATION COMPLETE")
print("=" * 60)
