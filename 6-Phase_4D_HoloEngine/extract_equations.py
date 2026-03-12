"""
extract_equations.py — Extract the CORRECT Einstein equations in z-coords
=========================================================================

Reads the SymPy-derived G_{μν} + Λg_{μν} = 0 and prints the equations
in a form suitable for converting to PyTorch code.

Key: works in z-coordinates (not u) first, then convert to u = ln(z).
"""

import sympy as sp
from sympy import (symbols, Function, exp, sqrt, Rational, simplify,
                   diff, Matrix, collect, factor, cancel, Symbol, Wild,
                   Derivative, trigsimp, nsimplify, apart, together)

# Coordinates
v, x, y, z = symbols('v x y z')
coords = [v, x, y, z]

# Metric functions
A_f = Function('A')(v, x, z)
S_f = Function('Sigma')(v, x, z)
B_f = Function('B')(v, x, z)
V_f = Function('V_s')(v, x, z)

# Build metric (same as derive_einstein.py)
g = Matrix.zeros(4, 4)
oz2 = 1 / z**2
g[0, 0] = oz2 * (-A_f)
g[0, 1] = oz2 * V_f
g[1, 0] = oz2 * V_f
g[0, 3] = oz2 * 1
g[3, 0] = oz2 * 1
g[1, 1] = oz2 * S_f**2 * exp(B_f)
g[2, 2] = oz2 * S_f**2 * exp(-B_f)

print("Computing inverse metric...")
g_inv = g.inv()
for i in range(4):
    for j in range(4):
        g_inv[i, j] = simplify(g_inv[i, j])

# Christoffel
print("Computing Christoffel symbols...")
Gamma = [[[None]*4 for _ in range(4)] for _ in range(4)]
for sigma in range(4):
    for mu in range(4):
        for nu in range(mu, 4):
            val = sum(
                Rational(1,2) * g_inv[sigma, rho] * (
                    diff(g[rho, nu], coords[mu])
                    + diff(g[rho, mu], coords[nu])
                    - diff(g[mu, nu], coords[rho])
                )
                for rho in range(4)
            )
            Gamma[sigma][mu][nu] = simplify(val)
            Gamma[sigma][nu][mu] = Gamma[sigma][mu][nu]

# Ricci tensor
print("Computing Ricci tensor...")
R = Matrix.zeros(4, 4)
for mu in range(4):
    for nu in range(mu, 4):
        val = sum(
            diff(Gamma[rho][mu][nu], coords[rho])
            - diff(Gamma[rho][mu][rho], coords[nu])
            + sum(Gamma[rho][rho][lam]*Gamma[lam][mu][nu]
                  - Gamma[rho][nu][lam]*Gamma[lam][mu][rho]
                  for lam in range(4))
            for rho in range(4)
        )
        R[mu, nu] = val
        R[nu, mu] = val
        print(f"  R_{mu}{nu} done")

# Ricci scalar
print("Computing Ricci scalar...")
R_scalar = sum(g_inv[mu, nu] * R[mu, nu] for mu in range(4) for nu in range(4))
R_scalar = simplify(R_scalar)

# Einstein + Lambda
Lambda = -3
G = Matrix.zeros(4, 4)
for mu in range(4):
    for nu in range(mu, 4):
        G[mu, nu] = R[mu, nu] - Rational(1,2) * R_scalar * g[mu, nu] + Lambda * g[mu, nu]
        G[nu, mu] = G[mu, nu]

# =================================================================
# EXTRACT EACH EQUATION
# =================================================================

# For the characteristic formulation, the key equations come from:
# (3,3) = (z,z) component: Nested radial R1 (Sigma)
# We need z² * (G_{zz} + Λ g_{zz}), but g_{zz} = 0, so just G_{zz} = 0

print("\n" + "=" * 70)
print("(z,z) COMPONENT => R1: Sigma equation")
print("=" * 70)
Ezz = simplify(G[3, 3])
# Multiply by z^2 * Sigma^2 to clean up
Ezz_clean = simplify(Ezz * z**2 * S_f**2)
print("z² Σ² G_{zz} =", Ezz_clean)

# Set B=0, V=0, phi=0 to get the pure Sigma equation
Ezz_simple = Ezz_clean.subs([(B_f, 0), (V_f, 0)])
for c in coords:
    Ezz_simple = Ezz_simple.subs(diff(B_f, c), 0).subs(diff(V_f, c), 0)
    for c2 in coords:
        Ezz_simple = Ezz_simple.subs(diff(B_f, c, c2), 0).subs(diff(V_f, c, c2), 0)
Ezz_simple = simplify(Ezz_simple)
print("Simplified (B=V=0):", Ezz_simple)

# Verify: A=1, Sigma=1
test = Ezz_simple.subs([(A_f, 1), (S_f, 1)])
for c in coords:
    test = test.subs(diff(A_f, c), 0).subs(diff(S_f, c), 0)
    for c2 in coords:
        test = test.subs(diff(A_f, c, c2), 0).subs(diff(S_f, c, c2), 0)
print("Pure AdS check:", simplify(test))

print("\n" + "=" * 70)
print("(x,z) COMPONENT => related to B equation")
print("=" * 70)
Exz = simplify(G[1, 3])
Exz_clean = simplify(Exz * z**2 * S_f**2)
print("z² Σ² G_{xz} =", Exz_clean)

print("\n" + "=" * 70)
print("(v,z) COMPONENT => evolution or A equation")
print("=" * 70)
Evz = simplify(G[0, 3])
Evz_clean = simplify(Evz * z**2 * S_f)
print("z² Σ G_{vz} =", Evz_clean)

# Simplify for B=V=phi=0
Evz_simple = Evz_clean.subs([(B_f, 0), (V_f, 0)])
for c in coords:
    Evz_simple = Evz_simple.subs(diff(B_f, c), 0).subs(diff(V_f, c), 0)
    for c2 in coords:
        Evz_simple = Evz_simple.subs(diff(B_f, c, c2), 0).subs(diff(V_f, c, c2), 0)
Evz_simple = simplify(Evz_simple)
print("Simplified (B=V=0):", Evz_simple)

print("\n" + "=" * 70)
print("(v,v) COMPONENT => constraint / A equation")
print("=" * 70)
Evv = simplify(G[0, 0])
Evv_clean = simplify(Evv * z**4 * S_f**2)
print("z^4 Σ² G_{vv} simplified...")

# Too complex to print full, simplify for B=V=0
Evv_simple = Evv_clean.subs([(B_f, 0), (V_f, 0)])
for c in coords:
    Evv_simple = Evv_simple.subs(diff(B_f, c), 0).subs(diff(V_f, c), 0)
    for c2 in coords:
        Evv_simple = Evv_simple.subs(diff(B_f, c, c2), 0).subs(diff(V_f, c, c2), 0)
Evv_simple = simplify(Evv_simple)
print("Simplified (B=V=0):", Evv_simple)

print("\n" + "=" * 70)
print("(v,x) COMPONENT => shift V equation")
print("=" * 70)
Evx = simplify(G[0, 1])
Evx_simple = simplify(Evx * z**4 * S_f**2)
print("z^4 Σ² G_{vx} simplified...")
Evx_s = Evx_simple.subs([(B_f, 0)])
for c in coords:
    Evx_s = Evx_s.subs(diff(B_f, c), 0)
    for c2 in coords:
        Evx_s = Evx_s.subs(diff(B_f, c, c2), 0)
Evx_s = simplify(Evx_s)
print("Simplified (B=0):", Evx_s)

# (x,x) - (y,y) => B equation (traceless spatial)
print("\n" + "=" * 70)
print("(x,x)/(g_xx) - (y,y)/(g_yy) => B equation")
print("=" * 70)
# G_{xx}/g_{xx} - G_{yy}/g_{yy} gives the traceless equation for B
Exx_norm = simplify(G[1,1] / g[1,1]) if g[1,1] != 0 else 0
Eyy_norm = simplify(G[2,2] / g[2,2]) if g[2,2] != 0 else 0
B_eq = simplify(Exx_norm - Eyy_norm)
B_eq_clean = simplify(B_eq * S_f**2)
print("Σ² (G_xx/g_xx - G_yy/g_yy)...")
# Too complex, just verify pure AdS
B_test = B_eq.subs([(A_f, 1), (S_f, 1), (B_f, 0), (V_f, 0)])
for f in [A_f, S_f, B_f, V_f]:
    for c in coords:
        B_test = B_test.subs(diff(f, c), 0)
        for c2 in coords:
            B_test = B_test.subs(diff(f, c, c2), 0)
B_test = simplify(B_test)
print("Pure AdS check:", B_test)

# =================================================================
# NOW: convert to u = ln(z) coordinates
# =================================================================
print("\n" + "=" * 70)
print("CONVERTING TO u = ln(z) COORDINATES")
print("=" * 70)

u = Symbol('u')

# In u coords: z = e^u, dz = e^u du
# d/dz = e^{-u} d/du
# d²/dz² = e^{-2u}(d²/du² - d/du)

# For the (z,z) component (R1), which already vanishes for pure AdS:
# Let's extract the structure of G_{zz} = 0

# The (z,z) component of Einstein for our metric:
# G_{zz} = R_{zz} - (1/2)R g_{zz} + Λ g_{zz} = R_{zz}  (since g_{zz}=0)
# So R_{zz} = 0

print("\nR_{zz} =", simplify(R[3,3]))

# Let's get the FULL R_{zz} and convert to code-ready form
Rzz = R[3, 3]
Rzz_s = simplify(Rzz)
print("\nSimplified R_{zz} =")
print(Rzz_s)

# Similarly for the other components
print("\nR_{vz} =")
Rvz = simplify(R[0, 3])
print(Rvz)

print("\n\nDONE — use these equations to fix einstein_equations.py")
