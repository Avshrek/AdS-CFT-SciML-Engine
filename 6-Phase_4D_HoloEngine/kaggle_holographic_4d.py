"""
kaggle_holographic_4d.py
========================
Single-file Kaggle-ready 6-Phase 4D Holographic Quantum Gravity Engine.

COMPREHENSIVE FIX — addresses all training instabilities:
  FIX  1: omega_0 = 30 (was 60)     — halves derivative magnitude, quarters d²
  FIX  2: epsilon = 0.1 (was 10)    — gentle causal PINN weighting
  FIX  3: kappa = 10/5 (was 150/50) — backreaction max 6× not 51×
  FIX  4: LR_B = 1e-4 (was 5e-5)   — adequate Phase B learning rate
  FIX  5: patience = 200 (was 100)  — prevents premature LR decay
  FIX  6: W_PDE = 1.0 (was 0.01)   — retuned for log-space residuals
  FIX  7: W_CAUCHY_DT = 0.1         — very soft momentum, Phase B only
  FIX  8: PDE_WARMUP = 200 epochs   — gradual physics introduction
  FIX  9: NaN recovery (5 retries)  — checkpoint reload on divergence
  FIX 10: log1p(res²) PDE           — maps any residual to O(1)–O(50)
  FIX 11: log1p Sommerfeld           — prevents O(10^13) domination
  FIX 12: Cauchy bulk-interior only  — prevents contradiction with bnd data
  FIX 13: log1p causality penalty    — bounded acausal penalty
  FIX 14: Phase A = PURE DATA       — no cauchy, no momentum, no PDE
  FIX 15: Phase B warmup on physics  — prevents gradient shock at transition
"""

# ====================================================================== #
#                        0.  IMPORTS                                       #
# ====================================================================== #
import os, sys, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except Exception:
    HAS_PENNYLANE = False
    print("[info] PennyLane not found – using classical entropy proxy.")


# ====================================================================== #
#                        1.  CONFIG                                        #
# ====================================================================== #
class Config:
    Z_MIN, Z_MAX       = 1e-4, 1.0
    U_MIN, U_MAX       = math.log(1e-4), 0.0          # ≈ -9.21 … 0
    U_BOUNDARY          = math.log(1e-4)
    X_RANGE, Y_RANGE    = (-1., 1.), (-1., 1.)
    T_RANGE             = (0., 1.)

    # ---------- Network Architecture ----------
    LATENT_DIM          = 128
    SIREN_HIDDEN        = 256
    SIREN_LAYERS        = 5
    # FIX 1: omega_0 = 30 (was 60). Halves 1st-derivative magnitude,
    # quarters 2nd-derivative magnitude. Still captures merger features.
    SIREN_OMEGA_0       = 30.0

    ENCODER_CHANNELS    = [1, 16, 32, 64]
    ENCODER_TEMPORAL_FRAMES = 100
    ENCODER_SPATIAL_RES     = 64

    # ---------- Hardware & Geometry ----------
    BOUNDARY_BATCH      = 1024
    BULK_BATCH          = 2048
    NUM_TIME_CHUNKS     = 16
    # FIX 2: epsilon = 0.1 (was 10). With ε=10, chunks after the first
    # get weight exp(-10·cumsum) ≈ 0. Now all time slices contribute.
    CAUSAL_EPSILON      = 0.1

    # ---------- Physics ----------
    DELTA, LAMBDA_NL    = 3.0, 1.0
    # FIX 3: kappa 10/5 (was 150/50). Old br = 1+50 = 51× PDE amplification.
    # New br = 1+5 = 6× maximum. Keeps backreaction meaningful but bounded.
    KAPPA, KAPPA_MAX    = 10.0, 5.0

    # ---------- Training Schedule ----------
    TOTAL_EPOCHS                = 2000
    CURRICULUM_PHASE_A_EPOCHS   = 800
    # FIX 4: Phase B LR = 1e-4 (was 5e-5). After Phase A produces a
    # well-fit SIREN, Phase B needs adequate LR to adjust for physics.
    LR_PHASE_A, LR_PHASE_B     = 5e-4, 1e-4
    GRAD_CLIP                   = 1.0
    GRADIENT_ACCUMULATION_STEPS = 1
    # FIX 5: patience = 200 (was 100). Old: 7+ halvings in Phase A
    # (LR fell to 3.9e-6). New: at most 3 halvings in 800 epochs.
    SCHEDULER_PATIENCE          = 200
    SCHEDULER_FACTOR            = 0.5

    # ---------- Loss Weights ----------
    W_DATA              = 500.0
    # FIX 6: W_PDE = 1.0 (was 0.01). With log1p, PDE values are O(10-50)
    # instead of O(10^29). Re-scaled so PDE contributes ~50 vs data ~500.
    W_PDE               = 1.0
    W_CAUCHY            = 1.0          # Phase B only (FIX 14)
    # FIX 7: W_CAUCHY_DT = 0.1 (was 1.0). Very soft momentum constraint,
    # only in Phase B, restricted to bulk interior (FIX 12).
    W_CAUCHY_DT         = 0.1
    W_SOMMERFELD        = 0.1
    W_HRT               = 0.0
    W_CAUSALITY_HRT     = 1.0
    W_QUANTUM           = 10.0

    # ---------- Warmup & Safety ----------
    # FIX 8: Gradual introduction of physics after Phase A.
    # At epoch 801 warmup=0.005, at epoch 1001 warmup=1.0.
    PDE_WARMUP_EPOCHS   = 200
    # FIX 9: Auto NaN recovery. On NaN: reload best checkpoint, halve LR.
    NAN_MAX_RECOVERIES  = 5

    NUM_QUBITS          = 10
    QUANTUM_LAYERS      = 3
    QUANTUM_UPDATE_EVERY = 10

    DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"
    MIXED_PRECISION     = False
    CHECKPOINT_DIR      = "checkpoints"
    LOG_EVERY           = 10
    SAVE_EVERY          = 200


# ====================================================================== #
#                2.  DATA                                                   #
# ====================================================================== #
def load_master_dataset(cfg=Config, sim_idx=1):
    """Load the Holographic Boundary Dataset from .npz (Kaggle)."""
    candidates = [
        '/kaggle/input/datasets/avshrek/holographic-4d-engine/apex_master_dataset.npz',
        '/kaggle/input/holographic-4d-engine/apex_master_dataset.npz',
        'apex_master_dataset.npz',
    ]
    dataset_path = None
    for p in candidates:
        if os.path.exists(p):
            dataset_path = p; break
    if dataset_path is None:
        return None  # Fall back to synthetic

    data = np.load(dataset_path)
    cnn_vol = torch.from_numpy(data['cnn_volumes'][sim_idx]).float()
    pinn_pts = data['pinn_points']
    sim_pts = pinn_pts[pinn_pts[:, 0] == sim_idx]
    bnd_coords = torch.from_numpy(sim_pts[:, 1:5]).float()
    bnd_values = torch.from_numpy(sim_pts[:, 5]).float()
    entropy_target = torch.from_numpy(data['entropy_targets'][sim_idx]).float()
    time_ticks = torch.from_numpy(data['time_ticks']).float()
    return cnn_vol, bnd_coords, bnd_values, entropy_target, time_ticks


def generate_synthetic_data(cfg=Config):
    """Synthetic spiraling binary merger for local testing."""
    T = cfg.ENCODER_TEMPORAL_FRAMES
    H = W = cfg.ENCODER_SPATIAL_RES
    xv = torch.linspace(*cfg.X_RANGE, W)
    yv = torch.linspace(*cfg.Y_RANGE, H)
    yy, xx = torch.meshgrid(yv, xv, indexing="ij")
    data = torch.zeros(T, H, W)
    for ti in range(T):
        t = ti / max(T - 1, 1)
        sep = 0.6 * (1.0 - t)
        ang = 2.5 * math.pi * t
        cx1, cy1 = sep * math.cos(ang), sep * math.sin(ang)
        sig = 0.15 + 0.10 * (1 - t)
        amp = 1.0 + 2.0 * t
        b1 = amp * torch.exp(-((xx - cx1)**2 + (yy - cy1)**2) / (2*sig**2))
        b2 = amp * torch.exp(-((xx + cx1)**2 + (yy + cy1)**2) / (2*sig**2))
        frame = b1 + b2
        if t > 0.7:
            rf = (t - 0.7) / 0.3
            r = torch.sqrt(xx**2 + yy**2 + 1e-8)
            frame += rf * 0.5 * torch.sin(12*r - 8*t) * torch.exp(-r**2 / 0.25)
        data[ti] = frame
    data = (data - data.min()) / (data.max() - data.min() + 1e-8)

    # Build boundary coordinates [t, x, y, u_boundary]
    tv = torch.linspace(*cfg.T_RANGE, T)
    tt, yy_c, xx_c = torch.meshgrid(tv, yv, xv, indexing="ij")
    bnd_coords = torch.stack([tt.flatten(), xx_c.flatten(), yy_c.flatten(),
                               torch.full((T*H*W,), cfg.U_BOUNDARY)], -1)
    bnd_values = data.flatten()
    entropy_target = torch.tensor(3.0)
    return data, bnd_coords, bnd_values, entropy_target, tv


# ====================================================================== #
#                    3.  APEX DUAL SAMPLER                                  #
# ====================================================================== #
class ApexDualSampler:
    def __init__(self, bnd_coords, bnd_values, cfg=Config):
        self.cfg = cfg
        self.bnd_c = bnd_coords
        self.bnd_v = bnd_values

    def sample_discrete_boundary(self, B):
        idx = torch.randint(0, self.bnd_c.shape[0], (B,))
        return self.bnd_c[idx], self.bnd_v[idx]

    def sample_continuous_bulk(self, B):
        c = self.cfg
        t = torch.rand(B) * (c.T_RANGE[1]-c.T_RANGE[0]) + c.T_RANGE[0]
        x = torch.rand(B) * (c.X_RANGE[1]-c.X_RANGE[0]) + c.X_RANGE[0]
        y = torch.rand(B) * (c.Y_RANGE[1]-c.Y_RANGE[0]) + c.Y_RANGE[0]
        u = torch.rand(B) * (c.U_MAX-c.U_MIN) + c.U_MIN
        return torch.stack([t,x,y,u], -1), torch.exp(-3.*u)

    def sample_cauchy_surface(self, B):
        c = self.cfg
        x = torch.rand(B)*(c.X_RANGE[1]-c.X_RANGE[0])+c.X_RANGE[0]
        y = torch.rand(B)*(c.Y_RANGE[1]-c.Y_RANGE[0])+c.Y_RANGE[0]
        u = torch.rand(B)*(c.U_MAX-c.U_MIN)+c.U_MIN
        return torch.stack([torch.zeros(B), x, y, u], -1)

    def sample_sommerfeld_boundary(self, B):
        c = self.cfg;  n = B // 4
        t = torch.rand(B)*(c.T_RANGE[1]-c.T_RANGE[0])+c.T_RANGE[0]
        u = torch.rand(B)*(c.U_MAX-c.U_MIN)+c.U_MIN
        cl, el = [], []
        for i,(fv,dn) in enumerate([(1.,"x+"),(- 1.,"x-"),(1.,"y+"),(- 1.,"y-")]):
            s = i*n;  e = s+n if i<3 else B
            te, ue, m = t[s:e], u[s:e], e-s
            if dn[0]=="x": xe=torch.full((m,),fv); ye=torch.rand(m)*2-1
            else:           xe=torch.rand(m)*2-1;  ye=torch.full((m,),fv)
            cl.append(torch.stack([te,xe,ye,ue],-1))
            el.append(torch.full((m,),i,dtype=torch.long))
        return torch.cat(cl), torch.cat(el)


# ====================================================================== #
#          4.  NEURAL ARCHITECTURE: FiLM-SIREN + ConvEncoder3D             #
# ====================================================================== #
class SineLayer(nn.Module):
    def __init__(self, din, dout, w0=30., first=False):
        super().__init__(); self.w0=w0; self.first=first
        self.linear = nn.Linear(din, dout)
        with torch.no_grad():
            b = 1./din if first else math.sqrt(6./din)/w0
            self.linear.weight.uniform_(-b, b)

    def forward(self, x, gamma=None, beta=None):
        h = self.w0 * self.linear(x)
        if gamma is not None: h = gamma * h + beta
        return torch.sin(h)

class FiLMSiren(nn.Module):
    def __init__(self, cfg=Config):
        super().__init__()
        H, nL = cfg.SIREN_HIDDEN, cfg.SIREN_LAYERS
        self.first = SineLayer(4, H, cfg.SIREN_OMEGA_0, first=True)
        self.hidden = nn.ModuleList([SineLayer(H, H, cfg.SIREN_OMEGA_0) for _ in range(nL-1)])
        self.out = nn.Linear(H, 1)
        with torch.no_grad():
            b = math.sqrt(6./H)/cfg.SIREN_OMEGA_0
            self.out.weight.uniform_(-b, b)
        self.film = nn.ModuleList([
            nn.Sequential(nn.Linear(cfg.LATENT_DIM, H), nn.SiLU(), nn.Linear(H, 2*H))
            for _ in range(nL)])

    def forward(self, coords, z):
        fp = []
        for g in self.film:
            o = g(z); ga, be = o.chunk(2,-1); fp.append((ga+1, be))
        h = self.first(coords, fp[0][0], fp[0][1])
        for i, ly in enumerate(self.hidden):
            h = ly(h, fp[i+1][0], fp[i+1][1])
        return self.out(h)

class ConvEncoder3D(nn.Module):
    def __init__(self, cfg=Config):
        super().__init__()
        ch = cfg.ENCODER_CHANNELS
        blk = []
        for i in range(len(ch)-1):
            blk += [nn.Conv3d(ch[i],ch[i+1],3,2,1), nn.BatchNorm3d(ch[i+1]), nn.LeakyReLU(.2,True)]
        self.conv = nn.Sequential(*blk)
        t,h,w = cfg.ENCODER_TEMPORAL_FRAMES, cfg.ENCODER_SPATIAL_RES, cfg.ENCODER_SPATIAL_RES
        for _ in range(len(ch)-1): t=(t+1)//2; h=(h+1)//2; w=(w+1)//2
        flat = ch[-1]*t*h*w
        self.fc = nn.Sequential(nn.Linear(flat, 512), nn.SiLU(),
                                nn.Linear(512, cfg.LATENT_DIM))

    def forward(self, x):
        h = self.conv(x); return self.fc(h.reshape(h.size(0),-1))


# ====================================================================== #
#          5.  PHYSICS ENGINE                                              #
# ====================================================================== #
def _ag(out, inp, cg=True):
    return torch.autograd.grad(out, inp, torch.ones_like(out),
                               create_graph=cg, retain_graph=True)[0]

def compute_derivs(phi, c):
    g = _ag(phi, c)
    dt,dx,dy,du = g[:,0:1],g[:,1:2],g[:,2:3],g[:,3:4]
    return dict(phi=phi, dt=dt, dx=dx, dy=dy, du=du,
                dt2=_ag(dt,c)[:,0:1], dx2=_ag(dx,c)[:,1:2],
                dy2=_ag(dy,c)[:,2:3], du2=_ag(du,c)[:,3:4])

def causal_bizon_pde(phi_R, coords, cfg=Config):
    """AdS wave equation with conformal-frame backreaction."""
    u = coords[:,3:4]; e2u = torch.exp(2.*u); e6u = torch.exp(6.*u)
    d = compute_derivs(phi_R, coords)
    with torch.no_grad():
        # Conformal frame stress-energy (numerically stable, no e6u)
        Tp_conformal = .5 * (d['dt']**2 + d['dx']**2 + d['dy']**2 + d['du']**2)
        br = 1. + torch.clamp(cfg.KAPPA * Tp_conformal, max=cfg.KAPPA_MAX)
    res = (-br*e2u*d['dt2'] + e2u*(d['dx2']+d['dy2'])
           + d['du2'] + 3.*d['du'] - cfg.LAMBDA_NL * phi_R**3 * e6u)
    return res, d

def causal_pinn_weights(res, t_c, cfg=Config, volume_w=None):
    """Causal PINN weighting with LOG-SPACE residual (FIX 10).

    Old: rs = res². With SIREN derivatives O(ω₀²)=3600 and br=51×,
    rs ~ O(10^7–10^10) per point → total loss O(10^29).

    New: rs = log1p(res²). Maps ANY residual to O(1)–O(50).
    Gradient: d/dθ log1p(r²) = 2r/(1+r²) · dr/dθ — naturally dampened
    for large residuals, equal treatment for small ones.
    """
    B = res.shape[0]; dev = res.device
    idx = torch.argsort(t_c)
    rs = torch.log1p(res[idx].squeeze(-1)**2)
    if volume_w is not None:
        ws = volume_w[idx]; ws = ws / (ws.mean() + 1e-8)
        rs = rs * ws
    dx = 2./(B**.25+1e-8); nc = min(max(int(math.ceil(1./dx)), cfg.NUM_TIME_CHUNKS),
                                     cfg.NUM_TIME_CHUNKS*2)
    cs = max(B//nc,1); cl=[]
    for k in range(nc):
        s=k*cs; e=min(s+cs,B) if k<nc-1 else B; cl.append(rs[s:e].mean())
    cl = torch.stack(cl)
    cum = torch.cumsum(cl,0).detach()
    sh = torch.cat([torch.zeros(1,device=dev), cum[:-1]])
    w = torch.exp(-cfg.CAUSAL_EPSILON * sh)
    return (w*cl).sum()/(w.sum()+1e-8)

def sommerfeld_loss(model, sampler, z_det, cfg=Config):
    """Sommerfeld radiation BC with log1p stabilization (FIX 11)."""
    dev = z_det.device
    c, et = sampler.sample_sommerfeld_boundary(cfg.BOUNDARY_BATCH)
    c = c.to(dev).requires_grad_(True); et = et.to(dev)
    phi = model(c, z_det[:1].expand(c.shape[0],-1))
    g = _ag(phi, c); dt,dx,dy = g[:,0:1],g[:,1:2],g[:,2:3]
    L = torch.zeros(1,device=dev)
    for i,op in enumerate([(dt,dx,1),(dt,dx,-1),(dt,dy,1),(dt,dy,-1)]):
        m = (et==i)
        if m.any(): L = L + ((op[0][m]+op[2]*op[1][m])**2).mean()
    return torch.log1p(L).squeeze()

def bulk_cauchy_loss(model, sampler, z_det, cfg=Config):
    """Cauchy IC (φ=0, ∂_tφ=0 at t=0) — BULK INTERIOR ONLY (FIX 12).

    Old code enforced φ=0 at ALL (x,y,u) including u≈U_BOUNDARY where
    the boundary data has non-zero initial conditions (Gaussian BH bumps).
    This contradiction + the momentum term drove ml to 10^12.

    Now: only enforce for u > U_MIN + 2.0, i.e., deep in the AdS bulk
    where vacuum conditions are physically correct.
    """
    dev = z_det.device
    c = sampler.sample_cauchy_surface(cfg.BOUNDARY_BATCH).to(dev).requires_grad_(True)
    phi = model(c, z_det[:1].expand(c.shape[0],-1))
    g = _ag(phi, c)
    # Restrict to bulk interior (away from AdS boundary at U_MIN ≈ -9.21)
    u_vals = c[:, 3].detach()
    mask = (u_vals > cfg.U_MIN + 2.0)
    if mask.any():
        fl = (phi[mask]**2).mean()
        ml = (g[mask, 0:1]**2).mean()
    else:
        fl = torch.tensor(0., device=dev, requires_grad=True)
        ml = torch.tensor(0., device=dev, requires_grad=True)
    return fl, ml

def hrt_covariant_area(phi_R, coords, cfg=Config):
    """HRT area via Differential Swelling + log1p causality (FIX 13)."""
    u = coords[:,3:4]; e2u = torch.exp(2.*u); e3u = torch.exp(3.*u)
    d = compute_derivs(phi_R, coords)
    # Conformal frame energy difference
    Ttt_conf = .5 * d['dt']**2
    Txx_conf = .5 * d['dx']**2
    diff_conf = torch.abs(Ttt_conf - Txx_conf)
    # Differential Swelling (bypasses 1.0 baseline washout in FP32)
    ke = torch.clamp(cfg.KAPPA * diff_conf, max=cfg.KAPPA_MAX)
    excess_growth = (torch.sqrt(1. + ke + 1e-8) - 1.0).mean()
    hrt = 1.0 + excess_growth
    # Bulk causality (speed of light)
    dt_b = e3u * d['dt']; dx_b = e3u * d['dx']
    dy_b = e3u * d['dy']; du_b = e3u * (d['du'] + 3.*d['phi'])
    gn = -e2u*dt_b**2 + e2u*dx_b**2 + e2u*dy_b**2 + du_b**2
    # log1p prevents causality penalty from O(10^14)
    cp = torch.log1p(1000.*torch.relu(gn).mean())
    return hrt, cp


# ====================================================================== #
#          6.  QUANTUM ENTROPY TETHER                                      #
# ====================================================================== #
class QuantumEntropyTether:
    def __init__(self, cfg=Config):
        self.cfg = cfg; self.cached = None
        if HAS_PENNYLANE:
            self.dev = qml.device("default.qubit", wires=cfg.NUM_QUBITS)
            nq, nl = cfg.NUM_QUBITS, cfg.QUANTUM_LAYERS
            @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
            def circ(p):
                for i in range(nq): qml.Hadamard(i)
                for l in range(nl):
                    for i in range(nq): qml.RY(p[l,i,0],i); qml.RZ(p[l,i,1],i)
                    for i in range(nq-1): qml.CNOT([i,i+1])
                    qml.CNOT([nq-1,0])
                return qml.state()
            self._circ = circ

    def _l2p(self, z):
        n = self.cfg.QUANTUM_LAYERS*self.cfg.NUM_QUBITS*2
        e = z.detach().repeat(math.ceil(n/self.cfg.LATENT_DIM))[:n]
        return (e*math.pi).reshape(self.cfg.QUANTUM_LAYERS,self.cfg.NUM_QUBITS,2)

    def compute(self, z):
        if not HAS_PENNYLANE:
            p = torch.softmax(z.detach().float(),-1).clamp(min=1e-12)
            return -(p*p.log()).sum()
        s = self._circ(self._l2p(z))
        nA = self.cfg.NUM_QUBITS//2; dA,dB = 2**nA, 2**(self.cfg.NUM_QUBITS-nA)
        psi = s.reshape(dA,dB); rho = psi @ psi.conj().T
        ev = torch.linalg.eigvalsh(rho.real).clamp(min=1e-12)
        return -(ev*ev.log()).sum()

    def update_cache(self, z):
        with torch.no_grad(): self.cached = self.compute(z).detach()

    def get_cached(self):
        return self.cached if self.cached is not None else torch.tensor(0.)

def quantum_tether_loss(hrt, S):
    return (hrt - S)**2


# ====================================================================== #
#          7.  LOSS FUNCTIONS  (Phase A / Phase B)                         #
# ====================================================================== #
def loss_phase_a(enc, sir, smp, bnd_in, cfg=Config):
    """Phase A: PURE DATA FITTING — no Cauchy, no momentum, no PDE (FIX 14).

    WHY: The old Phase A had bulk_cauchy_loss which enforced ∂_tφ(t=0)=0
    across all depths including the AdS boundary. But the boundary data at
    t=0 has non-zero initial conditions (BH Gaussians). This contradiction
    caused the momentum term to grow from 0.02 to 10^12 in 20 epochs,
    drowning the data gradient (W_DATA×data=750 << momentum=10^12).
    The optimizer spent 800 epochs fighting itself and never learned the data
    (peak_phi stuck at 0.17–0.25 instead of ≥1.0).

    The fix: Phase A trains ONLY on boundary data + t0 emphasis. This lets
    the encoder and SIREN fully learn the waveform before physics constraints
    are introduced in Phase B.
    """
    dev = cfg.DEVICE
    bc, bv = smp.sample_discrete_boundary(cfg.BOUNDARY_BATCH)
    bc, bv = bc.to(dev), bv.to(dev)

    z = enc(bnd_in)
    ze = z.expand(bc.shape[0], -1)
    pred = sir(bc, ze).squeeze(-1)

    # Amplitude weighting: 20× penalty for missing merger peaks
    err = (pred - bv)**2
    amp_weights = 1.0 + 20.0 * torch.abs(bv)
    data_l = (err * amp_weights).mean()

    # Strict t=0 initial condition lock (extra 10× weight on t<0.1 points)
    t0_mask = (bc[:, 0] < 0.1)
    if t0_mask.any():
        t0_err = (pred[t0_mask] - bv[t0_mask])**2
        t0_weights = 1.0 + 20.0 * torch.abs(bv[t0_mask])
        t0_loss = (t0_err * t0_weights).mean() * 10.0
    else:
        t0_loss = torch.tensor(0., device=dev)

    tot = cfg.W_DATA * data_l + t0_loss
    info = dict(data=data_l.item(),
                t0_lock=t0_loss.item() if torch.is_tensor(t0_loss) else t0_loss,
                total=tot.item())
    return tot, info

def loss_phase_b(enc, sir, smp, bnd_in, qt, cfg=Config, epoch=0):
    """Phase B: Full physics with warmup (FIX 15).

    Physics constraints are multiplied by a warmup factor:
      warmup = min(1.0, (epoch - Phase_A_end) / PDE_WARMUP_EPOCHS)

    At epoch  801: warmup = 0.005 → physics contributes ~0.5% of full weight
    At epoch 1001: warmup = 1.0   → full physics active

    This prevents the gradient shock that caused PDE=10^29 at epoch 810.
    """
    dev = cfg.DEVICE
    phase_b_ep = max(1, epoch - cfg.CURRICULUM_PHASE_A_EPOCHS)
    warmup = min(1.0, phase_b_ep / max(cfg.PDE_WARMUP_EPOCHS, 1))

    with torch.no_grad(): z = enc(bnd_in)
    zd = z.detach()

    # ---- Data loss (always full weight) ----
    bc, bv = smp.sample_discrete_boundary(cfg.BOUNDARY_BATCH)
    bc, bv = bc.to(dev), bv.to(dev)
    pred_bnd = sir(bc, zd.expand(bc.shape[0],-1)).squeeze(-1)
    err = (pred_bnd - bv)**2
    amp_weights = 1.0 + 20.0 * torch.abs(bv)
    data_l = (err * amp_weights).mean()

    # ---- PDE loss (warmed up, log-space) ----
    bk, bw = smp.sample_continuous_bulk(cfg.BULK_BATCH)
    bk = bk.to(dev).requires_grad_(True); bw = bw.to(dev)
    phi_R = sir(bk, zd.expand(bk.shape[0],-1))
    res, _ = causal_bizon_pde(phi_R, bk, cfg)
    pde_l = causal_pinn_weights(res, bk[:,0].detach(), cfg, volume_w=bw.detach())

    # ---- Cauchy (warmed up, bulk interior only) ----
    fl, ml = bulk_cauchy_loss(sir, smp, zd, cfg)

    # ---- Sommerfeld (warmed up, log-space) ----
    sl = sommerfeld_loss(sir, smp, zd, cfg)

    # ---- HRT + causality (warmed up causality, log-space) ----
    hc, _ = smp.sample_continuous_bulk(cfg.BULK_BATCH//2)
    hc = hc.to(dev).requires_grad_(True)
    phi_h = sir(hc, zd.expand(hc.shape[0],-1))
    hrt, cp = hrt_covariant_area(phi_h, hc, cfg)

    # ---- Quantum tether ----
    cS = qt.get_cached().to(dev)
    ql = quantum_tether_loss(hrt, cS)

    # Total: data always at full weight, physics multiplied by warmup
    tot = (cfg.W_DATA * data_l
           + warmup * (cfg.W_PDE * pde_l
                       + cfg.W_CAUCHY * fl
                       + cfg.W_CAUCHY_DT * ml
                       + cfg.W_SOMMERFELD * sl
                       + cfg.W_CAUSALITY_HRT * cp)
           + cfg.W_HRT * hrt
           + cfg.W_QUANTUM * ql)

    info = dict(data=data_l.item(), pde=pde_l.item(), cauchy=fl.item(),
                momentum=ml.item(), sommerfeld=sl.item(), hrt=hrt.item(),
                causality=cp.item(), quantum=ql.item(), warmup=warmup,
                total=tot.item())
    return tot, info


# ====================================================================== #
#          8.  CURRICULUM TRAINING LOOP                                    #
# ====================================================================== #
def train(cfg=Config):
    dev = cfg.DEVICE
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)

    print("=" * 72)
    print("  4D HOLOGRAPHIC QUANTUM GRAVITY ENGINE  –  FIXED Edition")
    print("=" * 72)

    # ---- Load data (master .npz or synthetic fallback) ----
    result = load_master_dataset(cfg, sim_idx=1)
    if result is not None:
        cnn_vol, bnd_coords, bnd_values, entropy_target, time_ticks = result
        bnd_in = cnn_vol
        while bnd_in.dim() < 5:
            bnd_in = bnd_in.unsqueeze(0)
        bnd_in = bnd_in.to(dev)
        print(f"[data]   loaded master dataset -> encoder in {tuple(bnd_in.shape)}")
    else:
        print("[data]   master dataset not found, generating synthetic data...")
        cnn_vol, bnd_coords, bnd_values, entropy_target, time_ticks = generate_synthetic_data(cfg)
        bnd_in = cnn_vol.unsqueeze(0).unsqueeze(0).to(dev)
        print(f"[data]   synthetic {tuple(cnn_vol.shape)} -> encoder in {tuple(bnd_in.shape)}")

    # ---- Models ----
    enc = ConvEncoder3D(cfg).to(dev)
    sir = FiLMSiren(cfg).to(dev)
    ne = sum(p.numel() for p in enc.parameters())
    ns = sum(p.numel() for p in sir.parameters())
    print(f"[model]  encoder {ne:,}  |  FiLM-SIREN {ns:,}  |  total {ne+ns:,}")

    # ---- Print fix summary ----
    print(f"[fixes]  omega_0={cfg.SIREN_OMEGA_0}  kappa={cfg.KAPPA}/{cfg.KAPPA_MAX}"
          f"  eps={cfg.CAUSAL_EPSILON}  patience={cfg.SCHEDULER_PATIENCE}")
    print(f"[fixes]  Phase A = PURE DATA (no cauchy/momentum/PDE)")
    print(f"[fixes]  Phase B = log1p PDE + {cfg.PDE_WARMUP_EPOCHS}-epoch warmup"
          f" + NaN recovery ({cfg.NAN_MAX_RECOVERIES} max)")

    smp = ApexDualSampler(bnd_coords, bnd_values, cfg)
    qt  = QuantumEntropyTether(cfg)

    opt_a = optim.Adam(list(enc.parameters())+list(sir.parameters()), lr=cfg.LR_PHASE_A)
    opt_b = optim.Adam(sir.parameters(), lr=cfg.LR_PHASE_B)
    sched_a = optim.lr_scheduler.ReduceLROnPlateau(opt_a, patience=cfg.SCHEDULER_PATIENCE,
                                                    factor=cfg.SCHEDULER_FACTOR)
    sched_b = optim.lr_scheduler.ReduceLROnPlateau(opt_b, patience=cfg.SCHEDULER_PATIENCE,
                                                    factor=cfg.SCHEDULER_FACTOR)

    history, best = [], float("inf")
    nan_count = 0
    accum = cfg.GRADIENT_ACCUMULATION_STEPS

    print(f"\n{'='*72}")
    print(f"  PHASE A  –  Pure Data Fitting  (1 -> {cfg.CURRICULUM_PHASE_A_EPOCHS})")
    print(f"{'='*72}")

    for ep in range(1, cfg.TOTAL_EPOCHS + 1):
        t0 = time.time()

        # ---- Phase transition ----
        if ep == cfg.CURRICULUM_PHASE_A_EPOCHS + 1:
            print(f"\n{'='*72}")
            print(f"  PHASE B  –  Full Physics + Warmup  ({ep} -> {cfg.TOTAL_EPOCHS})")
            print(f"  Freezing CNN Encoder ...")
            print(f"{'='*72}\n")
            for p in enc.parameters(): p.requires_grad = False
            enc.eval()
            with torch.no_grad(): z0 = enc(bnd_in).squeeze(0)
            qt.update_cache(z0)

        # ---- Phase A ----
        if ep <= cfg.CURRICULUM_PHASE_A_EPOCHS:
            enc.train(); sir.train(); opt_a.zero_grad()
            loss, info = loss_phase_a(enc, sir, smp, bnd_in, cfg)

            # NaN guard (FIX 9)
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                print(f"  [!NaN] Phase A epoch {ep}, recovery {nan_count}/{cfg.NAN_MAX_RECOVERIES}")
                if nan_count > cfg.NAN_MAX_RECOVERIES:
                    print("  [FATAL] Too many NaN events. Stopping."); break
                ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "best_model.pt")
                if os.path.exists(ckpt_path):
                    ckpt = torch.load(ckpt_path, map_location=dev)
                    enc.load_state_dict(ckpt['encoder']); sir.load_state_dict(ckpt['siren'])
                for pg in opt_a.param_groups: pg['lr'] *= 0.5
                opt_a.zero_grad(); continue

            (loss / accum).backward()
            if ep % accum == 0 or ep == cfg.CURRICULUM_PHASE_A_EPOCHS:
                nn.utils.clip_grad_norm_(list(enc.parameters())+list(sir.parameters()),
                                         cfg.GRAD_CLIP)
                opt_a.step(); opt_a.zero_grad()
            sched_a.step(info['total'])

        # ---- Phase B ----
        else:
            sir.train(); opt_b.zero_grad()
            if (ep - cfg.CURRICULUM_PHASE_A_EPOCHS) % cfg.QUANTUM_UPDATE_EVERY == 1:
                with torch.no_grad(): z0 = enc(bnd_in).squeeze(0)
                qt.update_cache(z0)
            loss, info = loss_phase_b(enc, sir, smp, bnd_in, qt, cfg, epoch=ep)

            # NaN guard (FIX 9)
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                print(f"  [!NaN] Phase B epoch {ep}, recovery {nan_count}/{cfg.NAN_MAX_RECOVERIES}")
                if nan_count > cfg.NAN_MAX_RECOVERIES:
                    print("  [FATAL] Too many NaN events. Stopping."); break
                ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "best_model.pt")
                if os.path.exists(ckpt_path):
                    ckpt = torch.load(ckpt_path, map_location=dev)
                    sir.load_state_dict(ckpt['siren'])
                for pg in opt_b.param_groups: pg['lr'] *= 0.5
                opt_b.zero_grad(); continue

            (loss / accum).backward()
            if ep % accum == 0:
                nn.utils.clip_grad_norm_(sir.parameters(), cfg.GRAD_CLIP)
                opt_b.step(); opt_b.zero_grad()
            sched_b.step(info['total'])

        dt_ep = time.time() - t0
        history.append(info)

        # ---- Logging ----
        if ep <= 5 or ep % cfg.LOG_EVERY == 0:
            ph = "A" if ep <= cfg.CURRICULUM_PHASE_A_EPOCHS else "B"
            lr = opt_a.param_groups[0]['lr'] if ph=="A" else opt_b.param_groups[0]['lr']

            with torch.no_grad():
                z_sample = enc(bnd_in)
                sample_coords, _ = smp.sample_discrete_boundary(1000)
                phi_peak = sir(sample_coords.to(dev), z_sample.expand(1000, -1)).abs().max().item()

            parts = " | ".join(f"{k}={v:.5f}" for k,v in info.items())
            print(f"[{ph}] E{ep:4d}/{cfg.TOTAL_EPOCHS}  lr={lr:.2e} | peak_phi={phi_peak:.3f} | {parts}  ({dt_ep:.1f}s)")

        # ---- Checkpoint ----
        if not (torch.isnan(loss) or torch.isinf(loss)):
            if info['total'] < best:
                best = info['total']
                torch.save(dict(epoch=ep, encoder=enc.state_dict(),
                                siren=sir.state_dict(), best_loss=best),
                           os.path.join(cfg.CHECKPOINT_DIR, "best_model.pt"))
        if ep % cfg.SAVE_EVERY == 0:
            torch.save(dict(epoch=ep, encoder=enc.state_dict(),
                            siren=sir.state_dict(), history=history),
                       os.path.join(cfg.CHECKPOINT_DIR, f"ckpt_{ep}.pt"))

    print(f"\n{'='*72}")
    print(f"  TRAINING COMPLETE  |  best loss = {best:.8f}")
    print(f"  NaN recoveries used: {nan_count}/{cfg.NAN_MAX_RECOVERIES}")
    print(f"{'='*72}")
    torch.save(dict(epoch=cfg.TOTAL_EPOCHS, encoder=enc.state_dict(),
                    siren=sir.state_dict(), history=history),
               os.path.join(cfg.CHECKPOINT_DIR, "final_model.pt"))
    return enc, sir, history


# ====================================================================== #
#          9.  ENTRY POINT                                                 #
# ====================================================================== #
if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU : {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
    else:
        print("WARNING: No GPU – running on CPU (will be slow).")
    train(Config)
