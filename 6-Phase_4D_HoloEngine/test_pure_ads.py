"""Test that all 7 Einstein equations give zero residual for pure AdS vacuum."""
import torch
import torch.nn as nn
from metric_model import MetricReconstructor
from einstein_equations import compute_all_einstein_residuals
from ads_config import BBHConfig


class ZeroNet(nn.Module):
    """Network that outputs exactly zero — simulates pure AdS."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 5, bias=True)
        with torch.no_grad():
            self.fc.weight.zero_()
            self.fc.bias.zero_()

    def forward(self, x):
        return self.fc(x)


def main():
    print("=" * 60)
    print("  PURE AdS VACUUM TEST  (all residuals must vanish)")
    print("=" * 60)

    net = ZeroNet()
    recon = MetricReconstructor()

    # Random coords in valid range
    torch.manual_seed(42)
    N = 256
    v = torch.rand(N, 1) * 0.5
    x = torch.rand(N, 1) * 2 - 1
    u = torch.rand(N, 1) * (BBHConfig.U_MAX - BBHConfig.U_MIN) + BBHConfig.U_MIN
    coords = torch.cat([v, x, u], dim=1).requires_grad_(True)

    raw = net(coords)  # (N, 5) all ~0
    metric = recon.reconstruct(raw, coords)

    # Verify metric values are pure AdS
    print(f"\n  A    range: [{metric['A'].min():.6f}, {metric['A'].max():.6f}] (expect ~1)")
    print(f"  Sigma range: [{metric['Sigma'].min():.6f}, {metric['Sigma'].max():.6f}] (expect ~1)")
    print(f"  B    range: [{metric['B'].min():.8f}, {metric['B'].max():.8f}] (expect ~0)")
    print(f"  V    range: [{metric['V_shift'].min():.8f}, {metric['V_shift'].max():.8f}] (expect ~0)")
    print()

    residuals, derivs = compute_all_einstein_residuals(metric, coords)

    all_pass = True
    for name, res in residuals.items():
        val = res.abs().max().item()
        mean_val = res.abs().mean().item()
        status = "PASS" if val < 1e-4 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {name:20s}  max|res| = {val:.6e}  mean = {mean_val:.6e}  [{status}]")

    print()
    if all_pass:
        print("  *** ALL 7 EQUATIONS PASS PURE AdS TEST ***")
    else:
        print("  *** SOME EQUATIONS STILL FAIL ***")
    print("=" * 60)


if __name__ == "__main__":
    main()
