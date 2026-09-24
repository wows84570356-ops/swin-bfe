
"""The state-space context block (Swin-BFE-Mamba).

A selective-scan core with foreground routing, four-direction scanning,
directional fusion, a detached drift measurement and an anti-dilution
gate. Corresponds to Eqs. 5-8 of the paper.
"""

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


def _inv_softplus(y):
    return math.log(math.expm1(y))


class SelectiveScanCore(nn.Module):

    def __init__(self, d_model, d_state=16, dt_rank="auto"):
        super().__init__()
        self.d_model, self.d_state = d_model, d_state
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.in_proj = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.x_proj = nn.Linear(d_model, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)
                      .repeat(d_model, 1)))
        self.D = nn.Parameter(torch.ones(d_model))
        self.p_bias = nn.Parameter(torch.zeros(d_model))

    @torch.autocast("cuda", enabled=False)
    def forward_direction(self, x_2d, p_2d=None):


        x_2d = x_2d.float()
        B, C, H, W = x_2d.shape
        x = self.in_proj(x_2d.reshape(B, C, H * W)).float()
        dt_raw, B_raw, C_raw = torch.split(
            self.x_proj(x.transpose(1, 2)).float(),
            [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt_raw).transpose(1, 2).float()
        if p_2d is not None:
            dt = dt + self.p_bias.view(1, -1, 1).float() * \
                p_2d.reshape(B, 1, H * W).float()
        A = -torch.exp(self.A_log.float().clamp(max=10))
        out = selective_scan_fn(
            x, dt.contiguous(), A,
            B_raw.transpose(1, 2).contiguous().float(),
            C_raw.transpose(1, 2).contiguous().float(),
            self.D.float(), z=None, delta_bias=None,
            delta_softplus=True, return_last_state=False)
        return out.reshape(B, C, H, W)


class ShapeOnlyCore(nn.Module):

    _warned = False

    def __init__(self, d_model, **kw):
        super().__init__()
        if not ShapeOnlyCore._warned:
            warnings.warn("mamba_ssm not found: using the ShapeOnlyCore stand-in, for shape tests only.")
            ShapeOnlyCore._warned = True
        self.conv = nn.Conv1d(d_model, d_model, 4, groups=d_model)
        self.p_bias = nn.Parameter(torch.zeros(d_model))

    def forward_direction(self, x_2d, p_2d=None):
        B, C, H, W = x_2d.shape
        s = F.pad(x_2d.reshape(B, C, H * W), (3, 0))
        y = F.silu(self.conv(s))
        if p_2d is not None:
            y = y * (1 + self.p_bias.view(1, -1, 1)
                     * p_2d.reshape(B, 1, H * W))
        return y.reshape(B, C, H, W)


class IdentityCore(nn.Module):

    def forward_direction(self, x_2d, p_2d=None):
        return x_2d


class DirectionalFusion(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Conv2d(channels * 4 + 4, 4, kernel_size=1)
        with torch.no_grad():
            self.attn.weight[:, channels * 4:, :, :].zero_()

    def forward(self, hs, dilution_summaries):
        a = F.softmax(self.attn(torch.cat(list(hs) + list(dilution_summaries),
                                          dim=1)), dim=1)
        m = sum(a[:, i:i + 1] * hs[i] for i in range(4))
        return m, a


class DilutionEvidence(nn.Module):

    def __init__(self, bg_pool=15):
        super().__init__()
        self.bg_pool = bg_pool

    @staticmethod
    def _hf(t):
        return t - F.avg_pool2d(t, 3, 1, 1)

    @torch.autocast("cuda", enabled=False)
    def forward(self, x_ref, hs, a):


        x_ref = x_ref.float()
        hs = [h.float() for h in hs]
        a = a.float()
        delta = sum(a[:, i:i + 1] * (hs[i] - x_ref) for i in range(4))
        m_ref = sum(a[:, i:i + 1] * hs[i] for i in range(4))
        k = self.bg_pool
        mu = F.avg_pool2d(x_ref, k, 1, k // 2)
        d_bg = mu - x_ref
        num = (delta * d_bg).sum(1, keepdim=True)
        den = torch.linalg.vector_norm(d_bg, dim=1, keepdim=True) + 1e-3
        proj = (num / den).clamp(-1e4, 1e4)
        hf_drop = (self._hf(x_ref).abs().mean(1, keepdim=True)
                   - self._hf(m_ref).abs().mean(1, keepdim=True)
                   ).clamp(-1e4, 1e4)
        return delta, F.relu(proj), F.relu(hf_drop)


class CoherenceMap(nn.Module):
    def __init__(self):
        super().__init__()
        sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8
        self.register_buffer("kx", sx.view(1, 1, 3, 3))
        self.register_buffer("ky", sx.t().contiguous().view(1, 1, 3, 3))

    def forward(self, s):
        gx = F.conv2d(s, self.kx, padding=1)
        gy = F.conv2d(s, self.ky, padding=1)
        w = lambda t: F.avg_pool2d(t, 5, 1, 2)
        jxx, jyy, jxy = w(gx * gx), w(gy * gy), w(gx * gy)
        coh = ((jxx - jyy) ** 2 + 4 * jxy ** 2) \
            / ((jxx + jyy) ** 2 + 1e-8)
        return coh.clamp(0, 1)


class AntiDilutionGate(nn.Module):
    def __init__(self, channels, groups=8, max_gate_init=0.7,
                 use_coherence=True):
        super().__init__()
        if channels % groups != 0:
            warnings.warn(f"channels({channels}) is not divisible by groups({groups}), "
                          f"falling back to groups=1")
            groups = 1
        self.groups = groups
        self.use_coherence = use_coherence
        mid = max(channels // 2, 16)
        self.det = nn.Sequential(
            nn.Conv2d(channels * 2 + 2, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.GELU(),
            nn.Conv2d(mid, groups, 3, padding=1))
        self.max_gate_raw = nn.Parameter(
            torch.tensor(_inv_softplus(max_gate_init)))
        if use_coherence:
            self.coherence = CoherenceMap()
            self.rho_raw = nn.Parameter(torch.tensor(-4.0))

    @torch.autocast("cuda", enabled=False)
    def forward(self, x, m, delta, proj, hf_drop, P_gate,
                disable_coherence=False):
        x, m = x.float(), m.float()
        delta = delta.float()
        proj, hf_drop = proj.float(), hf_drop.float()
        P_gate = P_gate.float()
        det_in = torch.cat([x.detach(), delta.abs().clamp(max=1e4),
                            proj, hf_drop], dim=1)
        I = torch.sigmoid(self.det(det_in))
        I = I * F.softplus(self.max_gate_raw) * P_gate
        if self.use_coherence and not disable_coherence:
            coh = self.coherence(delta.abs().mean(1, keepdim=True))
            I = I * (1 - torch.sigmoid(self.rho_raw) * (1 - coh))
        I_full = I.repeat_interleave(x.shape[1] // self.groups, dim=1)
        y = m + I_full * (x - m)
        return y, I


class StateSpaceContext(nn.Module):

    def __init__(self, channels, lambda_bg=0.1, max_gate=0.7,
                 gate_groups=8, use_coherence=True, use_delta_bias=True,
                 core=None):
        super().__init__()
        self.lambda_bg = lambda_bg
        self.use_delta_bias = use_delta_bias


        self.stroke_prior = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.BatchNorm2d(channels // 2), nn.GELU(),
            nn.Conv2d(channels // 2, 1, 1), nn.Sigmoid())


        lap = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]) / 4
        self.register_buffer("lap", lap.view(1, 1, 3, 3))
        self.thin_prior = nn.Sequential(
            nn.Conv2d(channels + 1, max(channels // 4, 16), 3, padding=1),
            nn.BatchNorm2d(max(channels // 4, 16)), nn.GELU(),
            nn.Conv2d(max(channels // 4, 16), 1, 1), nn.Sigmoid())

        if core is not None:
            self.mamba_core = core
        elif HAS_MAMBA:
            self.mamba_core = SelectiveScanCore(d_model=channels)
        else:
            self.mamba_core = ShapeOnlyCore(channels)

        self.fusion = DirectionalFusion(channels)
        self.evidence = DilutionEvidence()
        self.gate = AntiDilutionGate(channels, groups=gate_groups,
                                       max_gate_init=max_gate,
                                       use_coherence=use_coherence)

    def _scan4(self, x_fg, p):
        core = self.mamba_core
        h_right = core.forward_direction(x_fg, p)
        h_left = torch.flip(
            core.forward_direction(torch.flip(x_fg, [3]),
                                   None if p is None else torch.flip(p, [3])),
            [3])
        xT = x_fg.transpose(2, 3)
        pT = None if p is None else p.transpose(2, 3)
        h_down = core.forward_direction(xT, pT).transpose(2, 3)
        h_up = torch.flip(
            core.forward_direction(torch.flip(xT, [3]),
                                   None if pT is None else torch.flip(pT, [3])),
            [3]).transpose(2, 3)
        return [h_up, h_down, h_left, h_right]

    def forward(self, x, return_importance=False, ext_prior=None,
                return_aux=False, disable_gate=False, disable_coherence=False):
        P_route = self.stroke_prior(x)
        x_fg = x * P_route + self.lambda_bg * x * (1.0 - P_route)

        p = P_route if self.use_delta_bias else None
        hs = self._scan4(x_fg, p)


        s_d = [(x_fg - h).abs().mean(1, keepdim=True).detach() for h in hs]
        m, a = self.fusion(hs, s_d)

        if disable_gate:
            y, I = m, None
            delta = proj = hf_drop = coh_note = None
        else:


            if ext_prior is not None:
                P_gate = ext_prior.detach()
            else:
                hf = F.conv2d(x.mean(1, keepdim=True), self.lap,
                              padding=1).abs()
                P_gate = self.thin_prior(torch.cat([x, hf], dim=1))


            delta, proj, hf_drop = self.evidence(
                x_fg.detach(), [h.detach() for h in hs], a.detach())
            y, I = self.gate(x, m, delta, proj, hf_drop, P_gate,
                             disable_coherence=disable_coherence)

        if return_aux:
            aux = {"I": I, "P_route": P_route,
                   "P_gate": None if disable_gate else P_gate,
                   "dir_weights": a,
                   "dir_dilution": torch.cat(s_d, 1),
                   "delta_mag": None if delta is None
                   else delta.abs().mean(1, keepdim=True),
                   "proj_bg": proj, "hf_drop": hf_drop}
            return y, aux
        if return_importance:
            return y, (None if I is None else I.mean(1, keepdim=True))
        return y


__all__ = ["StateSpaceContext", "SelectiveScanCore",
           "AntiDilutionGate", "DirectionalFusion",
           "DilutionEvidence", "CoherenceMap", "ShapeOnlyCore", "IdentityCore"]
