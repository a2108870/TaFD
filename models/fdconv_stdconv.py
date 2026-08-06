
# -*- coding: utf-8 -*-
"""
FDConv (drop-in) with:
- rFFT routing on Zernike/ZCH basis
- Correct reshape using conv output H2/W2 (fixes shape invalid errors)
- Concentric-ring init using ONLY Zernike R2^0 + per-expert bias
  (others init to 0 but remain trainable)
- Hard routing: num_domains == num_experts, compute K paths,
  then select by sample domain index (no weighted fusion)

Usage: from models.fdconv import FDConv
"""

import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


def _zernike_basis_dim(n_max: int, m_max: int) -> int:
    """Calculate total number of Zernike basis functions (N_basis in paper)."""
    cnt = 0
    for n in range(n_max + 1):
        m_top = min(n, m_max)
        for m in range(m_top + 1):
            if (n - m) % 2 != 0:
                continue
            cnt += (1 if m == 0 else 2)
    return cnt


def _device_key(device: torch.device) -> str:
    """Generate unique key for device."""
    return f"{device.type}:{device.index if device.index is not None else -1}"


# ==================== Zernike Basis Generation ====================
def generate_zernike_basis(h: int, w: int, device: torch.device, n_max: int = 3, m_max: int = 3):
    """
    Generate Zernike basis functions on rFFT half-plane.

    Args:
        h: Height of spatial domain
        w: Width of spatial domain
        device: Torch device
        n_max: Maximum radial degree
        m_max: Maximum azimuthal degree

    Returns:
        B: [M, H, W'] Zernike basis (RMS-normalized)
        B_flat: [M, H*W'] flattened basis
        rn: [H, W'] normalized radius in [0,1] on rFFT half-plane
    """
    # Frequency coordinates
    fy = torch.fft.fftfreq(h, device=device)
    fx = torch.fft.rfftfreq(w, device=device)
    FY, FX = torch.meshgrid(fy, fx, indexing='ij')

    # Normalized radius
    r = torch.sqrt(FY * FY + FX * FX)
    r_max = torch.sqrt(FY.abs().max() ** 2 + FX.abs().max() ** 2) + 1e-12
    rn = torch.clamp(r / r_max, 0., 1.)

    # Angular components
    cos_t = torch.where(r > 1e-9, FX / (r + 1e-12), torch.ones_like(r))
    sin_t = torch.where(r > 1e-9, FY / (r + 1e-12), torch.zeros_like(r))

    # Compute cos(m*theta) and sin(m*theta) by recurrence
    cos_list = [torch.ones_like(rn)]
    sin_list = [torch.zeros_like(rn)]

    for m in range(1, m_max + 1):
        c_prev, s_prev = cos_list[-1], sin_list[-1]
        c_m = c_prev * cos_t - s_prev * sin_t
        s_m = s_prev * cos_t + c_prev * sin_t
        cos_list.append(c_m)
        sin_list.append(s_m)

    # Zernike radial polynomial
    def zernike_radial(n, m, rr):
        Rnm = torch.zeros_like(rr)
        half = (n - m) // 2
        for s in range(half + 1):
            cs1 = math.comb(n - s, s)
            cs2 = math.comb(n - 2 * s, half - s)
            coef = ((-1) ** s) * float(cs1 * cs2)
            Rnm = Rnm + coef * (rr ** (n - 2 * s))
        return Rnm

    # Build basis functions
    basis_list = []
    for n in range(n_max + 1):
        m_top = min(n, m_max)
        for m in range(m_top + 1):
            if (n - m) % 2 != 0:
                continue

            R_nm = zernike_radial(n, m, rn)
            if m == 0:
                basis_list.append(R_nm)  # Isotropic component
            else:
                basis_list.append(R_nm * cos_list[m])  # Cos component
                basis_list.append(R_nm * sin_list[m])  # Sin component

    # Stack and normalize
    B = torch.stack(basis_list, dim=0)  # [M, H, W']
    eps = 1e-6
    B = B / torch.sqrt((B * B).mean(dim=(1, 2), keepdim=True) + eps)
    B_flat = B.reshape(B.shape[0], -1).contiguous()

    return B, B_flat, rn


# ==================== Zernike Basis Cache ====================
class ZernikeBasisCache:
    """LRU cache for Zernike basis functions."""

    def __init__(self, max_items: int = 8, n_max: int = 3, m_max: int = 3):
        self.max = int(max_items)
        self.n_max = n_max
        self.m_max = m_max
        self.basis_cache = OrderedDict()

    def _touch(self, od: OrderedDict, key, value):
        """Add item and evict oldest if over capacity."""
        od[key] = value
        if len(od) > self.max:
            od.popitem(last=False)
        return value

    def get_zernike_basis(self, h: int, w: int, device: torch.device):
        """Get or generate Zernike basis with caching."""
        key = (h, w, _device_key(device), f'Zernike_{self.n_max}_{self.m_max}')

        if key in self.basis_cache:
            self.basis_cache.move_to_end(key)
            return self.basis_cache[key]

        B, B_flat, rn = generate_zernike_basis(h, w, device, self.n_max, self.m_max)
        return self._touch(self.basis_cache, key, (B, B_flat, rn))


# ==================== FC-Conv Module ====================
class FCConv(nn.Module):
    """
    Frequency-Conditional Convolution (FC-Conv) with hard routing.

    Processing pipeline:
    rFFT -> Zernike basis -> per-domain spectral masks -> iFFT -> K-expert conv
         -> hard routing by threat-domain index -> residual
    """

    # Zernike basis configuration
    DEFAULT_N_MAX = 3              # n_max: maximum radial degree
    DEFAULT_M_MAX = 3              # m_max: maximum azimuthal degree
    DEFAULT_SOFTMAX_TEMP = 1.0     # τ: softmax temperature for spectral masks

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
            num_experts: int = 4,
            num_domains: int = 1,
            prior_temperature: float = 1.0,
            freq_cache_items: int = 8
    ):
        super().__init__()

        assert num_experts >= 1, "num_experts must be >= 1"
        assert num_domains == num_experts, \
            f"num_domains ({num_domains}) must equal num_experts ({num_experts}) for hard routing"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_groups = int(groups)
        self.num_threat_domains = int(num_domains)
        self.prior_temperature = float(prior_temperature)
        self.K = int(num_experts)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # K dedicated experts + residual
        self.dedicated_experts = nn.Conv2d(
            in_channels * self.K,
            out_channels * self.K,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=self.K * groups,
            bias=False
        )

        self.residual_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=groups,
            bias=False
        )

        self.bias_param = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # Zernike basis configuration (instance-level)
        self.n_max = self.DEFAULT_N_MAX
        self.m_max = self.DEFAULT_M_MAX
        self.softmax_temperature = self.DEFAULT_SOFTMAX_TEMP
        self.num_basis_functions = _zernike_basis_dim(self.n_max, self.m_max)

        # Threat-domain embedding: each domain has vector of length K*(M+1)
        self.threat_domain_embedding = nn.Embedding(
            self.num_threat_domains,
            self.K * (self.num_basis_functions + 1)
        )
        nn.init.zeros_(self.threat_domain_embedding.weight)

        # Zernike basis cache (with n_max, m_max)
        self._basis_cache = ZernikeBasisCache(max_items=freq_cache_items, n_max=self.n_max, m_max=self.m_max)

    def _get_radial_basis_indices(self):
        """Get indices of radial (m=0) basis functions."""
        indices = []
        cur = 0

        for n in range(self.n_max + 1):
            m_top = min(n, self.m_max)
            for m in range(m_top + 1):
                if (n - m) % 2 != 0:
                    continue

                if m == 0:
                    indices.append((n, cur))
                    cur += 1
                else:
                    cur += 2

        return indices

    @torch.no_grad()
    def init_spectral_masks(
            self,
            H_ref: int,
            W_ref: int,
            mode: str = "equal_radius",
            gain: float = 12.0
    ):
        """
        Initialize spectral masks using concentric rings (R2^0 + bias).

        Args:
            H_ref: Reference height
            W_ref: Reference width
            mode: 'equal_radius' or 'equal_area'
            gain: Gain factor for slopes
        """
        device = self.threat_domain_embedding.weight.device
        K = self.K
        M_full = self.num_basis_functions

        # Find R2^0 index in basis sequence
        radial_indices = self._get_radial_basis_indices()
        defocus_idx = next((idx for (n, idx) in radial_indices if n == 2), None)

        if defocus_idx is None:
            raise RuntimeError("R_2^0 not available; ensure n_max>=2.")

        # Compute RMS normalization factor for (2r^2-1) on rFFT half-plane
        Bset, Bflat, rn = self._basis_cache.get_zernike_basis(H_ref, W_ref, device)
        s_unnorm = 2.0 * (rn ** 2) - 1.0
        s_rms = torch.sqrt((s_unnorm ** 2).mean()).item()

        # Target radii boundaries
        if mode == "equal_radius":
            r_bounds = [k / K for k in range(1, K)]
        elif mode == "equal_area":
            r_bounds = [(k / K) ** 0.5 for k in range(1, K)]
        else:
            raise ValueError("mode must be 'equal_radius' or 'equal_area'")

        # Map to normalized basis values
        s_bounds_norm = [(2 * (r * r) - 1.0) / s_rms for r in r_bounds]

        # Slopes and intercepts (monotonic b_k, recursive a_k)
        b = torch.linspace(-1.0, 1.0, K, device=device) * gain
        a = torch.zeros(K, device=device)

        for k in range(K - 1):
            a[k + 1] = a[k] + (b[k] - b[k + 1]) * s_bounds_norm[k]

        # Write to embedding: bias=a; only R2^0 dim set to b; others=0 (learnable)
        params = torch.zeros(K, M_full + 1, device=device)
        params[:, 0] = a
        params[:, 1 + defocus_idx] = b
        flat = params.reshape(-1)

        for d in range(self.num_threat_domains):
            self.threat_domain_embedding.weight[d] = flat.clone()

        return {
            "mode": mode,
            "gain": float(gain),
            "H_ref": int(H_ref),
            "W_ref": int(W_ref),
            "r_bounds": [float(x) for x in r_bounds],
            "s_rms": float(s_rms),
            "a": [float(x) for x in a],
            "b": [float(x) for x in b],
            "R20_index": int(defocus_idx),
            "NUM_BASIS": int(M_full),
        }

    def _compute_spectral_masks(
            self,
            h: int,
            w: int,
            device: torch.device,
            threat_domain_index: torch.Tensor,
            router_weights: torch.Tensor = None,
            bpda: bool = False
    ):
        """
        Generate spectral masks on rFFT half-plane.

        Returns:
            spectral_masks: [B, K, 1, H, W'] frequency masks
        """
        B_batch = threat_domain_index.shape[0]
        Bset, Bflat, _ = self._basis_cache.get_zernike_basis(h, w, device)
        H_f, W_f = h, Bset.shape[-1]

        # Threat-domain embedding: forward uses hard lookup; backward uses soft mixture
        params_hard = self.threat_domain_embedding(threat_domain_index)  # [B, K*(M+1)]

        use_soft = (
            bool(bpda)
            and (router_weights is not None)
            and (router_weights.dim() == 2)
            and (router_weights.size(0) == B_batch)
            and (router_weights.size(1) == self.num_threat_domains)
        )

        if use_soft:
            # soft domain embedding = w @ Embedding.weight
            W = self.threat_domain_embedding.weight.detach()              # [D, K*(M+1)]
            rw = router_weights.to(W.dtype)                         # [B, D]
            params_soft = rw @ W                                          # [B, K*(M+1)]

            # BPDA detach trick: forward == hard; backward gradient from soft
            params_flat = params_hard.detach() - params_soft.detach() + params_soft
        else:
            params_flat = params_hard

        params = params_flat.view(B_batch, self.K, self.num_basis_functions + 1)
        bias = params[..., 0]   # a_k
        C = params[..., 1:]     # Zernike coeffs

        # scores -> softmax masks
        S_flat = torch.matmul(C, Bflat) + bias.unsqueeze(-1)  # [B, K, H*W']
        spectral_masks_flat = F.softmax(
            S_flat / max(self.softmax_temperature, 1e-8),
            dim=1
        )
        spectral_masks = spectral_masks_flat.view(B_batch, self.K, H_f, W_f).unsqueeze(2)
        return spectral_masks

    def _compute_unique_selected_spectral_masks(
            self,
            h: int,
            w: int,
            device: torch.device,
            unique_domains: torch.Tensor,
    ):
        num_unique = unique_domains.numel()
        _, Bflat, _ = self._basis_cache.get_zernike_basis(h, w, device)
        H_f = h
        W_f = Bflat.shape[-1] // h

        params_flat = self.threat_domain_embedding(unique_domains)
        params = params_flat.view(num_unique, self.K, self.num_basis_functions + 1)
        bias = params[..., 0]
        coeffs = params[..., 1:]

        scores = torch.matmul(coeffs, Bflat) + bias.unsqueeze(-1)
        spectral_masks = F.softmax(
            scores / max(self.softmax_temperature, 1e-8),
            dim=1,
        )
        selected_idx = unique_domains.view(num_unique, 1, 1).expand(-1, 1, scores.size(-1))
        selected_masks = spectral_masks.gather(1, selected_idx)
        return selected_masks.view(num_unique, 1, 1, H_f, W_f)

    def _forward_hard_exact(self, x: torch.Tensor, threat_domain_index: torch.Tensor):
        B, _, _, _ = x.shape

        bands_flat = x.repeat(1, self.K, 1, 1)
        expert_out_flat = self.dedicated_experts(bands_flat)
        H2, W2 = expert_out_flat.shape[-2], expert_out_flat.shape[-1]
        expert_out = expert_out_flat.view(B, self.K, self.out_channels, H2, W2)

        batch_idx = torch.arange(B, device=expert_out.device)
        y_hard = expert_out[batch_idx, threat_domain_index, :, :, :]

        expert_usage = {}
        for d in torch.unique(threat_domain_index, sorted=False).tolist():
            v = torch.zeros(self.K, device=x.device)
            v[d] = 1.0
            expert_usage[f'domain_{d}'] = v

        residual = self.residual_conv(x)
        out = residual + y_hard

        if self.bias_param is not None:
            out = out + self.bias_param.view(1, -1, 1, 1)

        return out, expert_usage, None

    def forward(self, x, threat_domain_index=None, router_weights=None, bpda: bool = False, *_, **kwargs):
        """
        Forward pass with hard routing in forward, BPDA soft surrogate in backward.

        Args:
            x: [B, C_in, H, W]
            threat_domain_index: [B] conditioning signal k*
            router_weights: [B, K] soft weights from router (only used when bpda=True)
            bpda: enable BPDA surrogate gradient

        Returns:
            out: [B, C_out, H2, W2]
            expert_usage: dict
            spectral_masks: [B, K, 1, H, W']
        """
        B, C_in, H, W = x.shape
        device = x.device

        # Default domain 0
        if threat_domain_index is None:
            threat_domain_index = torch.zeros(B, dtype=torch.long, device=device)
        else:
            threat_domain_index = torch.clamp(threat_domain_index, 0, self.num_threat_domains - 1)

        use_soft = (
                bool(bpda)
                and (router_weights is not None)
                and (router_weights.dim() == 2)
                and (router_weights.size(0) == B)
                and (router_weights.size(1) == self.K)
        )

        if not use_soft:
            return self._forward_hard_exact(x, threat_domain_index)

        bands_flat = x.repeat(1, self.K, 1, 1)
        expert_out_flat = self.dedicated_experts(bands_flat)
        H2, W2 = expert_out_flat.shape[-2], expert_out_flat.shape[-1]
        expert_out = expert_out_flat.view(B, self.K, self.out_channels, H2, W2)

        # 5) Hard routing (forward), Soft routing surrogate (backward)
        batch_idx = torch.arange(B, device=expert_out.device)
        Y_hard = expert_out[batch_idx, threat_domain_index, :, :, :]  # [B, C_out, H2, W2]

        w = router_weights.to(expert_out.dtype).view(B, self.K, 1, 1, 1)
        Y_soft = (w * expert_out).sum(dim=1)  # [B, C_out, H2, W2]

        Y = Y_hard.detach() - Y_soft.detach() + Y_soft

        # 6) Residual
        residual = self.residual_conv(x)
        out = residual + Y

        if self.bias_param is not None:
            out = out + self.bias_param.view(1, -1, 1, 1)

        # 7) Expert usage (forward is still hard routing)
        expert_usage = {}
        for d in range(self.num_threat_domains):
            m = (threat_domain_index == d)
            if m.any():
                v = torch.zeros(self.K, device=expert_out.device)
                v[d] = 1.0
                expert_usage[f'domain_{d}'] = v

        return out, expert_usage, None
