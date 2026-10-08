from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import validate_paper_structure


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int, dropout: float):
        super().__init__()
        self.expand = nn.Linear(width, 2 * hidden)
        self.project = nn.Linear(hidden, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate, value = self.expand(x).chunk(2, dim=-1)
        return self.project(self.dropout(F.silu(gate) * value))


def mlp(in_features: int, hidden: int, out_features: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_features, hidden), nn.SiLU(), nn.Linear(hidden, out_features))


class AttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int, multiplier: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.feedforward = SwiGLU(width, width * multiplier, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid: torch.Tensor | None = None):
        padding = None
        if valid is not None:
            # Fully missing axial slices need one harmless key to avoid softmax(-inf).
            safe_valid = valid.clone()
            safe_valid[:, 0] |= ~valid.any(dim=1)
            padding = ~safe_valid
        attended = self.attention(x, x, x, key_padding_mask=padding, need_weights=False)[0]
        x = self.norm1(x + self.dropout(attended))
        x = self.norm2(x + self.dropout(self.feedforward(x)))
        return x if valid is None else x * valid.unsqueeze(-1)


class AxialEnvironmentBlock(nn.Module):
    def __init__(self, width: int, heads: int, multiplier: int, dropout: float):
        super().__init__()
        self.variable = AttentionBlock(width, heads, multiplier, dropout)
        self.region = AttentionBlock(width, heads, multiplier, dropout)
        self.temporal = AttentionBlock(width, heads, multiplier, dropout)

    def forward(self, x: torch.Tensor, valid: torch.Tensor):
        b, n, k, f, d = x.shape
        x = self.variable(x.reshape(b * n * k, f, d), valid.reshape(b * n * k, f)).reshape(b, n, k, f, d)
        r = x.permute(0, 1, 3, 2, 4).reshape(b * n * f, k, d)
        mask = valid.permute(0, 1, 3, 2).reshape(b * n * f, k)
        x = self.region(r, mask).reshape(b, n, f, k, d).permute(0, 1, 3, 2, 4)
        r = x.permute(0, 2, 3, 1, 4).reshape(b * k * f, n, d)
        mask = valid.permute(0, 2, 3, 1).reshape(b * k * f, n)
        return self.temporal(r, mask).reshape(b, k, f, n, d).permute(0, 3, 1, 2, 4)


def inverse_softplus(value: float) -> float:
    return value + math.log(-math.expm1(-value))


def glg_parameters(logits: torch.Tensor) -> dict[str, torch.Tensor]:
    eps = 1e-5
    return {
        "amplitude": F.softplus(logits[:, 0]),
        "mu": eps + (1 - 2 * eps) * torch.sigmoid(logits[:, 1]),
        "sigma": eps + F.softplus(logits[:, 2]),
        "logistic_scale": eps + F.softplus(logits[:, 3]),
        "kappa": eps + F.softplus(logits[:, 4]),
        "weights": torch.softmax(logits[:, 5:8], dim=-1),
    }


def glg_basis(tau: torch.Tensor, params: dict) -> torch.Tensor:
    difference = tau - params["mu"].unsqueeze(1)
    gaussian = torch.exp(-0.5 * (difference / params["sigma"].unsqueeze(1)) ** 2)
    logistic = torch.sigmoid(difference / params["logistic_scale"].unsqueeze(1))
    logistic = 4 * logistic * (1 - logistic)
    log_rho = -params["kappa"].unsqueeze(1) * difference
    # Log-domain evaluation avoids overflow in exp(exp(...)) and its gradients.
    log_rho = log_rho.clamp(-60, 20)
    gompertz = torch.exp(1 + log_rho - torch.exp(log_rho))
    return torch.stack((gaussian, logistic, gompertz), dim=-1)


class STVGLGFormer(nn.Module):
    def __init__(self, config: dict, glg_initial: list[float] | None = None, area_scale: float = 1.0):
        super().__init__()
        if config.get("paper_structure"):
            validate_paper_structure(config)
        model = config["model"]
        width, heads = model["d_model"], model["heads"]
        multiplier, dropout = model["ff_multiplier"], model["dropout"]
        self.max_days = config["max_days"]
        self.variables = config["variables"]
        self.area_scale = area_scale
        self.minimum_prefix = config.get("min_prefix", 1)
        k, f = config["region_rows"] * config["region_cols"], len(self.variables)
        self.area_embedding = mlp(2, width, width)
        self.observation_embedding = None
        self.calendar_embedding = mlp(3, width, width)
        self.area_position = nn.Linear(3, width, bias=False)
        self.env_embedding = mlp(1, width, width)
        self.region_embedding = nn.Embedding(k, width)
        self.variable_embedding = nn.Embedding(f, width)
        self.env_position = nn.Linear(3, width, bias=False)
        self.area_blocks = nn.ModuleList([AttentionBlock(width, heads, multiplier, dropout) for _ in range(model["encoder_layers"])])
        self.calendar_blocks = nn.ModuleList([AttentionBlock(width, heads, multiplier, dropout) for _ in range(model["encoder_layers"])])
        self.env_blocks = nn.ModuleList([AxialEnvironmentBlock(width, heads, multiplier, dropout) for _ in range(model["encoder_layers"])])
        self.time_cross = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.env_cross = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.gate_time = mlp(2 * width, width, width)
        self.gate_env = mlp(2 * width, width, width)
        self.route_logits = mlp(3 * width, width, 3 * width)
        self.condition = mlp(2 * width + 1, 2 * width, width)
        self.date_query = mlp(2, width, width)
        self.stage_query = mlp(1, width, width)
        self.lead_query = mlp(1, width, width)
        self.query_blocks = nn.ModuleList([AttentionBlock(width, heads, multiplier, dropout) for _ in range(model["query_layers"])])
        self.local_feature = mlp(2 * width, 2 * width, width)
        self.warp_head = mlp(width, width, 1)
        self.parameter_head = mlp(width, width, 8)
        self.reset_output_heads(glg_initial or [0.7, 0.28, 0.08, 0.05, 12.5])

    def reset_output_heads(self, initial: list[float]):
        amplitude, mu, sigma, logistic_scale, kappa = initial
        mu = min(max(mu, 1e-4), 1 - 1e-4)
        final = self.parameter_head[-1]
        nn.init.normal_(final.weight, mean=0, std=0.01)
        with torch.no_grad():
            final.bias.copy_(torch.tensor([
                inverse_softplus(max(amplitude, 1e-3)), math.log(mu / (1 - mu)),
                inverse_softplus(sigma), inverse_softplus(logistic_scale), inverse_softplus(kappa),
                0.0, 0.0, 0.0,
            ]))
        nn.init.zeros_(self.warp_head[-1].weight)
        nn.init.zeros_(self.warp_head[-1].bias)

    def encode(self, batch: dict) -> tuple[torch.Tensor, dict]:
        area, cloud = batch["area"], batch["cloud"]
        b, n = area.shape
        cal = batch["calendar"]
        area_state = self.area_embedding(torch.stack((area, cloud), dim=-1))
        area_state = area_state + self.area_position(cal)
        time_state = self.calendar_embedding(cal)
        for area_block, time_block in zip(self.area_blocks, self.calendar_blocks):
            area_state, time_state = area_block(area_state), time_block(time_state)
        temporal = self.time_cross(area_state, time_state, time_state, need_weights=False)[0]
        env, valid = batch["env"], batch["env_mask"]
        _, _, k, f = env.shape
        evidence_present = valid.flatten(2).any(dim=-1)
        environment = torch.zeros_like(area_state)
        if valid.any():
            x = self.env_embedding(env.unsqueeze(-1))
            x = x + self.region_embedding.weight[None, None, :, None]
            x = x + self.variable_embedding.weight[None, None, None, :]
            x = (x + self.env_position(cal)[:, :, None, None]) * valid.unsqueeze(-1)
            for block in self.env_blocks:
                x = block(x, valid)
            keys = x.reshape(b * n, k * f, -1)
            safe_valid = valid.reshape(b * n, k * f).clone()
            safe_valid[:, 0] |= ~safe_valid.any(dim=1)
            environment = self.env_cross(
                area_state.reshape(b * n, 1, -1), keys, keys,
                key_padding_mask=~safe_valid, need_weights=False,
            )[0].reshape(b, n, -1)
            environment = environment * evidence_present.unsqueeze(-1)
        gate_t = torch.sigmoid(self.gate_time(torch.cat((area_state, temporal), dim=-1)))
        gate_e = torch.sigmoid(self.gate_env(torch.cat((area_state, environment), dim=-1)))
        gate_e = gate_e * evidence_present.unsqueeze(-1)
        route_t = (1 - gate_t) * area_state + gate_t * temporal
        route_e = (1 - gate_e) * area_state + gate_e * environment
        logits = self.route_logits(torch.cat((area_state, route_t, route_e), dim=-1))
        weights = torch.softmax(logits.reshape(b, n, 3, -1), dim=2)
        routes = torch.stack((area_state, route_t, route_e), dim=2)
        state = (weights * routes).sum(dim=2)
        return state, {"time_gate": gate_t, "environment_gate": gate_e, "route_weights": weights}

    def forward(self, batch: dict) -> dict:
        state, auxiliary = self.encode(batch)
        b, n, _ = state.shape
        if n < self.minimum_prefix:
            raise ValueError(f"The model requires at least {self.minimum_prefix} observed days.")
        future = batch["future_calendar"]
        m = future.shape[1]
        if m != self.max_days - n:
            raise ValueError("Future queries must span the complete remaining max_days interval.")
        tau_obs = state.new_full((b, 1), (n - 1) / (self.max_days - 1))
        # Eq. (15) acts on D-dimensional states: normalize each channel over dates.
        alpha = torch.softmax(torch.tanh(state), dim=1)
        pooled = (alpha * state).sum(dim=1)
        condition = self.condition(torch.cat((pooled, state[:, -1], tau_obs), dim=-1))
        lead = torch.arange(1, m + 1, device=state.device, dtype=state.dtype) / self.max_days
        query = self.date_query(future[..., :2]) + self.stage_query(future[..., 2:3])
        query = query + self.lead_query(lead[None, :, None]).expand(b, -1, -1)
        for block in self.query_blocks:
            query = block(query)
        local = self.local_feature(torch.cat((query, condition[:, None].expand(-1, m, -1)), dim=-1))
        increments = F.softplus(self.warp_head(local).squeeze(-1)) + 1e-5
        # PyTorch 2.5 CUDA cumsum has no deterministic kernel; the small CPU sum preserves autograd.
        cumulative = (torch.cumsum(increments.cpu(), dim=1).to(increments.device)
                      if increments.is_cuda and torch.are_deterministic_algorithms_enabled()
                      else torch.cumsum(increments, dim=1))
        tau = tau_obs + (1 - tau_obs) * cumulative / cumulative[:, -1:]
        parameters = glg_parameters(self.parameter_head(condition))
        bases = glg_basis(tau, parameters)
        prediction = parameters["amplitude"][:, None] * (bases * parameters["weights"][:, None]).sum(-1)
        return {"prediction": prediction, "tau": tau, "parameters": parameters,
                "pool_weights": alpha, **auxiliary}
