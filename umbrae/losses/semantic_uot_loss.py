"""Differentiable semantic-only unbalanced optimal transport supervision."""

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SemanticUOTLoss(nn.Module):
    """Log-domain entropy-regularized UOT with uniform marginals."""

    def __init__(
        self,
        epsilon: float = 0.05,
        tau: float = 1.0,
        num_iters: int = 100,
        tol: float = 1e-5,
        eps_numerical: float = 1e-8,
    ) -> None:
        super().__init__()
        if epsilon <= 0 or tau <= 0:
            raise ValueError("epsilon and tau must be positive")
        if num_iters < 1 or tol < 0 or eps_numerical <= 0:
            raise ValueError("Invalid Sinkhorn iteration parameters")
        self.epsilon = float(epsilon)
        self.tau = float(tau)
        self.num_iters = int(num_iters)
        self.tol = float(tol)
        self.eps_numerical = float(eps_numerical)

    def forward(self, h_struct: Tensor, v_patch: Tensor) -> Dict[str, object]:
        if h_struct.ndim != 3 or v_patch.ndim != 3:
            raise ValueError("h_struct and v_patch must have shape [B,N,D]")
        if h_struct.shape[0] != v_patch.shape[0]:
            raise ValueError("Structural and visual batch sizes must match")
        if h_struct.shape[-1] != v_patch.shape[-1]:
            raise ValueError("Structural and visual feature dimensions must match")

        # Disable outer autocast explicitly: casting inputs to fp32 alone does
        # not prevent autocast from lowering matmul precision.
        with torch.autocast(device_type=h_struct.device.type, enabled=False):
            h = F.normalize(h_struct.float(), dim=-1)
            v = F.normalize(v_patch.float(), dim=-1)
            cost = 1.0 - torch.matmul(h, v.transpose(1, 2))
            batch, rows, cols = cost.shape
            log_a = cost.new_full(
                (batch, rows), -torch.log(cost.new_tensor(float(rows)))
            )
            log_b = cost.new_full(
                (batch, cols), -torch.log(cost.new_tensor(float(cols)))
            )
            log_kernel = -cost / self.epsilon
            rho = self.tau / (self.tau + self.epsilon)
            log_u = torch.zeros_like(log_a)
            log_v = torch.zeros_like(log_b)
            iterations = self.num_iters

            for iteration in range(self.num_iters):
                previous_u = log_u
                previous_v = log_v
                log_u = rho * (
                    log_a
                    - torch.logsumexp(log_kernel + log_v.unsqueeze(1), dim=2)
                )
                log_v = rho * (
                    log_b
                    - torch.logsumexp(log_kernel + log_u.unsqueeze(2), dim=1)
                )
                if self.tol > 0:
                    delta = torch.maximum(
                        (log_u - previous_u).abs().amax(),
                        (log_v - previous_v).abs().amax(),
                    )
                    if float(delta.detach()) <= self.tol:
                        iterations = iteration + 1
                        break

            log_transport = log_u.unsqueeze(2) + log_kernel + log_v.unsqueeze(1)
            transport = torch.exp(log_transport)
            transport_mass_per_sample = transport.sum(dim=(1, 2))
            loss_per_sample = (transport * cost).sum(dim=(1, 2)) / (
                transport_mass_per_sample + self.eps_numerical
            )
            entropy_per_sample = -(
                transport * log_transport
            ).sum(dim=(1, 2)) / (transport_mass_per_sample + self.eps_numerical)
            probability = transport / (
                transport_mass_per_sample[:, None, None] + self.eps_numerical
            )
            probability_entropy = -(
                probability * (probability + self.eps_numerical).log()
            ).sum(dim=(1, 2))
            normalized_entropy_per_sample = probability_entropy / torch.log(
                cost.new_tensor(float(rows * cols))
            )
            effective_support_per_sample = probability_entropy.exp()
            max_transport_fraction_per_sample = probability.amax(dim=(1, 2))
            row_mass = transport.sum(dim=2)
            column_mass = transport.sum(dim=1)
            row_mass_cv_per_sample = row_mass.std(dim=1, unbiased=False) / (
                row_mass.mean(dim=1) + self.eps_numerical
            )
            column_mass_cv_per_sample = column_mass.std(dim=1, unbiased=False) / (
                column_mass.mean(dim=1) + self.eps_numerical
            )
        return {
            "loss": loss_per_sample.mean(),
            "transport": transport,
            "cost": cost,
            "transport_mass": transport_mass_per_sample.mean(),
            "transport_mass_per_sample": transport_mass_per_sample,
            "transport_entropy": entropy_per_sample.mean(),
            "transport_entropy_per_sample": entropy_per_sample,
            "normalized_transport_entropy": normalized_entropy_per_sample.mean(),
            "normalized_transport_entropy_per_sample": normalized_entropy_per_sample,
            "effective_support": effective_support_per_sample.mean(),
            "effective_support_per_sample": effective_support_per_sample,
            "max_transport_fraction": max_transport_fraction_per_sample.mean(),
            "max_transport_fraction_per_sample": max_transport_fraction_per_sample,
            "row_mass": row_mass,
            "col_mass": column_mass,
            "column_mass": column_mass,
            "row_mass_cv": row_mass_cv_per_sample.mean(),
            "row_mass_cv_per_sample": row_mass_cv_per_sample,
            "column_mass_cv": column_mass_cv_per_sample.mean(),
            "column_mass_cv_per_sample": column_mass_cv_per_sample,
            "num_iters": iterations,
        }
