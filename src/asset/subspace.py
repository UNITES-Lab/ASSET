from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize


class Coefficients(torch.nn.Module):
    def __init__(self, rank: int):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.zeros(rank))


class Subspace(torch.nn.Module):
    def __init__(self, basis: torch.Tensor, coefficients: Coefficients):
        super().__init__()
        self.register_buffer("basis", basis)
        self.coefficients = coefficients

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight - torch.einsum("r,r...->...", self.coefficients.alpha, self.basis.to(weight.dtype))


def energy_rank(eigvals: torch.Tensor, energy: float) -> int:
    total = eigvals.sum()
    if float(total.item()) <= 0.0:
        return 0
    return int(torch.searchsorted(torch.cumsum(eigvals, dim=0), total * float(energy)).item() + 1)


def module_basis(grads: list[torch.Tensor], energy: float, rank_cap: int) -> tuple[torch.Tensor, float]:
    stacked = torch.stack([grad.flatten().cpu().to(dtype=torch.float64) for grad in grads], dim=0)
    rank_cap = min(rank_cap, int(stacked.shape[0]), int(stacked.shape[1]))
    eigvals, eigvecs = torch.linalg.eigh(stacked @ stacked.T)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order].clamp_min(0.0)
    eigvecs = eigvecs[:, order]
    total = float(eigvals.sum().item())
    if total <= 1e-12:
        return torch.zeros(stacked.shape[1], 0), total
    keep = eigvals > max(1e-12, 1e-8 * float(eigvals[0].item()))
    eigvals = eigvals[keep]
    eigvecs = eigvecs[:, keep]
    rank = min(energy_rank(eigvals, energy), rank_cap)
    basis = stacked.T @ eigvecs[:, :rank]
    basis = basis / eigvals[:rank].sqrt().unsqueeze(0)
    return F.normalize(basis.to(dtype=torch.float32), dim=0).cpu(), total


def gradient_bases(
    modules: dict[str, dict],
    grads: dict[str, list[torch.Tensor]],
    weights: list[float],
    energy: float,
    rank_cap: int,
    energy_floor: float,
) -> dict[str, torch.Tensor]:
    estimates = {}
    for name, info in modules.items():
        weighted = [grad * math.sqrt(max(weight, 1e-12)) for grad, weight in zip(grads[name], weights)]
        basis, total = module_basis(weighted, energy, min(rank_cap, int(info["a_shape"][0])))
        if basis.shape[1] > 0:
            estimates[name] = (basis, total)
    floor = max((total for _, total in estimates.values()), default=0.0) * energy_floor
    return {name: basis for name, (basis, total) in estimates.items() if total >= floor}


def random_bases(modules: dict[str, dict], rank_cap: int, seed: int) -> dict[str, torch.Tensor]:
    bases = {}
    for index, (name, info) in enumerate(modules.items()):
        size = math.prod(info["a_shape"]) + math.prod(info["b_shape"])
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + index)
        matrix = torch.randn(size, min(rank_cap, int(info["a_shape"][0])), generator=generator)
        bases[name] = torch.linalg.qr(matrix, mode="reduced")[0]
    return bases


def register_subspace(modules: dict[str, dict], bases: dict[str, torch.Tensor]) -> list[Coefficients]:
    coefficients = []
    for name, basis in bases.items():
        info = modules[name]
        device = info["a"].weight.device
        a_size = math.prod(info["a_shape"])
        b_size = math.prod(info["b_shape"])
        basis_a = basis[:a_size, :].T.reshape(basis.shape[1], *info["a_shape"]).to(device)
        basis_b = basis[a_size : a_size + b_size, :].T.reshape(basis.shape[1], *info["b_shape"]).to(device)
        holder = Coefficients(basis.shape[1]).to(device)
        parametrize.register_parametrization(info["a"], "weight", Subspace(basis_a, holder), unsafe=True)
        parametrize.register_parametrization(info["b"], "weight", Subspace(basis_b, holder), unsafe=True)
        coefficients.append(holder)
    return coefficients


def remove_subspace(modules: dict[str, dict], names: Iterable[str]) -> None:
    for name in names:
        parametrize.remove_parametrizations(modules[name]["a"], "weight", leave_parametrized=True)
        parametrize.remove_parametrizations(modules[name]["b"], "weight", leave_parametrized=True)
