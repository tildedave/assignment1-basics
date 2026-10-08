from collections.abc import Iterable
from math import sqrt, cos, pi
from operator import itemgetter

import torch
from torch import Tensor
from jaxtyping import Float, Int
from einops import rearrange


def cross_entropy(inputs: Float[Tensor, "... vocab_size"], targets: Int[Tensor, "..."]) -> Float[Tensor, ""]:
    largest, _ = inputs.max(dim=-1, keepdim=True)
    denom = torch.exp(inputs - largest)
    tensor = torch.log(denom.sum(dim=-1, keepdim=True)) - (inputs - largest)
    return torch.gather(tensor, -1, rearrange(targets, "... -> ... 1")).mean()


def perplexity(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """
    not yet tested ;-)
    """
    largest, _ = inputs.max(dim=-1, keepdim=True)
    denom = torch.exp(inputs - largest)
    tensor = torch.log(denom.sum(dim=-1, keepdim=True)) - (inputs - largest)
    losses = torch.gather(tensor, -1, rearrange(targets, "... -> ... 1"))
    return losses.mean().exp()


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}

        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, betas, eps, weight_decay = itemgetter("lr", "betas", "eps", "weight_decay")(group)
            beta_1, beta_2 = betas

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                m = state.get("m", torch.zeros(p.shape, device=p.device, dtype=p.dtype))
                v = state.get("v", torch.zeros(p.shape, device=p.device, dtype=p.dtype))
                t = state.get("t", 1)

                grad: torch.Tensor = p.grad.data
                a_t = lr * sqrt(1 - beta_2**t) / (1 - beta_1**t)
                p.data -= lr * weight_decay * p.data
                m = beta_1 * m + (1 - beta_1) * grad
                v = beta_2 * v + (1 - beta_2) * grad * grad
                p.data -= a_t * m / (v.sqrt() + eps)

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss


def learning_rate_schedule(
    t: int,
    a_max: float,
    a_min: float,
    t_w: int,
    t_c: int,
):
    if t < t_w:
        return (t / t_w) * a_max
    if t > t_c:
        return a_min

    return a_min + (1 / 2) * (1 + cos(pi * (t - t_w) / (t_c - t_w))) * (a_max - a_min)


def gradient_clipping(params: Iterable[torch.nn.Parameter], max_l2_norm: float, eps=1e-6):
    params_list = list(params)
    grads = [p.grad for p in params_list if p.grad is not None]
    if not grads:  # weird but OK
        return

    grads = [g.norm() for g in grads]
    total_norm = torch.linalg.norm(torch.stack(grads))
    s = max_l2_norm / (total_norm + eps)
    mult = torch.clamp(s, max=1.0)

    if total_norm < max_l2_norm:
        return

    for p in params_list:
        if p.grad is None:
            continue
        p.grad *= mult
