import torch
from torch import Tensor


def softclip(tensor: Tensor, min_value: float) -> Tensor:
    return min_value + F.softplus(tensor - min_value)


def gaussian_nll(mu: Tensor, log_sigma: Tensor, x: Tensor) -> Tensor:
    log_a = log_sigma + 0.5 * np.log(2 * np.pi)
    sigma = log_sigma.exp()
    return log_a + 0.5 * torch.pow((x - mu) / sigma, 2)