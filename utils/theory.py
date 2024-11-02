import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .utils import all_elements_in_targets, set_seed


def Phi(X: Tensor, Z: Tensor, gamma: float, n_sample: int = 1000, seed: int = 0) -> Tensor:
    '''
    Args:
        X: (N1, d) := (the number of data, dim)
        Z: (N2, d) := (the number of data, dim)
    
    Returns:
        Phi: (N1, N2), Phi_{ij} := Phi(X_i, Z_j)
    '''

    assert X.ndim == 2, X.ndim
    assert 0 <= gamma < 1, gamma

    set_seed(seed)

    d = X.shape[1]

    v = torch.normal(0, 1 / np.sqrt(d), (n_sample, d), device=X.device) # (n_sample, d)
    a = torch.normal(0, 1, (n_sample, 1), device=X.device) # (n_sample, 1)

    tX = v @ X.T + a # (n_sample, N1)
    tX = torch.where(tX >= 0, 1., gamma) # (n_sample, N1)

    tZ = v @ Z.T + a # (n_sample, N2)
    tZ = torch.where(tZ >= 0, 1., gamma) # (n_sample, N2)

    return tX.T @ tZ / n_sample # (N1, N2)


def direction(
    X: Tensor, 
    y: Tensor, 
    gamma: float, 
    n_sample: int = 1000, 
    seed: int = 0,
) -> Tensor:
    '''
    Args:
        X: (N, d) := (the number of data, dim)
        y: (N,) := (the number of data)
    
    Returns:
        direction: (N, d)
    '''

    assert all_elements_in_targets(y, [-1, 1])

    p = Phi(X, X, gamma, n_sample, seed) # (N, N)
    t = (y[None] * p) @ X # (N, d)
    return F.normalize(t) # (N, d)


def hat_f(
    X: Tensor, 
    y: Tensor, 
    Z: Tensor, 
    gamma: float, 
    n_sample: int = 1000, 
    seed: int = 0,
) -> Tensor:
    '''
    Args:
        X: (N1, d) := (the number of data, dim)
        y: (N1,) := (the number of data)
        Z: (N2, d) := (the number of data, dim)

    Returns:
        hat_f: (N2,)
    '''

    assert all_elements_in_targets(y, [-1, 1])

    XZ = X @ Z.T # (N1, N2)
    Phi_ = Phi(X, Z, gamma, n_sample, seed) # (N1, N2)
    t = y[:, None] * Phi_ * XZ # (N1, N2)
    return t.mean(0) # (N2,)


def hat_g(
    X: Tensor, 
    y: Tensor, 
    Z: Tensor, 
    R: Tensor, 
    gamma: float, 
    n_sample: int = 1000, 
    seed: int = 0,
) -> Tensor:
    '''
    Args:
        X: (N1, d) := (the number of data, dim)
        y: (N1,) := (the number of data)
        Z: (N2, d) := (the number of data, dim)
        R: (N1, d) := (the number of data, dim)

    Returns:
        hat_f: (N2,)
    '''

    assert all_elements_in_targets(y, [-1, 1])

    XZ = X @ Z.T # (N1, N2)
    Phi_XX = Phi(X, X, gamma, n_sample, seed) # (N1, N1)
    Phi_RZ = Phi(R, Z, gamma, n_sample, seed) # (N1, N2)
    t = (y[None, :] * Phi_XX) @ XZ / XZ.shape[0] # (N1, N2)
    t = Phi_RZ * t # (N1, N2)
    return t.mean(0) # (N2,)