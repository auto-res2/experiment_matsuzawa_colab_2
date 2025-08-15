import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

def compute_error(sample, noise_estimate, cond_grad):
    """
    Dummy error calculator: combines norm values of sample, noise_estimate, and cond_grad
    Returns a tensor of shape (batch_size, 4), representing 4 error metrics.
    """
    err1 = torch.norm(sample, p=2, dim=-1, keepdim=True)
    err2 = torch.norm(noise_estimate, p=2, dim=-1, keepdim=True)
    err3 = torch.norm(cond_grad, p=2, dim=-1, keepdim=True)
    err4 = torch.rand_like(err1) * 0.1
    return torch.cat([err1, err2, err3, err4], dim=-1)

def update_sample(sample, dt, noise_estimate, cond_grad):
    """
    Dummy update rule to simulate one step of a diffusion update.
    """
    return sample - dt * (noise_estimate + cond_grad)

def is_converged(sample, threshold=0.05, max_iter=20, current_iter=0):
    """
    Dummy convergence check: if current iteration exceeds max or sample norm is less than threshold
    """
    norm = torch.norm(sample)
    if current_iter >= max_iter or norm < threshold:
        return True
    return False

def should_realign(error_metrics, threshold=1.0):
    """
    Checks if the average error metric (from the first three components) exceeds a threshold.
    """
    avg_error = error_metrics[:, :3].mean().item()
    return avg_error > threshold

def recalc_guidance(current_state):
    """
    Dummy conditional guidance recalculation
    """
    return current_state * 0.95

def cached_guidance(current_state):
    """
    Dummy cached guidance: return the current state as guidance.
    """
    return current_state

class DummyDiffusionModel(nn.Module):
    def __init__(self):
        super(DummyDiffusionModel, self).__init__()
        pass

    def forward(self, sample, t):
        """
        Simulate the diffusion step: given sample and current timestep t,
        return a dummy noise_estimate and conditional gradient.
        """
        noise_estimate = torch.randn_like(sample) * 0.1
        cond_grad = torch.randn_like(sample) * 0.05
        return noise_estimate, cond_grad
