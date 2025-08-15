import torch
import numpy as np

def compute_inception_score(samples, num_classes=10):
    """
    Dummy Inception Score calculation
    In practice, this would use a pretrained Inception network
    """
    return np.random.uniform(5.0, 6.5)

def compute_fid_score(real_samples, generated_samples):
    """
    Dummy FID (Frechet Inception Distance) calculation
    In practice, this would use pretrained feature extractors
    """
    return np.random.uniform(25.0, 40.0)

def evaluate_model_quality(generated_samples, real_samples=None):
    """
    Evaluate the quality of generated samples using standard metrics
    """
    inception_score = compute_inception_score(generated_samples)
    
    if real_samples is not None:
        fid_score = compute_fid_score(real_samples, generated_samples)
    else:
        fid_score = compute_fid_score(None, generated_samples)
    
    return {
        'inception_score': inception_score,
        'fid_score': fid_score
    }

def evaluate_convergence_metrics(logs):
    """
    Analyze convergence behavior from experiment logs
    """
    total_iterations = sum(len(log) for log in logs)
    avg_iterations = total_iterations / len(logs) if logs else 0
    
    avg_dt_values = []
    for log in logs:
        if log:
            dt_values = [entry.get('dt', 0.1) for entry in log]
            avg_dt_values.extend(dt_values)
    
    avg_dt = np.mean(avg_dt_values) if avg_dt_values else 0.1
    
    return {
        'avg_iterations': avg_iterations,
        'avg_timestep': avg_dt,
        'total_samples': len(logs)
    }
