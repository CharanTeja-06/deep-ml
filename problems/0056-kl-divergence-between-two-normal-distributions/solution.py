import math

def kl_divergence_normal(mu_p: float, sigma_p: float, mu_q: float, sigma_q: float) -> float:
    # Compute the KL divergence for two normal distributions
    term1 = math.log(sigma_q / sigma_p)
    term2 = (sigma_p**2 + (mu_p - mu_q)**2) / (2 * sigma_q**2)
    term3 = 0.5
    
    return float(term1 + term2 - term3)
