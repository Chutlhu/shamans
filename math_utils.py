import numpy as np
from scipy.stats import levy_stable

def generate_emvas_noise(n_samples, dim, alpha, epsilon):
    """
    Generate elliptically multivariate alpha-stable (EMVAS) noise with a diagonal covariance matrix.

    Parameters:
    - n_samples: Number of samples
    - dim: Dimension of the noise vector
    - alpha: Stability parameter (0 < alpha ≤ 2)
    - epsilon: Coefficient for the diagonal covariance matrix

    Returns:
    - emvas_noise: (n_samples, dim) array of EMVAS noise
    """
    # Step 1: Generate Gaussian samples with diagonal covariance
    cov_matrix = epsilon * np.eye(dim)  # Diagonal covariance matrix
    gaussian_samples = np.random.multivariate_normal(mean=np.zeros(dim), cov=cov_matrix, size=n_samples)
    
    # Step 2: Normalize to unit sphere
    norms = np.linalg.norm(gaussian_samples, axis=1, keepdims=True)
    normalized_gaussian = gaussian_samples / norms
    
    # Step 3: Generate independent alpha-stable noise
    stable_noise = levy_stable.rvs(alpha, beta=1, loc=0, scale=np.cos(np.pi * alpha / 4.)**( 2./alpha), size=n_samples)[:, np.newaxis]
    
    # Step 4: Scale the normalized Gaussian vectors
    emvas_noise = normalized_gaussian * stable_noise
    
    return emvas_noise


def test_generate_emvas_noise():
    # Example usage
    n_samples = 1000  # Number of samples
    dim = 2  # Dimension of noise
    alpha = 1.5  # Stability parameter (0 < alpha ≤ 2)
    epsilon = 0.1  # Coefficient for diagonal covariance

    noise_samples = generate_emvas_noise(n_samples, dim, alpha, epsilon)
    print(noise_samples[:5])  # Print first 5 samples



if __name__ == "__main__":
    test_generate_emvas_noise()