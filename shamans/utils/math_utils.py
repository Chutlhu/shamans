import numpy as np
import math
from scipy.stats import levy_stable


def cart2sph(x, y, z):
    # From wikipedia, adapted from Easycom dataset code
    # DESCRIPTION: converts cartesian to spherical coordinate
    # according to the convention of wikipedia
    # *** INPUTS ***
    # x  (ndarray) x-coordinate(s) [N x 1]
    # y  (ndarray) y-coordinate(s) [N x 1]
    # z  (ndarray) z-coordinate(s) [N x 1]
    # *** OUTPUTS ***
    # r     (ndarray) range(s) in meter [N x 1]
    # theta (ndarray) inclination(s) in radians [N x 1]
    # phi   (ndarray) azimuth(s) in radians [N x 1]

    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    theta = np.arctan2(hxy, z)
    phi = np.arctan2(y, x)
    phi = np.mod(phi, 2 * np.pi)
    return r, theta, phi

def sph2cart(r, theta, phi, xp=np):
    # From wikipedia, adapted from Easycom dataset code
    # DESCRIPTION: converts spherical to cartesian coordinate
    # according to the convention of wikipedia

    # *** INPUTS ***
    # r     (ndarray) range(s) in meter [N x 1]
    # theta (ndarray) inclination(s) in radians [N x 1] \in [0, pi]
    # phi   (ndarray) azimuth(s) in radians [N x 1] \in [0, 2pi]
    # *** OUTPUTS ***
    # x  (ndarray) x-coordinate(s) [N x 1]
    # y  (ndarray) y-coordinate(s) [N x 1]
    # z  (ndarray) z-coordinate(s) [N x 1]

    rsin_theta = r * xp.sin(theta)
    x = rsin_theta * xp.cos(phi)
    y = rsin_theta * xp.sin(phi)
    z = r * xp.cos(theta)
    return x, y, z

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


def estimate_levy_exponent(X_FTM):
    """
    This is the function to estimate alpha for a FTM tensor (F: number of frequency, T: number of time frame, M: number of microphone)
    """
    
    def factor_int(n):
        val = math.ceil(math.sqrt(n) / 2)
        val2 = 2 * int(n/val)
        while val2 * val != float(n):
            val -= 1
            val2 = int(n/val)
        return val, val2, n

    def shuffle_along_axis(a, axis):
        idx = np.random.rand(*a.shape).argsort(axis=axis)
        return np.take_along_axis(a, idx, axis=axis)

    nFreq, nTime, nChan = X_FTM.shape

    K = nFreq * nTime
    K1, K2, _ = factor_int(K)
    rnd_X_FTM = shuffle_along_axis(X_FTM, axis=0)
    rnd_X_FTxM = np.array(shuffle_along_axis(rnd_X_FTM, axis=1)).reshape(-1, rnd_X_FTM.shape[-1])
    Y_K2M = np.zeros((K2, nChan)).astype(np.complex64)
    for k2 in range(K2):
        Y_K2M[k2] = rnd_X_FTxM[k2*K1:  K1 + k2*K1, :].sum(axis=0)
    logXnorm_FT = np.log(np.linalg.norm(rnd_X_FTxM, axis=-1))
    logYnorm_FT = np.log(np.linalg.norm(Y_K2M, axis=-1))
    alpha = (1 / np.log(K1) *
            (1/K2 * logYnorm_FT.sum() -
            1/K * logXnorm_FT.sum())) ** (-1)
    # print("alpha value for {}th source :{}".format(n +1, self.alphas[n]))
    return alpha


if __name__ == "__main__":
    test_generate_emvas_noise()