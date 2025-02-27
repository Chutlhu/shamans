import numpy as np
import scipy
from einops import rearrange

tol = 1e-14

def noise_subspace_decomposition(SCM, n_sources=None):
    nFreq, nChan, _ = SCM.shape
    eigval, eigvec = np.linalg.eigh(SCM)

    # This method (numpy.linalg.eigh) returns the eigenvalues (and
    # eigenvectors) in ascending order, so there is no need to sort Signal
    # comprises the leading eigenvalues Noise takes the rest

    eigval_s = eigval[..., -n_sources :]
    eigvec_s = eigvec[..., -n_sources :]
    eigval_n = eigval[..., : -n_sources]
    eigvec_n = eigvec[..., : -n_sources]
    
    identity = np.zeros_like(SCM)
    identity[:, list(np.arange(nChan)), list(np.arange(nChan))] = 1
    
    UUh = identity - np.einsum('fik,fjk->fij', eigvec_s, eigvec_s.conj())

    # UUh = np.einsum('fik,fjk->fij', eigvec_n, eigvec_n.conj())
    return UUh
    

def srp(X, svects):
    Fs, Js, Is = svects.shape
    Ix, Fx, Tx = X.shape
    assert Is == Ix, "Mismatch in the number of microphones"
    assert Fs == Fx, "Mismatch in the number of frequency bins"
    # compute bf weights
    w = np.conj(svects) # [F x J x I]
    # apply beamforming
    y = np.einsum('fji,ift->fj', w, X) / Tx
    # compute bf power
    ang_spec = np.real(y)
    return ang_spec

def srp_phat(X, svects, n_sources=None):
    Fs, Js, Is = svects.shape
    Ix, Fx, Tx = X.shape
    assert Is == Ix, "Mismatch in the number of microphones"
    assert Fs == Fx, "Mismatch in the number of frequency bins"
    a = svects
    absX = np.abs(X)
    absX[absX < tol] = tol
    X = X / np.abs(X)
    SCM = np.einsum('ift,Ift->fiI', X, np.conj(X)) / Tx
    SCM = SCM / np.trace(SCM, axis1=1, axis2=2)[:,None,None]
    ang_spec = np.real(np.einsum('fji,fiI,fjI->jf', a.conj(), SCM, a)) / Tx
    return ang_spec

def inv_wishart(X, svects, n_sources=1):
    Fs, Js, Is = svects.shape
    Ix, Fx, Tx = X.shape
    assert Is == Ix, "Mismatch in the number of microphones"
    assert Fs == Fx, "Mismatch in the number of frequency bins"
    a = svects
    SCM = np.einsum('ift,Ift->fiI', X, np.conj(X)) / Tx
    SCM = SCM / np.trace(SCM, axis1=1, axis2=2)[:,None,None]
    if n_sources is not None:
        eigval, eigvec = np.linalg.eigh(SCM)
        eigvec = eigvec[..., -n_sources:]
        SCM = np.einsum('fik,fjk->fij', eigvec, eigvec.conj()) + 1e-2 * np.eye(Ix)
    invSCM = np.linalg.inv(SCM)
    Psi = np.einsum('fji,fjI->fjiI', a.conj(), a) + 1e-6 * np.eye(Ix)
    ang_spec = 1 / np.einsum('fjiI,fiI->jf', Psi, invSCM).real
    return ang_spec

def wishart(X, svects, n_sources=None):
    Fs, Js, Is = svects.shape
    Ix, Fx, Tx = X.shape
    assert Is == Ix, "Mismatch in the number of microphones"
    assert Fs == Fx, "Mismatch in the number of frequency bins"
    a = svects
    SCM = np.einsum('ift,Ift->fiI', X, np.conj(X)) / Tx
    SCM = SCM / np.trace(SCM, axis1=1, axis2=2)[:,None,None]
    if n_sources is not None:
        eigva, eigve = np.linalg.eigh(SCM)
        eigve_s = eigve[..., -n_sources:]
        SCM = np.einsum('fik,fjk->fij', eigve_s, eigve_s.conj())
    ang_spec = np.abs(np.einsum('fji,fiI,fjI->jf', a.conj(), SCM, a))
    return ang_spec

def music(X, svects, n_sources=1):
    Fs, Js, Is = svects.shape
    Ix, Fx, Tx = X.shape
    assert Is == Ix, "Mismatch in the number of microphones"
    assert Fs == Fx, "Mismatch in the number of frequency bins"
    a = svects
    Sigma = np.einsum('ift,Ift->fiI', X, np.conj(X)) / Tx
    UUh = noise_subspace_decomposition(Sigma, n_sources)
    ang_spec = 1 / np.abs(np.einsum('fji,fiI,fjI->jf', a.conj(), UUh, a))
    return ang_spec

############################################
# Alpha-stable distribution based methods #
############################################

def LevyExp(svect, X, alpha=1.2):
    Fs, Js, Is = svect.shape
    Ix, Fx, Tx = X.shape
    assert Is == Ix, "Mismatch in the number of microphones"
    assert Fs == Fx, "Mismatch in the number of frequency bins"
    arg = np.mean(np.exp(1j * np.real(np.einsum('fji,ift->fjt', np.conj(svect), X)) / (2 ** (1 / alpha))), axis=-1)
    phi = np.abs(arg)**2
    return phi

def compute_beta_div(SM, ind_fun, Psi, beta):
    X_ = Psi @ SM
    Gplus = Psi.T @ X_**(beta - 1)
    Gmin = Psi.T @ (X_**(beta - 2) * ind_fun)
    return (Gplus - Gmin).sum()

def alpha_stable(X, svects, alpha=1.2, beta=0.0, eps=1e-3, n_iter=1000):

    nChan, nFreq, nTime = X.shape
    a = svects
    nFreq, nDoas, nChan = a.shape
    X = X / np.abs(X) # PHAT normalization
    
    phi = LevyExp(a, X, alpha = alpha)
    ind_func = - np.log(phi) # [nFreq x nDoas]
    ind_func = rearrange(ind_func, 'f j -> (f j) 1')
    Psi = np.abs(np.einsum('fji,fJi->fjJ', a.conj(), a))**alpha # [nFreq x nDoas x nDoas]
    Psi = rearrange(Psi, 'f j J -> (f j) J')
    
    SM = np.ones([nDoas, 1])
    
    for n in range(n_iter):
        foo = Psi @ SM
        num = Psi.T @ (foo**(beta - 2) * ind_func)
        den = eps + Psi.T @ (foo**(beta - 1))
        SM *= num / den
    
    ang_spec = SM
    return ang_spec

methods = {
    'alpha_stable' : alpha_stable,
    'music' : music,
    'srp_phat': srp_phat,
    # 'srp': srp,
    'wishart' : wishart,
    'inv_wishart' : inv_wishart,
}

