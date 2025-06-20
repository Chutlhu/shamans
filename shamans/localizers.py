import numpy as np
import scipy
from einops import rearrange
import logging
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)

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
    ang_spec_NF = np.real(np.einsum('fji,fiI,fjI->jf', a.conj(), SCM, a))
    return ang_spec_NF


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
    arg_FJT = 1j * np.real(np.einsum('fji,ift->fjt', np.conj(svect), X)) / (2 ** (1 / alpha))
    arg_FJ = np.mean(np.exp(arg_FJT), axis=-1)
    phi_FJ = np.abs(arg_FJ)**2
    return phi_FJ

def compute_beta_div(SM, ind_fun, Psi, beta):
    X_ = Psi @ SM
    Gplus = Psi.T @ X_**(beta - 1)
    Gmin = Psi.T @ (X_**(beta - 2) * ind_fun)
    return (Gplus - Gmin).sum()

def alpha_stable(X_IFT, svects, alpha=1.2, beta=0.0, eps=1e-3, n_iter=1000, plot_cost=False):

    n_warmup = 20

    nChan, nFreq, nTime = X_IFT.shape
    a = svects
    nFreq, nDoas, nChan = a.shape

    X_norm = X_IFT / np.linalg.norm(X_IFT, ord=1, axis=0, keepdims=True) # Normalize X
#
    phi = LevyExp(a, X_norm, alpha=alpha)
    ind_func = - np.log(phi) # [nFreq x nDoas]
    ind_func = rearrange(ind_func, 'f j -> (f j) 1')
    Psi = np.abs(np.einsum('fji,fJi->fjJ', a.conj(), a))**alpha # [nFreq x nDoas x nDoas]
    Psi = rearrange(Psi, 'f j J -> (f j) J')
    
    SM = srp_phat(X_IFT, a).mean(axis=1)[:, None]
    SM = SM / np.max(SM)  # Normalize SM
    assert SM.shape == (nDoas, 1), f"SM shape mismatch: {SM.shape} != {(nDoas, 1)}"

    
    # plt.figure(figsize=(8, 6))
    # plt.plot(SM.copy(), label=f'Iter {0}')
    # plt.xlabel('DOA index')
    # plt.ylabel('SM value')
    # plt.legend()
    # plt.grid()
    # plt.savefig(f'./tmp/alpha_stable_iter_{0}.png')
    # plt.close()

    cost = []
    SM_list = []
    for n in range(n_iter):
        Psi_dot_SM = Psi @ SM
        num = Psi.T @ (Psi_dot_SM**(beta - 2) * ind_func)
        den = eps + Psi.T @ (Psi_dot_SM**(beta - 1))
        SM *= num / den
        if n > n_warmup:
            SM_list.append(SM.copy())
            cost.append(compute_beta_div(SM, ind_func, Psi, beta))
        # if n % 25 == 0:
        #     plt.figure(figsize=(8, 6))
        #     plt.plot(SM.copy(), label=f'Iter {n+1}')
        #     plt.xlabel('DOA index')
        #     plt.ylabel('SM value')
        #     plt.legend()
        #     plt.grid()
        #     plt.savefig(f'./tmp/alpha_stable_iter_{n+1}.png')
        #     plt.close()


    best_cost_idx = np.argmin(cost)
    SM = SM_list[best_cost_idx]
    # SM = SM_list[-1]
    # best_cost_idx = len(SM_list) - 1
    
    logger.info(f'Alpha-stable method: alpha={alpha}, beta={beta}, eps={eps}, n_iter={n_iter}')
    logger.info(f'--- best cost={cost[best_cost_idx]:.2f} at iteration {best_cost_idx}')

    # plt.plot(cost)
    # plt.xlabel('Iteration')
    # plt.ylabel('Cost')
    # plt.title(f'Alpha-stable method: alpha={alpha}, beta={beta}, eps={eps}, n_iter={n_iter}')
    # plt.grid()
    # plt.savefig('./tmp/alpha_stable_cost.png')
    # plt.close()

    ang_spec = SM

    return ang_spec

methods = {
    'alpha_stable' : alpha_stable,
    'music' : music,
    'srp_phat': srp_phat,
    'wishart' : wishart,
    'inv_wishart' : inv_wishart,
}

