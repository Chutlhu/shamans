"""
This file perform sound source localization using neural steerer
and other interpolation methods.

Author: Diego DI CARLO
Created: 2025-02-19
Modified: 2025-02-19
"""

import json
import argparse
import datetime
from pathlib import Path
import pickle
from tqdm import tqdm

from shamans.localizers import alpha_stable, inv_wishart, wishart, music, srp_phat

import pandas as pd
import numpy as np
from einops import rearrange
import itertools
import scipy.signal
from scipy.stats import levy_stable

import librosa
import soundfile as sf
import pyroomacoustics as pra

import seaborn as sns
import matplotlib.pyplot as plt

from pprint import pprint
import logging
import parse

# # Set up logging
# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def set_logging_level(level):
    logger.setLevel(level)


do_plot = False
do_output_wav = False

expertiment_folder = Path("./")
path_to_speech_data = expertiment_folder / "data/SmallTimit"
path_to_resolved_models = expertiment_folder / "data/models" 
    

results_dir = expertiment_folder / "results/"
results_dir.mkdir(parents=True, exist_ok=True)
figure_dir = expertiment_folder / "results/figures"
figure_dir.mkdir(parents=True, exist_ok=True)
output_dir = expertiment_folder / "results/output"
output_dir.mkdir(parents=True, exist_ok=True)

ang_spec_methods = {
    'alpha-2.0_beta-2_eps-1E-3_iter-500': lambda X, svect : alpha_stable(X, svect, alpha=2.0, beta=2.0, eps=1e-3, n_iter=500),
    'alpha-1.2_beta-2_eps-1E-3_iter-500': lambda X, svect : alpha_stable(X, svect, alpha=1.2, beta=2.0, eps=1e-3, n_iter=500),
    'alpha-1.2_beta-2_eps-1E-5_iter-500': lambda X, svect : alpha_stable(X, svect, alpha=1.2, beta=2.0, eps=1e-5, n_iter=500),
    'alpha-1.2_beta-1_eps-1E-3_iter-500': lambda X, svect : alpha_stable(X, svect, alpha=1.2, beta=1.0, eps=1e-3, n_iter=500),
    'alpha-1.2_beta-0_eps-1E-3_iter-500': lambda X, svect : alpha_stable(X, svect, alpha=1.2, beta=0.0, eps=1e-3, n_iter=500),
    'alpha-2.0_beta-1_eps-1E-3_iter-500': lambda X, svect : alpha_stable(X, svect, alpha=2.0, beta=1.0, eps=1e-3, n_iter=500),
    'alpha-1.6_beta-1_eps-1E-3_iter-500': lambda X, svect : alpha_stable(X, svect, alpha=1.6, beta=1.0, eps=1e-3, n_iter=500),
    'alpha-0.8_beta-1_eps-1E-3_iter-500': lambda X, svect : alpha_stable(X, svect, alpha=0.8, beta=1.0, eps=1e-3, n_iter=500),
    'alpha-0.4_beta-1_eps-1E-3_iter-500': lambda X, svect : alpha_stable(X, svect, alpha=0.4, beta=1.0, eps=1e-3, n_iter=500),
    'alpha-0.1_beta-1_eps-1E-3_iter-500': lambda X, svect : alpha_stable(X, svect, alpha=0.1, beta=1.0, eps=1e-3, n_iter=500),
    'music_s-1': lambda X, svect : music(X, svect, n_sources=1),
    'music_s-2': lambda X, svect : music(X, svect, n_sources=2),
    'music_s-3': lambda X, svect : music(X, svect, n_sources=3),
    'music_s-4': lambda X, svect : music(X, svect, n_sources=4),
    'music_s-5': lambda X, svect : music(X, svect, n_sources=5),
    'music_s-6': lambda X, svect : music(X, svect, n_sources=6),
    'srp_phat': srp_phat,
}

eusipco_names_to_exp_name = {
    "nf": "nf-subfreq",
    "nf-gw": "nf-subfreq-gw",
    "nn": "nn",
    "sp": "sp",
    "sh": "sh",
    "pinn": "pinn",
    "gp-steerer": "gpdkl-sph-shlow",
}


sv_nObs_choice = [8, 16, 32, 64, 128]
sv_seed_choice = [13, 42, 666]


parser = argparse.ArgumentParser(description='Sound Source Localization with NSteerer and AlphaStable')
parser.add_argument('--sv-method', type=str, default='nf-subfreq', help='Steering vector interpolation method', choices=['nf', 'nf-gw' 'nn', 'sp', 'pinn', 'gp-steerer'])
parser.add_argument('--nObs', type=int, default=8, help='Number of observations used to fit the sv model', choices=[8, 16, 32, 64, 128])
parser.add_argument('--seed', type=int, default=13, help='Random seed used for training the interpolation methods', choices=[13, 42, 666])
parser.add_argument('--alpha', type=float, default=1.2, help='Alpha parameter for the alpha-stable distribution')
parser.add_argument('--exp-id', type=int, default=None, help='Name of the experiment')


def make_data(src_doas_idx, source_type, sound_duration, SNR, noise_type='awgn', RT60=False, mc_seed=1):
    
    n_sources = len(src_doas_idx)
    
    file_name_data = f"doas-{src_doas_idx}_snr-{SNR}_noise-{noise_type}_reverb-{RT60}_mc-{mc_seed}"
    
    # # Load the ground truth data
    path_to_model = path_to_resolved_models / f"ref_nObs-8_seed-13.pkl"
    resolved_sv_dict = load_resolved_svects(path_to_model) # dict(x, y, svect)
    nfft = int(resolved_sv_dict['nfft'])
    fs = int(resolved_sv_dict['fs'])

    coords = resolved_sv_dict['coords'] # [nFreq x nDoas x nChan x 4]
    svect_ref = resolved_sv_dict['svects'] # [nFreq x nDoas x nChan x 1]

    # Constraints to the azimuthal plane
    dirs = coords[0, :, 0, 1:3].reshape(-1, 2)  # [nDoas x 2]
    idx_el0 = np.where(dirs[:, 0] == np.pi / 2)[0]
    
    doa_space = dirs[idx_el0].squeeze()
    
    coords = coords[:, idx_el0, :, :]
    svect_ref = svect_ref[:, idx_el0, :]
    
    # svect in the time domain
    nFreq, nDoas, nChan = svect_ref.shape
    svect_ref_time = np.fft.irfft(svect_ref, nfft, axis=0)
    
    # add reverberation
    if RT60 >= 0:
        logger.info('Load RIRs')
        # load RIRs pickle file
        path_to_rirs = expertiment_folder / "data/directives_rirs_with_spear_rt60-{}.pkl".format(RT60)
        with open(path_to_rirs, 'rb') as f:
            rirs_dict = pickle.load(f)
        azimuths = rirs_dict['azimuths']
        spat_rirs = rirs_dict['rirs']
        svect_ref_time = rearrange(spat_rirs, 'chan doas time -> time doas chan')
        assert svect_ref_time.shape[2] == nChan, "Mismatch in the number of channels"
        # do something with the rirs and svect_ref_time
    elif RT60 == -1:
        pass
    else:
        raise ValueError(f"Unknown RT60 value {RT60}")
    
    # make mixtures
    
    # load speech signal
    # get the list of all the speech files
    if source_type == "speech":
        speech_files = list(path_to_speech_data.glob("*.wav"))
        # randomly select n_sources speech files
        speech_files = np.random.choice(speech_files, n_sources, replace=False)
        src_signals = []
        for i, speech_file in enumerate(speech_files):
            s, fs = librosa.load(speech_file, sr=fs, duration=sound_duration, mono=True, offset=0.5)
            s = s / np.std(s)
            src_signals.append(s)
    elif "alpha" in source_type:
        src_alpha = float(source_type.split("-")[1])
        size = (int(fs * sound_duration),)
        src_signals = []
        for i in range(n_sources):
            s = levy_stable.rvs(alpha=src_alpha, beta=0, loc=0, size=size)
            s = s / np.std(s)
            src_signals.append(s)
        speech_files = [f'{i}' for i in range(n_sources)]
    elif source_type == "gauss":
        size = (int(fs * sound_duration),)
        src_signals = []
        for i in range(n_sources):
            s = np.random.randn(size)
            s = s / np.std(s)
            src_signals.append(s)
        speech_files = [f'{i}' for i in range(n_sources)]
    src_signals = np.array(src_signals)
    logger.debug("src_signals shape: ", src_signals.shape)
    # check the the selected source are active
    assert np.all(np.std(src_signals, axis=-1) > 0), "Source signals are empty"
    
    # make mixture
    logger.info('Make mixture')
    x = []
    for i in range(nChan):
        xi = []
        for j in range(len(src_signals)):
            s = src_signals[j] / np.std(src_signals[j])
            xi.append(scipy.signal.fftconvolve(svect_ref_time[:,src_doas_idx[j],i], s, mode='full'))
        x.append(np.sum(np.array(xi), axis=0))
    mixture = np.array(x) # [nChan x nSamples]
    mixture = mixture[:,:len(s)] # [nChan x nSamples]
    logger.debug("Mixture shape: ", mixture.shape)

    # add noise
    if noise_type == 'awgn':
        noise = np.random.randn(*mixture.shape)
    elif "alpha" in noise_type:
        noise_alpha = float(noise_type.split("-")[1])
        noise = levy_stable.rvs(alpha=noise_alpha, beta=0, loc=0, size=mixture.shape)
    else:
        raise ValueError(f"Unknown noise type {noise_type}")
    noise = noise / np.linalg.norm(noise) * np.linalg.norm(mixture) / 10**(SNR/20)
    mixture = mixture + noise # [nChan x nSamples]
    logger.debug("Mixture shape: ", mixture.shape)
    
    time_src = np.arange(src_signals.shape[1]) / fs
    time_mix = np.arange(mixture.shape[1]) / fs 
    if do_plot:
        fig, axarr = plt.subplots(n_sources+1, 1, figsize=(10, 5), sharex=True)
        for i, ax in enumerate(axarr):
            if i < n_sources:
                ax.plot(time_src, src_signals[i], label=f's{i+1}')
            else:
                ax.plot(time_mix, mixture[0], label='mixture')
            ax.legend()
        plt.savefig(figure_dir / f'{file_name_data}_mixture.png')
        plt.close()
    
    doas_ref = np.array([doa_space[src_doas_idx[j]] for j in range(n_sources)])
    
    # save mixture as wave with doa in the file name
    if do_output_wav:
        mixture_to_save = mixture / np.max(np.abs(mixture))
        sf.write(output_dir / f'{file_name_data}.wav', mixture_to_save.T, fs)
    
    return mixture, doas_ref, speech_files
    
    
def plot_ang_spec(S:np.array, doas_est_idx:np.array, doas_ref:np.array=None, title=None):
    """
    S in [nDoas x nFreq]
    doas in [nSources]
    """
    
    fig, axarr = plt.subplots(2,1, figsize=(6,3), sharex=True)
    axarr[0].imshow(S.T, aspec='auto', origin='lower')
    if doas_ref is not None:
        for j in range(len(doas_ref)):
            axarr[0].axvline(doas_ref[j], color='r', linestyle='--', label=f'Doa GT {doas_ref[j]}')
    axarr[0].set_ylabel('Freq [Hz]')
    axarr[0].set_yscale('symlog')
    axarr[1].plot(np.mean(S, axis=1), label='Ang spec')
    if doas_ref is not None:
        for j in range(len(doas_ref)):
            axarr[1].axvline(doas_ref[j], color='r', linestyle='--', label=f'Doa GT {doas_ref[j]}')
    for j in range(len(doas_est_idx)):
        axarr[1].scatter(doas_est_idx, np.mean(S, axis=1)[doas_est_idx], color='g', label='Doa est')
    axarr[1].legend()
    axarr[1].set_xlabel('DOAs')
    plt.suptitle(title)
    return


def find_peaks(values, k=1):
    # make circular
    n_points = len(values)
    val_ext = np.append(values, values[:10])

    # run peak finding
    indexes = pra.doa.detect_peaks(val_ext, show=False) % n_points
    candidates = np.unique(indexes)  # get rid of duplicates, if any

    # Select k largest
    peaks = values[candidates]
    max_idx = np.argsort(peaks)[-k:]

    # return the indices of peaks found
    return candidates[max_idx]


def load_resolved_svects(path_to_model):
    with open(path_to_model, 'rb') as f:
        sv_dict = pickle.load(f)
    return sv_dict


def localize(
    mixture: np.ndarray, 
    loc_method: str,
    freq_range: list,
    n_sources: int,
    sv_method: str,
    seed: int,
    nObs: int,
    sv_normalization: bool,
    ):
    
    # Load the ground truth data
    path_to_model = path_to_resolved_models / f"{sv_method}_nObs-{nObs}_seed-{seed}.pkl"
    resolved_sv_dict = load_resolved_svects(path_to_model) # dict(x, y, svect)
    
    coords = resolved_sv_dict['coords'] # [nFreq x nDoas x nChan x 6]
    svects = resolved_sv_dict['svects'] # [nFreq x nDoas x nChan x 1]
    
    # Constraints to the azimuthal plane
    dirs = coords[0, :, 0, 1:3].reshape(-1, 2)  # [nDoas x 2]
    idx_el0 = np.where(dirs[:, 0] == np.pi / 2)[0]
    doa_space = dirs[idx_el0].squeeze()
    
    coords = coords[:, idx_el0, :, :]
    freqs = coords[:, 0, 0, 0]
    svect = svects[:, idx_el0, :]
    nFreq, nDoas, nChan_ = svect.shape
    
    # normalize the svects used for "scanning"
    if sv_normalization:
        svect = svect / np.linalg.norm(svect, axis=-1, keepdims=True)
    
    # Sound source localization 
    nfft = int(resolved_sv_dict['nfft'])
    fs =  int(resolved_sv_dict['fs'])
    X = librosa.stft(mixture, n_fft=nfft, hop_length=nfft//2) # [nChan, nfft/2, nFrames]
    X = X[:,:nFreq,:] # [nChan, nFreq, nFrames]
        
    # Focus on a specific frequency range
    freqs_ = np.fft.fftfreq(nfft, 1/fs)[:nFreq]
    assert np.allclose(freqs, freqs_), "Mismatch in the frequency bins"
    
    idx_freq_range = np.where((freqs >= freq_range[0]) & (freqs <= freq_range[1]))[0]
    X = X[:,idx_freq_range,:] # [nChan, nFreq, nFrames]
    svect = svect[idx_freq_range,:,:]
    nChan, nFreq, nFrames = X.shape
    nFreq_, nDoas, nChan_ = svect.shape
    assert nFreq == nFreq_, "Mismatch in the number of frequency bins"
    assert nChan == nChan_, "Mismatch in the number of channels"
    
    ang_spec = ang_spec_methods[loc_method](X, svect) # [nDoas x nFreq]
    assert ang_spec.shape[0] == (nDoas), f"Expected first shape {(nDoas)}, got {ang_spec.shape[0]}"
    
    # estimate source location, get the n_sources highest peaks
    ang_spec_poll = np.mean(ang_spec, axis=1) # pool over frequencies
    ang_spec_poll = ang_spec_poll / np.max(ang_spec_poll)
    doas_est_idx = find_peaks(ang_spec_poll, k=n_sources)
    if len(doas_est_idx) < n_sources:
        # it means that find peaks f
        # just take the the peak with a simple argmax
        doas_est_idx = np.argsort(-ang_spec_poll)[:n_sources]
    logger.debug("Estimated DOAs: ", doas_est_idx)
    doas_est = [doa_space[doas_est_idx[i]] for i in range(n_sources)]
    ang_spec_freqs = freqs[idx_freq_range]
    
    return doas_est, doas_est_idx, ang_spec, ang_spec_freqs


def convert_to_azimuth_inclination(doas_incl_az):
    """
    Convert DOAs from (elevation, azimuth) to (azimuth, inclination).
    """
    assert doas_incl_az.shape[-1] == 2, "Output should have shape (N, 2) for azimuth and inclination"
    
    assert np.max(np.abs(doas_incl_az[...,0])) <= np.pi,     "Eleveation should be in the range [0, pi]"
    assert np.min(np.abs(doas_incl_az[...,0])) >= 0,         "Eleveation should be in the range [0, pi]"
    assert np.max(np.abs(doas_incl_az[...,1])) <= 2 * np.pi, "Azimuth should be in the range [-2pi, 2pi]"
    assert np.min(np.abs(doas_incl_az[...,1])) >= 0,         "Azimuth should be in the range [-2pi, 2pi]"
    
    doas_az_elev = np.stack([doas_incl_az[..., 1], np.pi / 2 - doas_incl_az[..., 0]], axis=-1)
    
    assert np.max(np.abs(doas_az_elev[...,0])) <= 2 * np.pi,  "Azimuth should be in the range [-2pi, 2pi]"
    assert np.min(np.abs(doas_az_elev[...,0])) >= 0,          "Azimuth should be in the range [0, 2pi]"
    assert np.max(np.abs(doas_az_elev[...,1])) <= np.pi / 2,  "Inclination should be in the range [-pi, pi]"
    assert np.min(np.abs(doas_az_elev[...,1])) >= -np.pi / 2, "Inclination should be in the range [-pi, pi]"

    # Test that the output is azimuth, inclination
    assert doas_az_elev.shape == doas_incl_az.shape, "Output should have the same shape as the input"
    
    return doas_az_elev


def compute_angle_between(doas_est, doas_ref):
    
    # change convention  
    # 1. (elevation, azimuth) -> (azimuth, elevation)    

    # Convert estimated and reference DOAs
    doas_est = convert_to_azimuth_inclination(doas_est)
    doas_ref = convert_to_azimuth_inclination(doas_ref)

    assert len(doas_est) == len(doas_ref), "Mismatch in the number of DOAs"

    # ref and x are now both in spherical angles (radian) and match in size
    # data is in the form [azimuth, elevation]
    # use Haversine formula to get angle between
    dlon = doas_ref[:, 0] - doas_est[:, 0]  # azimuth differences
    dlat = doas_ref[:, 1] - doas_est[:, 1]  # elevation differences
    a = np.sin(dlat / 2) ** 2 + np.cos(doas_est[:, 1]) * np.cos(doas_ref[:, 1]) * np.sin(dlon / 2) ** 2
    a = 2 * np.arcsin(np.sqrt(a))
    return a


def compute_metrics(doas_est, doas_ref):
    doas_est = np.array(doas_est)
    doas_ref = np.array(doas_ref)
    # compute the angular error between the estimated and the ground truth DOAs
    nDoas = len(doas_est)
    # find the best order of the estimated DOAs to match the ground truth DOAs
    if nDoas == 1:
        best_error = compute_angle_between(doas_est, doas_ref)
        best_perm = [0]
        best_doas_est = doas_est
    else:
        best_perm = None
        best_error = None
        min_error = np.inf
        for perm in itertools.permutations(range(nDoas)):
            error = compute_angle_between(doas_est[list(perm)], doas_ref)
            logger.debug("Permutation: ", perm, "Error: ", error.sum())
            if error.sum() < min_error:
                min_error = error.sum()
                best_error = error
                best_perm = perm
                best_doas_est = doas_est[list(perm)]
    return best_error, best_doas_est, best_perm


def process_experiment(
    src_doas_idx, source_type, sound_duration, 
    snr, noise_type, rt60, 
    loc_method, freq_range, sv_method, seed, nObs, sv_normalization, 
    mc_seed=1, exp_name=None
    ):
    
    # set seed for reproducibility
    np.random.seed(mc_seed)
    
    mixture, doas_ref, speech_files = make_data(src_doas_idx, source_type, sound_duration, snr, noise_type, rt60, mc_seed=mc_seed)
    n_sources = len(src_doas_idx)
    
    doas_est, doas_est_idx, ang_spec, ang_spec_freqs = localize(mixture, loc_method, freq_range, n_sources, sv_method, seed, nObs, sv_normalization)
    
    if do_plot:
        plot_ang_spec(ang_spec, doas_est_idx, title='Estimated Ang Spec')
        plt.savefig(figure_dir / f'{exp_name}_ang_spec_est.png')
        plt.close()
    
    logger.debug("Ground truth DOAs: ", doas_ref)
    error, doas_est, perm = compute_metrics(doas_est, doas_ref)
    doas_est_idx = [doas_est_idx[i] for i in perm]
    logger.debug("Best permutation: ", doas_est)
    logger.debug("Estimated DOAs: ", doas_est)
    logger.debug("Error: ", error)
    
    # if not list, make it a list
    doas_ref = np.array(doas_ref).tolist()
    doas_est = np.array(doas_est).tolist()
    doas_est_idx = np.array(doas_est_idx).tolist()
    doas_ref_idx = np.array(src_doas_idx).tolist()
    error = np.array(error).tolist()
    
    logger.info(f"Estimated DOAs: {doas_est}")
    logger.info(f"Ground truth DOAs: {doas_ref}")
    logger.info(f"Error: {error}")
    
    return doas_est, doas_est_idx, error, doas_ref, doas_ref_idx, ang_spec, ang_spec_freqs, speech_files


if __name__ == "__main__":
    args = parser.parse_args()

    src_doas = [5, 40]
    source_type = 'speech'
    sound_duration = 0.5
    snr = -5
    noise_type = 'awgn'
    rt60 = 0.273
    
    loc_method = 'music_s-1'
    freq_range = [200, 2000]
    
    sv_method = 'gp-steerer'
    nObs = 8
    seed = 13
    sv_normalization = True

    # Create a string for the file name
    process_experiment(
        src_doas, source_type, sound_duration,
        snr, noise_type, rt60,
        loc_method, freq_range,
        sv_method, seed, nObs, sv_normalization,
    )