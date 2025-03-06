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

from localizers import alpha_stable, inv_wishart, wishart, music, srp_phat

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
path_to_resolved_models = expertiment_folder / "data/selected_models" 


results_dir = expertiment_folder / "results/"
results_dir.mkdir(parents=True, exist_ok=True)
figure_dir = expertiment_folder / "results/figures"
figure_dir.mkdir(parents=True, exist_ok=True)
output_dir = expertiment_folder / "results/output"
output_dir.mkdir(parents=True, exist_ok=True)


eusipco_names_to_exp_name = {
    "nf": "nf-subfreq",
    "nf-gw": "nf-subfreq-gw",
    "nn": "nn",
    "sp": "sp",
    "sh": "sh",
    "pinn": "pinn",
    "gp-steerer": "gpdkl-sph-shlow",
}


ang_spec_methods = {
    'alpha-2.0_beta-2_eps-1E-3_iter-500': lambda X, svect : alpha_stable(X, svect, alpha=2.0, beta=2.0, eps=1e-3, n_iter=500),
    'alpha-1.2_beta-2_eps-1E-3_iter-500': lambda X, svect : alpha_stable(X, svect, alpha=1.2, beta=2.0, eps=1e-3, n_iter=500),
    'alpha-0.8_beta-2_eps-1E-3_iter-500': lambda X, svect : alpha_stable(X, svect, alpha=0.8, beta=2.0, eps=1e-3, n_iter=500),
    'alpha-2.0_beta-1_eps-1E-3_iter-500': lambda X, svect : alpha_stable(X, svect, alpha=2.0, beta=1.0, eps=1e-3, n_iter=500),
    'alpha-1.2_beta-1_eps-1E-3_iter-500': lambda X, svect : alpha_stable(X, svect, alpha=1.2, beta=1.0, eps=1e-3, n_iter=500),
    'alpha-0.8_beta-1_eps-1E-3_iter-500': lambda X, svect : alpha_stable(X, svect, alpha=0.8, beta=1.0, eps=1e-3, n_iter=500),
    'alpha-2.0_beta-2_eps-1E-5_iter-500': lambda X, svect : alpha_stable(X, svect, alpha=2.0, beta=2.0, eps=1e-5, n_iter=500),
    'alpha-1.2_beta-2_eps-1E-5_iter-500': lambda X, svect : alpha_stable(X, svect, alpha=1.2, beta=2.0, eps=1e-5, n_iter=500),
    'alpha-0.8_beta-2_eps-1E-5_iter-500': lambda X, svect : alpha_stable(X, svect, alpha=0.8, beta=2.0, eps=1e-5, n_iter=500),
    'alpha-2.0_beta-1_eps-1E-5_iter-500': lambda X, svect : alpha_stable(X, svect, alpha=2.0, beta=1.0, eps=1e-5, n_iter=500),
    'alpha-1.2_beta-1_eps-1E-5_iter-500': lambda X, svect : alpha_stable(X, svect, alpha=1.2, beta=1.0, eps=1e-5, n_iter=500),
    'alpha-0.8_beta-1_eps-1E-5_iter-500': lambda X, svect : alpha_stable(X, svect, alpha=0.8, beta=1.0, eps=1e-5, n_iter=500),
    'music': music,
    'srp_phat': srp_phat,
    'wishart': wishart,
    'inv_wishart': inv_wishart,
}
ang_spec_methods_choices = ang_spec_methods.keys()
sv_methods_choices = eusipco_names_to_exp_name.keys()
sv_nObs_choice = [8, 16, 32, 64, 128]
sv_seed_choice = [13, 42, 666]


parser = argparse.ArgumentParser(description='Sound Source Localization with NSteerer and AlphaStable')
parser.add_argument('--sv-method', type=str, default='nf-subfreq', help='Steering vector interpolation method', choices=['nf', 'nf-gw' 'nn', 'sp', 'pinn', 'gp-steerer'])
parser.add_argument('--nObs', type=int, default=8, help='Number of observations used to fit the sv model', choices=[8, 16, 32, 64, 128])
parser.add_argument('--seed', type=int, default=13, help='Random seed used for training the interpolation methods', choices=[13, 42, 666])
parser.add_argument('--alpha', type=float, default=1.2, help='Alpha parameter for the alpha-stable distribution')
parser.add_argument('--exp-id', type=int, default=None, help='Name of the experiment')

def make_data(src_doas_idx, source_type, sound_duration, SNR, noise_type='awgn', add_reverberation=False, mc_seed=1):
    
    n_sources = len(src_doas_idx)
    
    file_name_data = f"doas-{src_doas_idx}_snr-{snr}_noise-{noise_type}_reverb-{add_reverberation}_mc-{mc_seed}"
    
    # # Load the ground truth data
    path_to_model = path_to_resolved_models / f"ref_nObs-8_seed-13.pkl"
    resolved_sv_dict = load_resolved_svects(path_to_model) # dict(x, y, svect)
    nfft = int(resolved_sv_dict['nfft'])
    fs = int(resolved_sv_dict['fs'])

    coords = resolved_sv_dict['coords'] # [nFreq x nDoas x nChan x 6]
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
    RT60 = add_reverberation
    if RT60 > 0:
        # generate spatial reverberation with pyroomacoustics

        # convolve in the spatial domain
        pass
    
    # make mixtures
    
    # load speech signal
    # get the list of all the speech files
    if source_type == "speech":
        speech_files = list(path_to_speech_data.glob("*.wav"))
        # randomly select n_sources speech files
        speech_files = np.random.choice(speech_files, n_sources)
        src_signals = []
        for i, speech_file in enumerate(speech_files):
            s, fs = librosa.load(speech_file, sr=fs, duration=sound_duration, mono=True, offset=0.5)
            s = s / np.std(s)
            src_signals.append(s)
    elif "alpha" in source_type:
        src_alpha = float(source_type.split("-")[0])
        size = (int(fs * sound_duration),)
        s = levy_stable.rvs(alpha=src_alpha, beta=0, loc=0, size=size)
        s = s / np.std(s)
    elif source_type == "gauss":
        size = (int(fs * sound_duration),)
        s = np.random.randn(size)
        s = s / np.std(s)
    src_signals = np.array(src_signals)
    logger.debug("src_signals shape: ", src_signals.shape)
    # check the the selected source are active
    assert np.all(np.std(src_signals, axis=-1) > 0), "Source signals are empty"
    
    # make mixture
    x = []
    for i in range(svect_ref_time.shape[2]):
        xi = []
        for j in range(len(src_signals)):
            s = src_signals[j] / np.std(src_signals[j])
            xi.append(np.convolve(svect_ref_time[:,src_doas_idx[j],i], s, mode='full'))
        x.append(np.sum(np.array(xi), axis=0))
    mixture = np.array(x) # [nChan x nSamples]
    logger.debug("Mixture shape: ", mixture.shape)

    # add noise
    if noise_type == 'awgn':
        noise = np.random.randn(*mixture.shape)
    elif "alpha" in noise_type:
        noise_alpha = float(noise_type.split("-")[0])
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
    
    return mixture, doas_ref
    
    
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
    snr, noise_type, add_reverberation, 
    loc_method, freq_range, sv_method, seed, nObs, sv_normalization, 
    mc_seed=1, exp_name=None
    ):
    
    # set seed for reproducibility
    np.random.seed(mc_seed)
    
    mixture, doas_ref = make_data(src_doas_idx, source_type, sound_duration, snr, noise_type, add_reverberation, mc_seed=mc_seed)
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
    
    return doas_est, doas_est_idx, error, doas_ref, doas_ref_idx, ang_spec, ang_spec_freqs


if __name__ == "__main__":
    args = parser.parse_args()
    
    experiment_case = args.exp_id
    
    # Experiment 0 - Tuning alpha-stable
    if experiment_case == 0:
        
        """ 
        This expermiment check the rebustness of the localizatiion against the SNR
        - one source at DOAs every 12 degree from -90 to 90
        - SNR from -40 to 10 dB
        """
                
        # define the hyperparameters space for the data
        target_doa = np.concatenate([np.arange(0, 15, 2), np.arange(60-15, 60, 2)]).tolist()
        logger.debug("Target DOAs: ", target_doa)
        source_type = ['speech']
        snr = np.arange(-10, 20, 5).tolist()
        logger.debug("SNRs: ", snr)
        noise_type = ['alpha_stable', 'awgn']
        sound_duration = [1.]
        add_reverberation = [False]
        sv_normalization = True
        
        monte_carlo_run_per_setting = np.arange(3).tolist()
        
        # compile the list of mixtures
        data_settings = list(itertools.product(target_doa, source_type, sound_duration, snr, noise_type, add_reverberation, monte_carlo_run_per_setting))
                
        # Steering vector models
        # compile the list of models combining the method with nObs and seed
        sv_model_choices = [['ref', 8, 13], ['alg', 8, 13]]
        # sv_model_choices += list(itertools.product(sv_methods_choices, sv_nObs_choice, sv_seed_choice))
        # sv_model_choices += list(itertools.product(['gp-steerer'], [32], [666]))
        
        # Localization method hparams
        min_freq = 200
        max_freq = 4000
        freq_range = [min_freq, max_freq]
        
        results_df = pd.DataFrame()
        results_list = []
        
        # load the results dataframe if it exists
        csv_filename = f"experiment_results_exp-{experiment_case}.csv"
        if (results_dir / csv_filename).exists():
            results_df = pd.read_csv(results_dir / csv_filename)
        
        counter_exp = 0
        
        for setting in tqdm(data_settings):
            
            target_doa, source_type, sound_duration, snr, noise_type, add_reverberation, mc_seed = setting
            src_doas_idx = [target_doa]
            
            now = datetime.datetime.now()
            date_str = datetime.datetime.strftime(now, "%Y%m%d-%H%M%S")
            
            for loc_method in ang_spec_methods_choices:
                
                # decode loc_method name
                if "alpha" in loc_method:
                    format_string = 'alpha-{alpha:f}_beta-{beta:f}_eps-{eps:f}_iter-{niter:d}'
                    parsed = parse.parse(format_string, loc_method)
                    sm_alpha = parsed['alpha']
                    sm_beta = parsed['beta']
                    sm_eps = parsed['eps']
                    sm_niter = parsed['niter']
                else:
                    sm_alpha = sm_beta = sm_eps = sm_niter = None
                
                results = dict()
                
                # run for all the other steering vectors
                for sv_model_name in sv_model_choices:
                    
                    sv_method, nObs, seed = sv_model_name
                
                    logger.debug("### Running experiment ###")
                    logger.debug("scene settings: ", src_doas_idx, sound_duration, snr, noise_type, add_reverberation, mc_seed)
                    logger.debug("model settings: ", sv_method, nObs, seed, sv_normalization)
                    logger.debug("loc_method: ", loc_method)
                    
                    exp_name =  f"exp-{experiment_case}_doas-{src_doas_idx}_duration-{sound_duration}-snr-{snr}_noise-{noise_type}_reverb-{add_reverberation}_loc-{loc_method}_freq-{freq_range}_sv-{sv_method}_nObs-{nObs}_seed-{seed}_norm-{sv_normalization}_mc-{mc_seed}"
                    
                    # check if the experiment has already been run by checking the exp_name in the results_df
                    if len(results_df) > 0 and results_df.query(f"exp_name == '{exp_name}'").shape[0] > 0:
                        logger.info(f"Experiment {exp_name} already run")
                        continue
                    
                    doas_est, doas_est_idx, error, doas_ref, doas_ref_idx, ang_spec = process_experiment(
                        src_doas_idx, source_type, sound_duration, snr, noise_type, add_reverberation,
                        loc_method, freq_range, 
                        sv_method, seed, nObs, sv_normalization,
                        mc_seed=mc_seed,
                        exp_name=exp_name,
                    )
                    results = {
                        "doas_est_idx_s1": doas_est_idx[0],
                        "doas_ref_idx_s1": doas_ref_idx[0],
                        "doas_ref_s1_az": doas_ref[0][1],
                        "doas_ref_s1_el": doas_ref[0][0],
                        "doas_est_s1_az": doas_est[0][1],
                        "doas_est_s1_el": doas_est[0][0],
                        "error_s1": error[0],
                    }

                    scene_parameters = dict(
                        target_doa=target_doa,
                        duration=sound_duration,
                        snr=snr,
                        noise_type=noise_type,
                        add_reverberation=add_reverberation,
                        mc_seed=mc_seed,
                    )
                    model_parameters = dict(
                        loc_method=loc_method,
                        freq_min=freq_range[0],
                        freq_max=freq_range[1],
                        sv_method=sv_method,
                        nObs=nObs,
                        seed=seed,
                        sv_normalization=sv_normalization,
                        sm_alpha=sm_alpha,
                        sm_beta=sm_beta,
                        sm_eps=sm_eps,
                        sm_niter=sm_niter,
                    )
                    record = dict(
                        scene_parameters=scene_parameters,
                        model_parameters=model_parameters,
                        results=results,
                        time=date_str,
                        exp_name=exp_name,
                    )

                    # pprint(record)
                    
                    results_list.append(record)
                    
                    # Append the record to the dataframe
                    # Flatten the nested dictionary
                    flat_record = {
                        **record['scene_parameters'],
                        **record['model_parameters'],
                        **record['results'],
                        'time': record['time'],
                        'exp_name': record['exp_name']
                    }
                    results_df = pd.concat([results_df, pd.DataFrame([flat_record])], ignore_index=True)

                    counter_exp += 1
                    
                    if counter_exp % 5 == 0:
                        # Save the dataframe to a CSV file
                        results_df.to_csv(results_dir / csv_filename, index=False)
                        # Save the list of dict to a json file
                        with open(results_dir / csv_filename.replace("csv", "json"), 'w') as f:
                            json.dump(results_list, f)
    
    # Experiment 1 - SNR robustness
    elif experiment_case == 1:
        
        """ 
        This expermiment check the rebustness of the localizatiion against the SNR
        - one source at DOAs every 10 degree from -90 to 90
        - SNR from -30 to 15 dB
        """
                
        # define the hyperparameters space for the data
        target_doa = np.concatenate([np.arange(0, 15, 2), np.arange(60-15, 60, 2)]).tolist()
        print("Target DOAs: ", target_doa)
        snr = np.arange(-30, 10, 5).tolist()
        print("SNRs: ", snr)
        noise_type = ['awgn']
        sound_duration = [1.]
        add_reverberation = [False]
        sv_normalization = True
        
        monte_carlo_run_per_setting = np.arange(1).tolist()
        
        # compile the list of mixtures
        data_settings = list(itertools.product(target_doa, sound_duration, snr, noise_type, add_reverberation, monte_carlo_run_per_setting))
                
        # Steering vector models
        # compile the list of models combining the method with nObs and seed
        sv_model_choices = [['ref', 8, 13], ['alg', 8, 13]]
        # sv_model_choices += list(itertools.product(sv_methods_choices, sv_nObs_choice, sv_seed_choice))
        sv_model_choices += list(itertools.product(['gp-steerer'], sv_nObs_choice, sv_seed_choice))
        
        # Localization method hparams
        min_freq = 200
        max_freq = 4000
        freq_range = [min_freq, max_freq]
        
        results_df = pd.DataFrame()
        results_list = []
        
        # load the results dataframe if it exists
        csv_filename = f"experiment_results_exp-{experiment_case}.csv"
        if (results_dir / csv_filename).exists():
            results_df = pd.read_csv(results_dir / csv_filename)
        
        counter_exp = 0
        
        for setting in tqdm(data_settings):
            
            target_doa, sound_duration, snr, noise_type, add_reverberation, mc_seed = setting
            src_doas_idx = [target_doa]
            
            now = datetime.datetime.now()
            date_str = datetime.datetime.strftime(now, "%Y%m%d-%H%M%S")
            
            for loc_method in ang_spec_methods_choices:
                
                results = dict()
                
                # run for all the other steering vectors
                for sv_model_name in sv_model_choices:
                    
                    sv_method, nObs, seed = sv_model_name
                
                    print("### Running experiment ###")
                    print("scene settings: ", src_doas_idx, sound_duration, snr, noise_type, add_reverberation, mc_seed)
                    print("model settings: ", sv_method, nObs, seed, sv_normalization)
                    print("loc_method: ", loc_method)
                    
                    exp_name =  f"exp-{experiment_case}_doas-{src_doas_idx}_duration-{sound_duration}-snr-{snr}_noise-{noise_type}_reverb-{add_reverberation}_loc-{loc_method}_freq-{freq_range}_sv-{sv_method}_nObs-{nObs}_seed-{seed}_norm-{sv_normalization}_mc-{mc_seed}"
                    
                    # check if the experiment has already been run by checking the exp_name in the results_df
                    if len(results_df) > 0 and results_df.query(f"exp_name == '{exp_name}'").shape[0] > 0:
                        print(f"Experiment {exp_name} already run")
                        continue
                    
                    doas_est, doas_est_idx, error, doas_ref, doas_ref_idx, ang_spec = process_experiment(
                        src_doas_idx, sound_duration, snr, noise_type, add_reverberation,
                        loc_method, freq_range, 
                        sv_method, seed, nObs, sv_normalization,
                        mc_seed=mc_seed,
                        exp_name=exp_name,
                    )
                    results = {
                        "doas_est_idx_s1": doas_est_idx[0],
                        "doas_ref_idx_s1": doas_ref_idx[0],
                        "doas_ref_s1_az": doas_ref[0][1],
                        "doas_ref_s1_el": doas_ref[0][0],
                        "doas_est_s1_az": doas_est[0][1],
                        "doas_est_s1_el": doas_est[0][0],
                        "error_s1": error[0],
                    }

                    scene_parameters = dict(
                        target_doa=target_doa,
                        duration=sound_duration,
                        snr=snr,
                        noise_type=noise_type,
                        add_reverberation=add_reverberation,
                        mc_seed=mc_seed,
                    )
                    model_parameters = dict(
                        loc_method=loc_method,
                        freq_min=freq_range[0],
                        freq_max=freq_range[1],
                        sv_method=sv_method,
                        nObs=nObs,
                        seed=seed,
                        sv_normalization=sv_normalization,
                    )
                    record = dict(
                        scene_parameters=scene_parameters,
                        model_parameters=model_parameters,
                        results=results,
                        time=date_str,
                        exp_name=exp_name,
                    )

                    # pprint(record)
                    
                    results_list.append(record)
                    
                    # Append the record to the dataframe
                    # Flatten the nested dictionary
                    flat_record = {
                        **record['scene_parameters'],
                        **record['model_parameters'],
                        **record['results'],
                        'time': record['time'],
                        'exp_name': record['exp_name']
                    }
                    results_df = pd.concat([results_df, pd.DataFrame([flat_record])], ignore_index=True)

                    counter_exp += 1
                    
                    if counter_exp % 5 == 0:
                        # Save the dataframe to a CSV file
                        results_df.to_csv(results_dir / csv_filename, index=False)
                        # Save the list of dict to a json file
                        with open(results_dir / csv_filename.replace("csv", "json"), 'w') as f:
                            json.dump(results_list, f)
    
    # Experiment 2 - Separation angle robustness
    elif experiment_case == 2:
        
        """ 
        This expermiment check the rebustness of the localizatiion for the separation angle
        - one source at DOAs every 10 degree from -90 to 90
        - second source is vary for 0 to 90 degree
        - SNR is set to 0
        """
        
        # define the hyperparameters space for the data
        target_doa = np.concatenate([np.arange(0, 15, 1), np.arange(60-15, 60, 1)]).tolist()
        sep_angle = np.arange(1, 15, 1).tolist()
        print("Target DOAs: ", target_doa)
        print("Separation angles: ", sep_angle)
        snr = [0]
        print("SNRs: ", snr)
        noise_type = ['awgn']
        sound_duration = [1.]
        add_reverberation = [False]
        monte_carlo_run_per_setting = np.arange(1).tolist()
        
        sv_normalization = True
        
        # compile the list of mixtures
        data_settings = list(itertools.product(target_doa, sep_angle, sound_duration, snr, noise_type, add_reverberation, monte_carlo_run_per_setting))
                
        # Steering vector models
        # compile the list of models combining the method with nObs and seed
        sv_model_choices = [['ref', 8, 13], ['alg', 8, 13]]
        # sv_model_choices += list(itertools.product(sv_methods_choices, sv_nObs_choice, sv_seed_choice))
        sv_model_choices += list(itertools.product(['gp-steerer'], sv_nObs_choice, sv_seed_choice))
        
        # Localization method hparams
        min_freq = 200
        max_freq = 4000
        freq_range = [min_freq, max_freq]
        
        results_df = pd.DataFrame()
        results_list = []
        
        # load the results dataframe if it exists
        csv_filename = f"experiment_results_exp-{experiment_case}.csv"
        if (results_dir / csv_filename).exists():
            results_df = pd.read_csv(results_dir / csv_filename)
        
        counter_exp = 0
        
        for setting in data_settings:
            
            target_doa, sep_angle, sound_duration, snr, noise_type, add_reverberation, mc_seed = setting
            src_doas_idx = [target_doa, target_doa + sep_angle]
            
            now = datetime.datetime.now()
            date_str = datetime.datetime.strftime(now, "%Y%m%d-%H%M%S")
            
            for loc_method in ang_spec_methods_choices:
                
                results = dict()
                
                # run for all the other steering vectors
                for sv_model_name in sv_model_choices:
                    
                    sv_method, nObs, seed = sv_model_name
                
                    print("### Running experiment ###")
                    print("scene settings: ", src_doas_idx, sound_duration, snr, noise_type, add_reverberation, mc_seed)
                    print("model settings: ", sv_method, nObs, seed, sv_normalization)
                    print("loc_method: ", loc_method)
                    
                    exp_name =  f"exp-{experiment_case}_doas-{src_doas_idx}_duration-{sound_duration}-snr-{snr}_noise-{noise_type}_reverb-{add_reverberation}_loc-{loc_method}_freq-{freq_range}_sv-{sv_method}_nObs-{nObs}_seed-{seed}_norm-{sv_normalization}_mc-{mc_seed}"
                    
                    # check if the experiment has already been run by checking the exp_name in the results_df
                    if len(results_df) > 0 and results_df.query(f"exp_name == '{exp_name}'").shape[0] > 0:
                        print(f"Experiment {exp_name} already run")
                        continue
                    
                    doas_est, doas_est_idx, error, doas_ref, doas_ref_idx, ang_spec = process_experiment(
                        src_doas_idx, sound_duration, snr, noise_type, add_reverberation,
                        loc_method, freq_range, 
                        sv_method, seed, nObs, sv_normalization,
                        mc_seed=mc_seed,
                        exp_name=exp_name,
                    )
                    
                    results = {
                        "doas_est_idx_s1": doas_est_idx[0],
                        "doas_est_idx_s2": doas_est_idx[1],
                        "doas_ref_idx_s1": doas_ref_idx[0],
                        "doas_ref_idx_s2": doas_ref_idx[1],
                        "doas_ref_s1_az": doas_ref[0][1],
                        "doas_ref_s1_el": doas_ref[0][0],
                        "doas_ref_s2_az": doas_ref[1][1],
                        "doas_ref_s2_el": doas_ref[1][0],
                        "doas_est_s1_az": doas_est[0][1],
                        "doas_est_s1_el": doas_est[0][0],
                        "doas_est_s2_az": doas_est[1][1],
                        "doas_est_s2_el": doas_est[1][0],
                        "error_s1": error[0],
                        "error_s2": error[1],
                    }

                    scene_parameters = dict(
                        target_doa=target_doa,
                        duration=sound_duration,
                        snr=snr,
                        noise_type=noise_type,
                        add_reverberation=add_reverberation,
                    )
                    model_parameters = dict(
                        loc_method=loc_method,
                        freq_min=freq_range[0],
                        freq_max=freq_range[1],
                        sv_method=sv_method,
                        nObs=nObs,
                        seed=seed,
                        sv_normalization=sv_normalization,
                    )
                    record = dict(
                        scene_parameters=scene_parameters,
                        model_parameters=model_parameters,
                        results=results,
                        time=date_str,
                        exp_name=exp_name,
                    )

                    pprint(record)
                    
                    results_list.append(record)
                    
                    # Append the record to the dataframe
                    # Flatten the nested dictionary
                    flat_record = {
                        **record['scene_parameters'],
                        **record['model_parameters'],
                        **record['results'],
                        'time': record['time'],
                        'exp_name': record['exp_name']
                    }
                    results_df = pd.concat([results_df, pd.DataFrame([flat_record])], ignore_index=True)

                    counter_exp += 1
                    
                    if counter_exp % 5 == 0:
                        # Save the dataframe to a CSV file
                        results_df.to_csv(results_dir / csv_filename, index=False)
                        # Save the list of dict to a json file
                        with open(results_dir / csv_filename.replace("csv", "json"), 'w') as f:
                            json.dump(results_list, f)                
    
    # Experiment 3 / 4 - Multiple sources, SNR = 20, speech (exp 4 uses the ang spec with a thr. I can be done later)
    elif experiment_case == 3:
        
        figure_dir = figure_dir / f"exp-{experiment_case}"
        figure_dir.mkdir(parents=True, exist_ok=True)
        output_dir = output_dir / f"exp-{experiment_case}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        """ 
        This expermiment check the rebustness of the localization of multiple sources
        - draw n_sources sources at random DOAs
        - SNR is set to 20
        """
        
        # define the hyperparameters space for the data
        max_number_of_sources = 5
        n_sources_choice = np.arange(1,max_number_of_sources+1).tolist()
        source_type = ['speech']
        snr = [20]
        noise_type = ['awgn']
        sound_duration = [0.5, 1.]
        add_reverberation = [False]
        monte_carlo_run_per_setting = np.arange(15).tolist() # <- how many times to run the same setting (sampling a source and a DOA)
        
        sv_normalization = True
        
        # compile the list of mixtures
        data_settings = list(itertools.product(n_sources_choice, source_type, sound_duration, snr, noise_type, add_reverberation, monte_carlo_run_per_setting))
                
        # Steering vector models
        # compile the list of models combining the method with nObs and seed
        sv_model_choices = [['ref', 8, 13], ['alg', 8, 13]]
        # sv_model_choices += list(itertools.product(sv_methods_choices, sv_nObs_choice, sv_seed_choice))
        sv_model_choices += list(itertools.product(['gp-steerer'], sv_nObs_choice, sv_seed_choice))
        
        # Localization method hparams
        min_freq = 200
        max_freq = 4000
        freq_range = [min_freq, max_freq]
        # indeces of the DOA grid of the EASYCOM/SPEAR data, that is [0,6degree,360]
        doa_grid = np.arange(0, 360, 6)
        doa_grid_idx = np.arange(60)
        
        results_all_concat_df = pd.DataFrame()
        ang_spec_all_concat_list = []
        
        # load the results dataframe if it exists
        csv_filename = f"experiment_results_exp-{experiment_case}.csv"
        if (results_dir / csv_filename).exists():
            results_all_concat_df = pd.read_csv(results_dir / csv_filename)
        
        counter_exp = 0
        
        for n_sources, source_type, sound_duration, snr, noise_type, add_reverberation, mc_seed in tqdm(data_settings, desc="For scene settings"):
            
            # set the seed for reproducibility
            np.random.seed(mc_seed)
            
            # draw n_sources sources at random DOAs
            src_doas_idx = np.random.choice(doa_grid_idx, n_sources)
            
            # create a frame_id using the sources DOAs
            frame_id = f"nSrc-{n_sources}_doas-{src_doas_idx}_type-{source_type}-duration-{sound_duration}-snr-{snr}_noise-{noise_type}_reverb-{add_reverberation}_mc-{mc_seed}"
            
            now = datetime.datetime.now()
            date_str = datetime.datetime.strftime(now, "%Y%m%d-%H%M%S")
            
            for loc_method in tqdm(ang_spec_methods_choices):
                
                results = dict()
                
                # run for all the other steering vectors
                for sv_method, nObs, seed in sv_model_choices:
                    
                    method_id = f"{loc_method}_freqs-{freq_range}_{sv_method}_nObs-{nObs}_seed-{seed}_norm-{sv_normalization}"
                
                    logger.info("### Running experiment ###")
                    logger.info("scene settings: ", src_doas_idx, sound_duration, snr, noise_type, add_reverberation, mc_seed)
                    logger.info("model settings: ", sv_method, nObs, seed, sv_normalization)
                    logger.info("loc_method: ", loc_method)
                    
                    exp_name =  f"exp-{experiment_case}_{frame_id}_{method_id}"
                                        
                    # check if the experiment has already been run by checking the exp_name in the results_all_concat_df
                    if len(results_all_concat_df) > 0 and results_all_concat_df.query(f"exp_name == '{exp_name}'").shape[0] > 0:
                        logger.warning(f"Experiment {exp_name} already run")
                        continue
                    
                    doas_est, doas_est_idx, error, doas_ref, doas_ref_idx, ang_spec, ang_spec_freqs = process_experiment(
                        src_doas_idx, source_type, sound_duration, snr, noise_type, add_reverberation,
                        loc_method, freq_range, 
                        sv_method, seed, nObs, sv_normalization,
                        mc_seed=mc_seed,
                        exp_name=exp_name,
                    )
                    
                    # Record the results
                    results = {
                        "exp_name": exp_name,
                        "time": date_str,
                        "record_id" : [f's{i}' for i in np.arange(n_sources)],
                        "num_srcs" : n_sources,
                        "src_ids": np.arange(n_sources).tolist(),
                        "doas_est_idx": doas_est_idx,
                        "doas_ref_idx": doas_ref_idx,
                        "doas_ref_az": [doa[1] for doa in doas_ref],
                        "doas_est_az": [doa[1] for doa in doas_est],
                        "doas_ref_el": [doa[0] for doa in doas_ref],
                        "doas_est_el": [doa[0] for doa in doas_est],
                        "errors": error,                        
                    }
                    df_results_ = pd.DataFrame(results)
            
                    # Record the scene parameters
                    scene_parameters = {
                        "record_id" : [f's{i}' for i in np.arange(n_sources)],
                        "frame_id" : [frame_id] * n_sources,
                        "target_doa" : doas_ref,
                        "n_sources" : [n_sources] * n_sources,
                        "duration" : [sound_duration] * n_sources,
                        "snr" : [snr] * n_sources,
                        "noise_type" : [noise_type] * n_sources,
                        "add_reverberation" : [add_reverberation] * n_sources,
                        "mc_seed" : [mc_seed] * n_sources,
                    }
                    df_scene_ = pd.DataFrame(scene_parameters)
                    df_results_scene_ = pd.merge(df_results_, df_scene_, on='record_id')
                    
                    # Record the model parameters
                    model_parameters = {
                        "record_id" : [f's{i}' for i in np.arange(n_sources)],
                        "method_id" : [method_id] * n_sources,
                        "loc_method": [loc_method] * n_sources,
                        "freq_min": [freq_range[0]] * n_sources,
                        "freq_max": [freq_range[1]] * n_sources,
                        "sv_method": [sv_method] * n_sources,
                        "nObs": [nObs] * n_sources,
                        "seed": [seed] * n_sources,
                        "sv_normalization": [sv_normalization] * n_sources,
                    }
                    df_model_ = pd.DataFrame(model_parameters)
                    df_results_scene_model_ = pd.merge(df_results_scene_, df_model_, on='record_id')
                    
                    results_all_concat_df = pd.concat([results_all_concat_df, df_results_scene_model_], ignore_index=True)
                    
                    doa_freq_grid = np.stack(np.meshgrid(doa_grid, ang_spec_freqs, indexing='ij'), -1)
                    ang_spec_dict = {
                        "frame_id" : frame_id,
                        "method_id" : method_id,
                        "loc_method" : loc_method,
                        "ang_spec_shape" : ang_spec.shape,
                        "doa_grid" : doa_freq_grid[...,0],
                        "freq_grid" : doa_freq_grid[...,1],
                        "ang_spec" : ang_spec,
                    }
                    ang_spec_all_concat_list.append(ang_spec_dict)
                    
                    counter_exp += 1
                    
                    if counter_exp % 20 == 0:
                        # Save the dataframe to a CSV file
                        results_all_concat_df.to_csv(results_dir / csv_filename, index=False)
                        # Save the list of dict to pickle
                        with open(results_dir / csv_filename.replace(".csv", "_with_ang_specs.pkl"), 'wb') as f:
                            pickle.dump(ang_spec_all_concat_list, f)
    
                       
    # Experiment 5 - Multiple sources, SNR = 20, varying alpha with synthetic sources
    elif experiment_case == 5:
        
        """ 
        This expermiment check the rebustness of the localization of multiple sources with different distribution
        """
        
        # define the hyperparameters space for the data
        nSrc = 5
        n_sources_choice = np.arange(1,nSrc+1).tolist()
        source_type = ['speech', 'alpha-1.2', 'alpha-2.0', 'gauss']
        snr = [20]
        noise_type = ['awgn']
        sound_duration = [1.]
        add_reverberation = [False]
        monte_carlo_run_per_setting = np.arange(15).tolist() # <- how many times to run the same setting (sampling a source and a DOA)
        
        sv_normalization = True
        
        # compile the list of mixtures
        data_settings = list(itertools.product(n_sources_choice, source_type, sound_duration, snr, noise_type, add_reverberation, monte_carlo_run_per_setting))
                
        # Steering vector models
        # compile the list of models combining the method with nObs and seed
        sv_model_choices = [['ref', 8, 13], ['alg', 8, 13]]
        # sv_model_choices += list(itertools.product(sv_methods_choices, sv_nObs_choice, sv_seed_choice))
        sv_model_choices += list(itertools.product(['gp-steerer'], sv_nObs_choice, sv_seed_choice))
        
        # Localization method hparams
        min_freq = 200
        max_freq = 4000
        freq_range = [min_freq, max_freq]
        # indeces of the DOA grid of the EASYCOM/SPEAR data, that is [0,6degree,360]
        doa_grid = np.arange(60) 
        
        results_df = pd.DataFrame()
        results_list = []
        
        # load the results dataframe if it exists
        csv_filename = f"experiment_results_exp-{experiment_case}.csv"
        if (results_dir / csv_filename).exists():
            results_df = pd.read_csv(results_dir / csv_filename)
        
        counter_exp = 0
        
        for setting in tqdm(data_settings):
            
            n_sources, source_type, sound_duration, snr, noise_type, add_reverberation, mc_seed = setting
            
            # set the seed for reproducibility
            np.random.seed(mc_seed)
            
            # draw n_sources sources at random DOAs
            src_doas_idx = np.random.choice(doa_grid, n_sources)
            
            # create a frame_id using the sources DOAs
            frame_id = f"nSrc-{n_sources}_srcIds-{src_doas_idx}-duration-{sound_duration}-snr-{snr}_noise-{noise_type}_reverb-{add_reverberation}_mc-{mc_seed}"
            
            now = datetime.datetime.now()
            date_str = datetime.datetime.strftime(now, "%Y%m%d-%H%M%S")
            
            for loc_method in ang_spec_methods_choices:
                
                results = dict()
                
                # run for all the other steering vectors
                for sv_model_name in sv_model_choices:
                    
                    sv_method, nObs, seed = sv_model_name
                
                    logger.info("### Running experiment ###")
                    logger.info("scene settings: ", src_doas_idx, sound_duration, snr, noise_type, add_reverberation, mc_seed)
                    logger.info("model settings: ", sv_method, nObs, seed, sv_normalization)
                    logger.info("loc_method: ", loc_method)
                    
                    exp_name =  f"exp-{experiment_case}_doas-{src_doas_idx}_duration-{sound_duration}-snr-{snr}_noise-{noise_type}_reverb-{add_reverberation}_loc-{loc_method}_freq-{freq_range}_sv-{sv_method}_nObs-{nObs}_seed-{seed}_norm-{sv_normalization}_mc-{mc_seed}"
                    
                    # check if the experiment has already been run by checking the exp_name in the results_df
                    if len(results_df) > 0 and results_df.query(f"exp_name == '{exp_name}'").shape[0] > 0:
                        logger.warning(f"Experiment {exp_name} already run")
                        continue
                    
                    doas_est, doas_est_idx, error, doas_ref, doas_ref_idx, ang_spec = process_experiment(
                        src_doas_idx, source_type, sound_duration, snr, noise_type, add_reverberation,
                        loc_method, freq_range, 
                        sv_method, seed, nObs, sv_normalization,
                        mc_seed=mc_seed,
                        exp_name=exp_name,
                    )
                    
                    results = {
                        "exp_name": exp_name,
                        "time": date_str,
                        "example_id" : [f's{i}' for i in np.arange(n_sources)],
                        "num_srcs": np.arange(n_sources).tolist(),
                        "doas_est_idx": doas_est_idx,
                        "doas_ref_idx": doas_ref_idx,
                        "doas_ref_az": [doa[1] for doa in doas_ref],
                        "doas_ref_el": [doa[0] for doa in doas_ref],
                        "doas_est_az": [doa[1] for doa in doas_est],
                        "doas_est_el": [doa[0] for doa in doas_est],
                        "errors": error,                        
                    }
                    df_results = pd.DataFrame(results)
            
                    scene_parameters = {
                        "example_id" : [f's{i}' for i in np.arange(n_sources)],
                        "source_type" : [source_type] * n_sources,
                        "target_doa" : doas_ref,
                        "frame_id" : [frame_id] * n_sources,
                        "n_sources" : [n_sources] * n_sources,
                        "duration" : [sound_duration] * n_sources,
                        "snr" : [snr] * n_sources,
                        "noise_type" : [noise_type] * n_sources,
                        "add_reverberation" : [add_reverberation] * n_sources,
                        "mc_seed" : [mc_seed] * n_sources,
                    }
                    df_scene = pd.DataFrame(scene_parameters)
                    df_tmp = pd.merge(df_results, df_scene, on='example_id')
                    method_id = [f"{loc_method}_freqs-{freq_range}_{sv_method}_nObs-{nObs}_seed-{seed}_norm-{sv_normalization}"]
                    model_parameters = {
                        "example_id" : [f's{i}' for i in np.arange(n_sources)],
                        "method_id" : [method_id] * n_sources,
                        "loc_method": [loc_method] * n_sources,
                        "freq_min": [freq_range[0]] * n_sources,
                        "freq_max": [freq_range[1]] * n_sources,
                        "sv_method": [sv_method] * n_sources,
                        "nObs": [nObs] * n_sources,
                        "seed": [seed] * n_sources,
                        "sv_normalization": [sv_normalization] * n_sources,
                    }
                    df_model = pd.DataFrame(model_parameters)
                    df_results_ = pd.merge(df_tmp, df_model, on='example_id')
                    
                    results_df = pd.concat([results_df, df_results_], ignore_index=True)

                    counter_exp += 1
                    
                    if counter_exp % 20 == 0:
                        # Save the dataframe to a CSV file
                        results_df.to_csv(results_dir / csv_filename, index=False)
    
    else:
        
        src_doas = [5, 40]
        source_type = 'speech'
        sound_duration = 0.5
        snr = -5
        noise_type = 'awgn'
        add_reverberation = False
        
        loc_method = 'alpha_stable'
        
        sv_method = 'gp-steerer'
        nObs = 8
        seed = 13
        sv_normalization = True

        # Create a string for the file name
        process_experiment(
            src_doas, source_type, sound_duration, snr, noise_type, add_reverberation,
            loc_method,
            sv_method, seed, nObs, sv_normalization,
        )