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

from localizers import methods as ang_spect_methods

import numpy as np
import librosa
import soundfile as sf

import seaborn as sns
import matplotlib.pyplot as plt

from einops import rearrange
import itertools
import scipy.signal

import pyroomacoustics as pra

from pprint import pprint

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


ang_spect_methods_choices = ang_spect_methods.keys()
sv_methods_choices = eusipco_names_to_exp_name.keys()
sv_nObs_choice = [8, 16, 32, 64, 128]
sv_seed_choice = [13, 42, 666]


parser = argparse.ArgumentParser(description='Sound Source Localization with NSteerer and AlphaStable')
parser.add_argument('--sv-method', type=str, default='nf-subfreq', help='Steering vector interpolation method', choices=['nf', 'nf-gw' 'nn', 'sp', 'pinn', 'gp-steerer'])
parser.add_argument('--nObs', type=int, default=8, help='Number of observations used to fit the sv model', choices=[8, 16, 32, 64, 128])
parser.add_argument('--seed', type=int, default=13, help='Random seed used for training the interpolation methods', choices=[13, 42, 666])
parser.add_argument('--sv-normalization', action='store_true', help='Normalize the steering vectors')
parser.add_argument('--exp-id', type=int, default=None, help='Name of the experiment')

def make_data(src_doas_idx, sound_duration, SNR, noise_type='white', add_reverberation=False):
    
    file_name_data = f"doas-{src_doas_idx}_snr-{snr}_noise-{noise_type}_reverb-{add_reverberation}"
    
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
    
    # make mixtures
    n_sources = len(src_doas_idx)
    
    # load speech signal
    s1, fs = librosa.load(path_to_speech_data / 'DR2_MDWD0_SI1260_SI557_SX90_8s.wav', sr=fs, duration=sound_duration, mono=True, offset=0.5)
    s2, fs = librosa.load(path_to_speech_data / 'DR1_MWAR0_SI2305_SI1045_7s.wav', sr=fs, duration=sound_duration, mono=True, offset=0.5)
    s3, fs = librosa.load(path_to_speech_data / 'DR5_FSKP0_SI1728_SI468_SX288_8s.wav', sr=fs, duration=sound_duration, mono=True, offset=0.5)
    s1 = s1 / np.std(s1)
    s2 = s2 / np.std(s2)
    s3 = s3 / np.std(s3)
    src_signals = np.array([s1, s2, s3])[:n_sources]
    print("src_signals shape: ", src_signals.shape)
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
    print("Mixture shape: ", mixture.shape)

    # add noise
    if noise_type == 'white':
        noise = np.random.randn(*mixture.shape)
    else:
        raise ValueError(f"Unknown noise type {noise_type}")
    noise = noise / np.linalg.norm(noise) * np.linalg.norm(mixture) / 10**(SNR/20)
    mixture = mixture + noise # [nChan x nSamples]
    print("Mixture shape: ", mixture.shape)

    time_src = np.arange(src_signals.shape[1]) / fs
    time_mix = np.arange(mixture.shape[1]) / fs 
    fig, axarr = plt.subplots(n_sources+1, 1, figsize=(10, 5), sharex=True)
    for i, ax in enumerate(axarr):
        if i < n_sources:
            ax.plot(time_src, src_signals[i], label=f's{i+1}')
        else:
            ax.plot(time_mix, x[0], label='mixture')
        ax.legend()
    plt.savefig(figure_dir / f'{file_name_data}_mixture.png')
    plt.close()
    
    doas_ref = np.array([doa_space[src_doas_idx[j]] for j in range(n_sources)])
    
    # save mixture as wave with doa in the file name
    mixture_to_save = mixture / np.max(np.abs(mixture))
    sf.write(output_dir / f'{file_name_data}.wav', mixture_to_save.T, fs)
    
    return mixture, doas_ref
    
    
def plot_ang_spec(S:np.array, doas_est_idx:np.array, doas_ref:np.array=None, title=None):
    """
    S in [nDoas x nFreq]
    doas in [nSources]
    """
    
    fig, axarr = plt.subplots(2,1, figsize=(6,3), sharex=True)
    axarr[0].imshow(S.T, aspect='auto', origin='lower')
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
    
    ang_spec = ang_spect_methods[loc_method](X, svect) # [nDoas x nFreq]
    assert ang_spec.shape[0] == (nDoas), f"Expected first shape {(nDoas)}, got {ang_spec.shape[0]}"
    
    # estimate source location, get the n_sources highest peaks
    ang_spec_poll = np.mean(ang_spec, axis=1) # pool over frequencies
    ang_spec_poll = ang_spec_poll / np.max(ang_spec_poll)
    doas_est_idx = find_peaks(ang_spec_poll, k=n_sources)
    if len(doas_est_idx) < n_sources:
        # it means that find peaks f
        # just take the the peak with a simple argmax
        doas_est_idx = np.argsort(-ang_spec_poll)[:n_sources]
    print("Estimated DOAs: ", doas_est_idx)
    doas_est = [doa_space[doas_est_idx[i]] for i in range(n_sources)]
    return doas_est, doas_est_idx, ang_spec


def compute_angle_between(doas_est, doas_ref):
    
    # change convention  
    # 1. (elevation, azimuth) -> (azimuth, elevation)
    doas_est = np.flip(doas_est.copy(), axis=1) 
    doas_ref = np.flip(doas_ref.copy(), axis=1) 
    # 2. (azimuth, inclination) -> (azimuth, elevation)
    doas_est[:,1] -= np.pi / 2
    doas_ref[:,1] -= np.pi / 2
    
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
            # print("Permutation: ", perm, "Error: ", error.sum())
            if error.sum() < min_error:
                min_error = error.sum()
                best_error = error
                best_perm = perm
                best_doas_est = doas_est[list(perm)]
    return best_error, best_doas_est, best_perm


def process_experiment(
    src_doas_idx, sound_duration, snr, noise_type, add_reverberation, loc_method, freq_range, sv_method, seed, nObs, sv_normalization, exp_name=None
    
    ):
    mixture, doas_ref = make_data(src_doas_idx, sound_duration, snr, noise_type, add_reverberation)
    n_sources = len(src_doas_idx)
    
    doas_est, doas_est_idx, ang_spec = localize(mixture, loc_method, freq_range, n_sources, sv_method, seed, nObs, sv_normalization)
    
    plot_ang_spec(ang_spec, doas_est_idx, title='Estimated Ang Spec')
    plt.savefig(figure_dir / f'{exp_name}_ang_spec_est.png')
    plt.close()
    
    print("Ground truth DOAs: ", doas_ref)
    error, doas_est, perm = compute_metrics(doas_ref, doas_est)
    doas_est_idx = [doas_est_idx[i] for i in perm]
    print("Best permutation: ", doas_est)
    print("Estimated DOAs: ", doas_est)
    print("Error: ", error)
    
    # if not list, make it a list
    doas_ref = np.array(doas_ref).tolist()
    doas_est = np.array(doas_est).tolist()
    doas_est_idx = np.array(doas_est_idx).tolist()
    doas_ref_idx = np.array(src_doas_idx).tolist()
    error = np.array(error).tolist()
    
    return doas_est, doas_est_idx, error, doas_ref, doas_ref_idx, ang_spec


if __name__ == "__main__":
    args = parser.parse_args()
    
    experiment_case = args.exp_id
    
    if experiment_case == 1:
        
        """ 
        This expermiment check the rebustness of the localizatiion against the SNR
        - one source at DOAs every 10 degree from -90 to 90
        - SNR from -30 to 15 dB
        """
        
        import pandas as pd
        
        # define the hyperparameters space for the data
        target_doa = np.concatenate([np.arange(0, 15, 1), np.arange(60-15, 60, 1)]).tolist()
        print("Target DOAs: ", target_doa)
        snr = np.arange(-30, 10, 5).tolist()
        print("SNRs: ", snr)
        noise_type = ['white']
        sound_duration = [0.5, 1.]
        add_reverberation = [False]
        sv_normalization = True
        
        # compile the list of mixtures
        data_settings = list(itertools.product(target_doa, sound_duration, snr, noise_type, add_reverberation))
                
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
            
            target_doa, sound_duration, snr, noise_type, add_reverberation = setting
            src_doas_idx = [target_doa]
            
            now = datetime.datetime.now()
            date_str = datetime.datetime.strftime(now, "%Y%m%d-%H%M%S")
            
            for loc_method in ang_spect_methods_choices:
                
                results = dict()
                
                # run for all the other steering vectors
                for sv_model_name in sv_model_choices:
                    
                    sv_method, nObs, seed = sv_model_name
                
                    print("### Running experiment ###")
                    print("scene settings: ", src_doas_idx, sound_duration, snr, noise_type, add_reverberation)
                    print("model settings: ", sv_method, nObs, seed, sv_normalization)
                    print("loc_method: ", loc_method)
                    
                    exp_name =  f"exp-{experiment_case}_doas-{src_doas_idx}_duration-{sound_duration}-snr-{snr}_noise-{noise_type}_reverb-{add_reverberation}_loc-{loc_method}_freq-{freq_range}_sv-{sv_method}_nObs-{nObs}_seed-{seed}_norm-{sv_normalization}"
                    
                    # check if the experiment has already been run by checking the exp_name in the results_df
                    if len(results_df) > 0 and results_df.query(f"exp_name == '{exp_name}'").shape[0] > 0:
                        print(f"Experiment {exp_name} already run")
                        continue
                    
                    doas_est, doas_est_idx, error, doas_ref, doas_ref_idx, ang_spec = process_experiment(
                        src_doas_idx, sound_duration, snr, noise_type, add_reverberation,
                        loc_method, freq_range, 
                        sv_method, seed, nObs, sv_normalization,
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
    
    elif experiment_case == 2:
        
        """ 
        This expermiment check the rebustness of the localizatiion for the separation angle
        - one source at DOAs every 10 degree from -90 to 90
        - second source is vary for 0 to 90 degree
        - SNR is set to 0
        """
        
        import pandas as pd
        
        # define the hyperparameters space for the data
        target_doa = np.concatenate([np.arange(0, 15, 1), np.arange(60-15, 60, 1)]).tolist()
        sep_angle = np.arange(1, 15, 1).tolist()
        print("Target DOAs: ", target_doa)
        print("Separation angles: ", sep_angle)
        snr = [0]
        print("SNRs: ", snr)
        noise_type = ['white']
        sound_duration = [1.]
        add_reverberation = [False]
        sv_normalization = True
        
        # compile the list of mixtures
        data_settings = list(itertools.product(target_doa, sep_angle, sound_duration, snr, noise_type, add_reverberation))
                
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
            
            target_doa, sep_angle, sound_duration, snr, noise_type, add_reverberation = setting
            src_doas_idx = [target_doa, target_doa + sep_angle]
            
            now = datetime.datetime.now()
            date_str = datetime.datetime.strftime(now, "%Y%m%d-%H%M%S")
            
            for loc_method in ang_spect_methods_choices:
                
                results = dict()
                
                # run for all the other steering vectors
                for sv_model_name in sv_model_choices:
                    
                    sv_method, nObs, seed = sv_model_name
                
                    print("### Running experiment ###")
                    print("scene settings: ", src_doas_idx, sound_duration, snr, noise_type, add_reverberation)
                    print("model settings: ", sv_method, nObs, seed, sv_normalization)
                    print("loc_method: ", loc_method)
                    
                    exp_name =  f"exp-{experiment_case}_doas-{src_doas_idx}_duration-{sound_duration}-snr-{snr}_noise-{noise_type}_reverb-{add_reverberation}_loc-{loc_method}_freq-{freq_range}_sv-{sv_method}_nObs-{nObs}_seed-{seed}_norm-{sv_normalization}"
                    
                    # check if the experiment has already been run by checking the exp_name in the results_df
                    if len(results_df) > 0 and results_df.query(f"exp_name == '{exp_name}'").shape[0] > 0:
                        print(f"Experiment {exp_name} already run")
                        continue
                    
                    doas_est, doas_est_idx, error, doas_ref, doas_ref_idx, ang_spec = process_experiment(
                        src_doas_idx, sound_duration, snr, noise_type, add_reverberation,
                        loc_method, freq_range, 
                        sv_method, seed, nObs, sv_normalization,
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
    else:
        
        src_doas = [5, 40]
        sound_duration = 0.5
        snr = -5
        noise_type = 'white'
        add_reverberation = False
        
        loc_method = 'alpha_stable'
        
        sv_method = 'gp-steerer'
        nObs = 8
        seed = 13
        sv_normalization = True

        # Create a string for the file name
        process_experiment(
            src_doas, sound_duration, snr, noise_type, add_reverberation,
            loc_method,
            sv_method, seed, nObs, sv_normalization,
        )