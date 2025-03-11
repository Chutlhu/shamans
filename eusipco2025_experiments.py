#!/usr/bin/env python3
import argparse
import itertools
import datetime
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import socket

from pathlib import Path

from shamans.main import process_experiment
from shamans.localizers import alpha_stable, music, srp_phat


# Assumed to be defined/imported elsewhere:
sv_nObs_choice = [8, 16, 32, 64, 128]
sv_seed_choice = [13, 42, 666]

# Setup directories
figure_dir = Path("./figures")
output_dir = Path("./output")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def setup_directory(base_dir, exp_id):
    """Create and return a directory specific to the experiment."""
    dir_path = base_dir / f"exp-{exp_id}"
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def load_results_csv(results_dir:Path, csv_filename:str) -> pd.DataFrame:
    """Load an existing CSV results file if it exists; otherwise return an empty DataFrame."""
    file_path = Path(results_dir) / csv_filename
    if file_path.exists():
        return pd.read_csv(file_path)
    return pd.DataFrame()

def load_results_pkl(results_dir:Path, pkl_filename:str) -> list:
    """Load an existing pickle file with angular spectrum results."""
    file_path = Path(results_dir) / pkl_filename
    if file_path.exists():
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return []

def save_results(results_df, ang_spec_list, results_dir, csv_filename):
    """Save results DataFrame and angular spectrum list."""
    results_df.to_csv(results_dir / csv_filename, index=False)
    pkl_filename = csv_filename.replace(".csv", "_with_ang_specs.pkl")
    with open(results_dir / pkl_filename, 'wb') as f:
        pickle.dump(ang_spec_list, f)


def run_experiment_1(exp_id, results_dir, mc_seed=None):
    """
    Experiment 1: Localize one source while varying the SNR.
    
    - Source: one source (speech)
    - DOAs: chosen at random from a defined DOA grid
    - SNR: from -30 to 30 dB
    """
    # Hyperparameter space
    n_sources_choice = [1]
    source_type_choices = ['speech']
    snr_choices = np.arange(-15, 24, 3).tolist()
    noise_type_choices = ['awgn', 'alpha-0.8']
    sound_duration_choices = [0.1, 1.0]
    rt60_choices = [0.0, 0.123, 0.273]
    if mc_seed is None:
        monte_carlo_run_choices = np.arange(10).tolist()  # multiple runs per setting
    else:
        monte_carlo_run_choices = [mc_seed]

    sv_normalization = True

    # Create the grid of experiment settings
    data_settings = list(itertools.product(
        n_sources_choice, source_type_choices, sound_duration_choices,
        snr_choices, noise_type_choices, rt60_choices, monte_carlo_run_choices
    ))

    # Steering vector model configurations
    sv_methods_choices = ['gp-steerer']
    base_sv_models = [['ref', 8, 13], ['alg', 8, 13]]
    additional_sv_models = list(itertools.product(sv_methods_choices, sv_nObs_choice, sv_seed_choice))
    sv_model_choices = base_sv_models + additional_sv_models

    # Localization method settings
    min_freq, max_freq = 200, 4000
    freq_range = [min_freq, max_freq]
    ang_spec_methods_choices = [
        'alpha-1.2_beta-2_eps-1E-3_iter-500',
        'alpha-1.2_beta-1_eps-1E-3_iter-500',
        'alpha-1.2_beta-0_eps-1E-3_iter-500',
        'music_s-1',
        'srp_phat'
    ]

    # DOA grid settings
    doa_grid = np.arange(0, 360, 6)
    doa_grid_idx = np.concatenate([np.arange(0, 20), np.arange(60-20, 60)])

    # Load previously saved results (if any)
    csv_filename = f"experiment_results_exp-{exp_id}_run-{mc_seed}.csv"
    results_df = load_results_csv(results_dir, csv_filename)
    ang_spec_list = load_results_pkl(results_dir, csv_filename.replace(".csv", "_with_ang_specs.pkl"))
    counter_exp = 0

    for setting in tqdm(data_settings, desc="Scene settings"):
        n_sources, source_type, sound_duration, snr, noise_type, rt60, mc_seed = setting
        np.random.seed(mc_seed)
        src_doas_idx = np.random.choice(doa_grid_idx, n_sources, replace=False)
        frame_id = f"nSrc-{n_sources}_doas-{src_doas_idx}_type-{source_type}-duration-{sound_duration}-snr-{snr}_noise-{noise_type}_reverb-{rt60}_mc-{mc_seed}"
        date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        for loc_method in tqdm(ang_spec_methods_choices, leave=False, desc="Loc methods"):
            for sv_method, nObs, seed in sv_model_choices:
                method_id = f"{loc_method}_freqs-{freq_range}_{sv_method}_nObs-{nObs}_seed-{seed}_norm-{sv_normalization}"
                exp_name = f"exp-{exp_id}_{frame_id}_{method_id}"

                # Skip if experiment has already been run
                if not results_df.empty and results_df.query(f"exp_name == '{exp_name}'").shape[0] > 0:
                    logger.warning(f"Experiment {exp_name} already run")
                    continue

                # Run the experiment (assumes process_experiment is defined elsewhere)
                doas_est, doas_est_idx, error, doas_ref, doas_ref_idx, ang_spec, ang_spec_freqs, speech_files = process_experiment(
                    src_doas_idx, source_type, sound_duration, snr, noise_type, rt60,
                    loc_method, freq_range,
                    sv_method, seed, nObs, sv_normalization,
                    mc_seed=mc_seed,
                    exp_name=exp_name,
                )

                # Build results dictionary
                records = {
                    "exp_name": exp_name,
                    "time": date_str,
                    "record_id": [f's{i}' for i in range(n_sources)],
                    "num_srcs": n_sources,
                    "src_ids": list(range(n_sources)),
                    "doas_est_idx": doas_est_idx,
                    "doas_ref_idx": doas_ref_idx,
                    "doas_ref_az": [d[1] for d in doas_ref],
                    "doas_est_az": [d[1] for d in doas_est],
                    "doas_ref_el": [d[0] for d in doas_ref],
                    "doas_est_el": [d[0] for d in doas_est],
                    "errors": error,
                }
                df_results = pd.DataFrame(records)

                # Scene parameters
                scene_params = {
                    "record_id": [f's{i}' for i in range(n_sources)],
                    "speech_files": speech_files,
                    "frame_id": [frame_id] * n_sources,
                    "target_doa": doas_ref,
                    "n_sources": [n_sources] * n_sources,
                    "duration": [sound_duration] * n_sources,
                    "snr": [snr] * n_sources,
                    "noise_type": [noise_type] * n_sources,
                    "rt60": [rt60] * n_sources,
                    "mc_seed": [mc_seed] * n_sources,
                }
                df_scene = pd.DataFrame(scene_params)
                df_results_scene = pd.merge(df_results, df_scene, on='record_id')

                # Model parameters
                model_params = {
                    "record_id": [f's{i}' for i in range(n_sources)],
                    "method_id": [method_id] * n_sources,
                    "loc_method": [loc_method] * n_sources,
                    "freq_min": [freq_range[0]] * n_sources,
                    "freq_max": [freq_range[1]] * n_sources,
                    "sv_method": [sv_method] * n_sources,
                    "nObs": [nObs] * n_sources,
                    "seed": [seed] * n_sources,
                    "sv_normalization": [sv_normalization] * n_sources,
                }
                df_model = pd.DataFrame(model_params)
                df_full = pd.merge(df_results_scene, df_model, on='record_id')

                results_df = pd.concat([results_df, df_full], ignore_index=True)

                # Record angular spectrum results
                doa_freq_grid = np.stack(np.meshgrid(doa_grid, ang_spec_freqs, indexing='ij'), -1)
                ang_spec_dict = {
                    "frame_id": frame_id,
                    "method_id": method_id,
                    "loc_method": loc_method,
                    "ang_spec_shape": ang_spec.shape,
                    "doa_grid": doa_freq_grid[..., 0],
                    "freq_grid": doa_freq_grid[..., 1],
                    "ang_spec": ang_spec,
                }
                ang_spec_list.append(ang_spec_dict)

                counter_exp += 1
                if counter_exp % 20 == 0:
                    save_results(results_df, ang_spec_list, results_dir, csv_filename)

    save_results(results_df, ang_spec_list, results_dir, csv_filename)


def run_experiment_3(exp_id, results_dir, mc_seed=None):
    """
    Experiment 3: Localization robustness for multiple sources.
    
    - Sources: from 1 up to 6 (speech)
    - SNR: set to 20 dB
    """
    max_number_of_sources = 6
    n_sources_choice = np.arange(1, max_number_of_sources + 1).tolist()
    # n_sources_choice = [3]
    source_type_choices = ['speech']
    snr_choices = [20]
    noise_type_choices = ['awgn']
    sound_duration_choices = [1.0]
    rt60_choices = [0.0, 0.123, 0.273]
    if mc_seed is None:
        monte_carlo_run_choices = np.arange(10).tolist()
    else:
        monte_carlo_run_choices = [mc_seed]

    sv_normalization = True

    data_settings = list(itertools.product(
        n_sources_choice, source_type_choices, sound_duration_choices,
        snr_choices, noise_type_choices, rt60_choices, monte_carlo_run_choices
    ))

    base_sv_models = [['ref', 8, 13], ['alg', 8, 13]]
    additional_sv_models = list(itertools.product(['gp-steerer'], sv_nObs_choice, [666]))
    sv_model_choices = base_sv_models + additional_sv_models

    min_freq, max_freq = 200, 4000
    freq_range = [min_freq, max_freq]
    ang_spec_methods_choices = {
        'alpha-2.0_beta-2_eps-1E-3_iter-500',
        'alpha-1.2_beta-2_eps-1E-3_iter-500',
        'alpha-1.2_beta-1_eps-1E-3_iter-500',
        'alpha-1.2_beta-0_eps-1E-3_iter-500',
        'music_s-1',
        'music_s-2',
        'music_s-3',
        'music_s-4',
        'srp_phat',
    }

    doa_grid = np.arange(0, 360, 6)
    doa_grid_idx = np.concatenate([np.arange(0, 20), np.arange(60-20, 60)])

    csv_filename = f"experiment_results_exp-{exp_id}_run-{mc_seed}.csv"
    results_df = load_results_csv(results_dir, csv_filename)
    ang_spec_list = load_results_pkl(results_dir, csv_filename.replace(".csv", "_with_ang_specs.pkl"))
    
    counter_exp = 0

    for setting in tqdm(data_settings, desc="Scene settings"):
        n_sources, source_type, sound_duration, snr, noise_type, rt60, mc_seed = setting
        np.random.seed(mc_seed)
        src_doas_idx = np.random.choice(doa_grid_idx, n_sources, replace=False)
        frame_id = f"nSrc-{n_sources}_doas-{src_doas_idx}_type-{source_type}-duration-{sound_duration}-snr-{snr}_noise-{noise_type}_reverb-{rt60}_mc-{mc_seed}"
        date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        for loc_method in tqdm(ang_spec_methods_choices, leave=False, desc="Loc methods"):
            for sv_method, nObs, seed in sv_model_choices:
                method_id = f"{loc_method}_freqs-{freq_range}_{sv_method}_nObs-{nObs}_seed-{seed}_norm-{sv_normalization}"
                exp_name = f"exp-{exp_id}_{frame_id}_{method_id}"

                if not results_df.empty and results_df.query(f"exp_name == '{exp_name}'").shape[0] > 0:
                    logger.warning(f"Experiment {exp_name} already run")
                    continue

                doas_est, doas_est_idx, error, doas_ref, doas_ref_idx, ang_spec, ang_spec_freqs, speech_files = process_experiment(
                    src_doas_idx, source_type, sound_duration, snr, noise_type, rt60,
                    loc_method, freq_range,
                    sv_method, seed, nObs, sv_normalization,
                    mc_seed=mc_seed,
                    exp_name=exp_name,
                )

                records = {
                    "exp_name": exp_name,
                    "time": date_str,
                    "record_id": [f's{i}' for i in range(n_sources)],
                    "num_srcs": n_sources,
                    "src_ids": list(range(n_sources)),
                    "doas_est_idx": doas_est_idx,
                    "doas_ref_idx": doas_ref_idx,
                    "doas_ref_az": [d[1] for d in doas_ref],
                    "doas_est_az": [d[1] for d in doas_est],
                    "doas_ref_el": [d[0] for d in doas_ref],
                    "doas_est_el": [d[0] for d in doas_est],
                    "errors": error,
                }
                df_results = pd.DataFrame(records)

                scene_params = {
                    "record_id": [f's{i}' for i in range(n_sources)],
                    "speech_files": speech_files,
                    "frame_id": [frame_id] * n_sources,
                    "target_doa": doas_ref,
                    "n_sources": [n_sources] * n_sources,
                    "duration": [sound_duration] * n_sources,
                    "snr": [snr] * n_sources,
                    "noise_type": [noise_type] * n_sources,
                    "rt60": [rt60] * n_sources,
                    "mc_seed": [mc_seed] * n_sources,
                }
                df_scene = pd.DataFrame(scene_params)
                df_results_scene = pd.merge(df_results, df_scene, on='record_id')

                model_params = {
                    "record_id": [f's{i}' for i in range(n_sources)],
                    "method_id": [method_id] * n_sources,
                    "loc_method": [loc_method] * n_sources,
                    "freq_min": [freq_range[0]] * n_sources,
                    "freq_max": [freq_range[1]] * n_sources,
                    "sv_method": [sv_method] * n_sources,
                    "nObs": [nObs] * n_sources,
                    "seed": [seed] * n_sources,
                    "sv_normalization": [sv_normalization] * n_sources,
                }
                df_model = pd.DataFrame(model_params)
                df_full = pd.merge(df_results_scene, df_model, on='record_id')

                results_df = pd.concat([results_df, df_full], ignore_index=True)

                doa_freq_grid = np.stack(np.meshgrid(doa_grid, ang_spec_freqs, indexing='ij'), -1)
                ang_spec_dict = {
                    "frame_id": frame_id,
                    "method_id": method_id,
                    "loc_method": loc_method,
                    "ang_spec_shape": ang_spec.shape,
                    "doa_grid": doa_freq_grid[..., 0],
                    "freq_grid": doa_freq_grid[..., 1],
                    "ang_spec": ang_spec,
                }
                ang_spec_list.append(ang_spec_dict)

                counter_exp += 1
                if counter_exp % 20 == 0:
                    save_results(results_df, ang_spec_list, results_dir, csv_filename)

    if not results_df.empty:
        save_results(results_df, ang_spec_list, results_dir, csv_filename)


def run_experiment_4(exp_id, results_dir, mc_seed=None):
    raise NotImplementedError("Experiment 4 is the same as Experiment 3.")


def run_experiment_5(exp_id, results_dir, mc_seed=None):
    
    n_sources_choice = [3]
    source_type_choices = [f'alpha-{a}' for a in ['0.1', '0.4', '0.8', '1.2', '1.6', '2.0']]
    snr_choices = [20]
    noise_type_choices = ['awgn']
    sound_duration_choices = [1.0]
    rt60_choices = [0.123]
    
    if mc_seed is None:
        monte_carlo_run_choices = np.arange(10).tolist()
    else:
        monte_carlo_run_choices = [mc_seed]

    sv_normalization = True

    data_settings = list(itertools.product(
        n_sources_choice, source_type_choices, sound_duration_choices,
        snr_choices, noise_type_choices, rt60_choices, monte_carlo_run_choices
    ))

    base_sv_models = [['ref', 8, 13], ['alg', 8, 13]]
    additional_sv_models = list(itertools.product(['gp-steerer'], sv_nObs_choice, [666]))
    sv_model_choices = base_sv_models + additional_sv_models

    min_freq, max_freq = 200, 4000
    freq_range = [min_freq, max_freq]
    ang_spec_methods_choices = {
        'alpha-2.0_beta-2_eps-1E-3_iter-500',
        'alpha-2.0_beta-1_eps-1E-3_iter-500',
        'alpha-1.6_beta-1_eps-1E-3_iter-500',
        'alpha-1.2_beta-1_eps-1E-3_iter-500',
        'alpha-0.8_beta-1_eps-1E-3_iter-500',
        'alpha-0.4_beta-1_eps-1E-3_iter-500',
        'alpha-0.1_beta-1_eps-1E-3_iter-500',
        'music_s-3',
        'srp_phat',
    }

    doa_grid = np.arange(0, 360, 6)
    doa_grid_idx = np.concatenate([np.arange(0, 20), np.arange(60-20, 60)])

    csv_filename = f"experiment_results_exp-{exp_id}_run-{mc_seed}.csv"
    results_df = load_results_csv(results_dir, csv_filename)
    ang_spec_list = load_results_pkl(results_dir, csv_filename.replace(".csv", "_with_ang_specs.pkl"))
    
    counter_exp = 0

    for setting in tqdm(data_settings, desc="Scene settings"):
        n_sources, source_type, sound_duration, snr, noise_type, rt60, mc_seed = setting
        np.random.seed(mc_seed)
        src_doas_idx = np.random.choice(doa_grid_idx, n_sources, replace=False)
        frame_id = f"nSrc-{n_sources}_doas-{src_doas_idx}_type-{source_type}-duration-{sound_duration}-snr-{snr}_noise-{noise_type}_reverb-{rt60}_mc-{mc_seed}"
        date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        for loc_method in tqdm(ang_spec_methods_choices, leave=False, desc="Loc methods"):
            for sv_method, nObs, seed in sv_model_choices:
                method_id = f"{loc_method}_freqs-{freq_range}_{sv_method}_nObs-{nObs}_seed-{seed}_norm-{sv_normalization}"
                exp_name = f"exp-{exp_id}_{frame_id}_{method_id}"

                if not results_df.empty and results_df.query(f"exp_name == '{exp_name}'").shape[0] > 0:
                    logger.warning(f"Experiment {exp_name} already run")
                    continue

                doas_est, doas_est_idx, error, doas_ref, doas_ref_idx, ang_spec, ang_spec_freqs, speech_files = process_experiment(
                    src_doas_idx, source_type, sound_duration, snr, noise_type, rt60,
                    loc_method, freq_range,
                    sv_method, seed, nObs, sv_normalization,
                    mc_seed=mc_seed,
                    exp_name=exp_name,
                )

                records = {
                    "exp_name": exp_name,
                    "time": date_str,
                    "record_id": [f's{i}' for i in range(n_sources)],
                    "num_srcs": n_sources,
                    "src_ids": list(range(n_sources)),
                    "doas_est_idx": doas_est_idx,
                    "doas_ref_idx": doas_ref_idx,
                    "doas_ref_az": [d[1] for d in doas_ref],
                    "doas_est_az": [d[1] for d in doas_est],
                    "doas_ref_el": [d[0] for d in doas_ref],
                    "doas_est_el": [d[0] for d in doas_est],
                    "errors": error,
                }
                df_results = pd.DataFrame(records)

                scene_params = {
                    "record_id": [f's{i}' for i in range(n_sources)],
                    "speech_files": speech_files,
                    "frame_id": [frame_id] * n_sources,
                    "source_type": [source_type] * n_sources,
                    "target_doa": doas_ref,
                    "n_sources": [n_sources] * n_sources,
                    "duration": [sound_duration] * n_sources,
                    "snr": [snr] * n_sources,
                    "noise_type": [noise_type] * n_sources,
                    "rt60": [rt60] * n_sources,
                    "mc_seed": [mc_seed] * n_sources,
                }
                df_scene = pd.DataFrame(scene_params)
                df_results_scene = pd.merge(df_results, df_scene, on='record_id')

                model_params = {
                    "record_id": [f's{i}' for i in range(n_sources)],
                    "method_id": [method_id] * n_sources,
                    "loc_method": [loc_method] * n_sources,
                    "freq_min": [freq_range[0]] * n_sources,
                    "freq_max": [freq_range[1]] * n_sources,
                    "sv_method": [sv_method] * n_sources,
                    "nObs": [nObs] * n_sources,
                    "seed": [seed] * n_sources,
                    "sv_normalization": [sv_normalization] * n_sources,
                }
                df_model = pd.DataFrame(model_params)
                df_full = pd.merge(df_results_scene, df_model, on='record_id')

                results_df = pd.concat([results_df, df_full], ignore_index=True)

                doa_freq_grid = np.stack(np.meshgrid(doa_grid, ang_spec_freqs, indexing='ij'), -1)
                ang_spec_dict = {
                    "frame_id": frame_id,
                    "method_id": method_id,
                    "loc_method": loc_method,
                    "ang_spec_shape": ang_spec.shape,
                    "doa_grid": doa_freq_grid[..., 0],
                    "freq_grid": doa_freq_grid[..., 1],
                    "ang_spec": ang_spec,
                }
                ang_spec_list.append(ang_spec_dict)

                counter_exp += 1
                if counter_exp % 20 == 0:
                    save_results(results_df, ang_spec_list, results_dir, csv_filename)

    if not results_df.empty:
        save_results(results_df, ang_spec_list, results_dir, csv_filename)
       
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=int, default=None, help="Experiment case id (e.g., 1, 3, etc.)")
    # Optionally, add extra arguments (e.g., --mc_seed) if you want to vary seeds in parallel
    parser.add_argument("--mc_seed", type=int, default=1, help="Optional Monte Carlo seed override")
    parser.add_argument("--results_dir", type=Path, default="./results", help="Results directory")
    args = parser.parse_args()
    exp_id = args.exp_id

    # Setup directories (assumes figure_dir, output_dir, results_dir are defined globally)
    global figure_dir, output_dir, results_dir
    figure_dir = setup_directory(figure_dir, exp_id)
    output_dir = setup_directory(output_dir, exp_id)
    results_dir = args.results_dir

    # Optionally, if a Monte Carlo seed override is given, you could filter the data grid
    if args.mc_seed is not None:
        # For example, wrap your process_experiment calls with a seed override if desired.
        logger.info(f"Using Monte Carlo seed override: {args.mc_seed}")

    if exp_id == 1:
        run_experiment_1(exp_id, results_dir, args.mc_seed)
        
    elif exp_id == 3:
        run_experiment_3(exp_id, results_dir, args.mc_seed)
        
    elif exp_id == 4:
        run_experiment_4(exp_id, results_dir, args.mc_seed)
        
    elif exp_id == 5:
        run_experiment_5(exp_id, results_dir, args.mc_seed)
        
    else:
        # Fallback: a single experiment run with fixed parameters.
        src_doas = [5, 40]
        source_type = 'speech'
        sound_duration = 0.5
        snr = -5
        noise_type = 'awgn'
        rt60 = False
        loc_method = 'music'
        sv_method = 'gp-steerer'
        nObs = 8
        seed = 13
        sv_normalization = True
        process_experiment(src_doas, source_type, sound_duration, snr, noise_type, rt60,
                           loc_method, sv_method, seed, nObs, sv_normalization)

if __name__ == "__main__":
    main()
