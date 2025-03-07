"""
This file perform sound source localization using neural steerer
and other interpolation methods.

Author: Diego DI CARLO
Created: 2025-02-19
Modified: 2025-02-19
"""


import argparse
from pathlib import Path
import pickle

from nsteerer_utils import load_ground_truth_data, load_upsampled_svects

import numpy as np

from einops import rearrange
import itertools


expertiment_folder = Path("/home/dicarlod/Documents/Code/NeuralSteerer/")
path_to_speech_data = expertiment_folder / "data/SmallTimit"
path_to_best_models = expertiment_folder / "results_tmp/talsp_good_baselines/best_model_interps.txt"

results_dir = expertiment_folder / "results_tmp/eusipco2025/"
results_dir.mkdir(parents=True, exist_ok=True)
output_dir = results_dir / "models/"
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


sv_methods_choices = eusipco_names_to_exp_name.keys()
sv_nObs_choice = [8, 16, 32, 64, 128]
sv_seed_choice = [13, 42, 666]


parser = argparse.ArgumentParser(description='Sound Source Localization with NSteerer and AlphaStable')
parser.add_argument('--sv-method', type=str, help='Steering vector interpolation method', choices=['nf', 'nf-gw' 'nn', 'sp', 'pinn', 'gp-steerer'])
parser.add_argument('--nObs', type=int, help='Number of observations used to fit the sv model', choices=[8, 16, 32, 64, 128])
parser.add_argument('--seed', type=int, help='Random seed used for training the interpolation methods', choices=[13, 42, 666])

def export_model_to_tensor(
    sv_method: str,
    seed: int,
    nObs: int,
    ):
    
    # Load the ground truth data
    data_dict = load_ground_truth_data(path_to_best_models) # dict(x, y, svect)
    
    coords = data_dict['x']           # [nFreq x nEle x nAzi x nChan x 6]
    svect_alg = data_dict['svects']   # [nFreq x nEle x nAzi x nChan x 1]
    svect_ref = data_dict['y']        # [nFreq x nEle x nAzi x nChan x 1]
    
    print("Problem dimensions:")
    print("coords.shape: ", coords.shape)
    print("svect_alg.shape: ", svect_alg.shape)
    print("svect_ref.shape: ", svect_ref.shape)
    nFreq, nAzi, nEle, nChan, nVars = coords.shape
    
    coords = rearrange(coords, 'f el az ch d -> f (el az) ch d')
    svect_alg = rearrange(svect_alg, 'f el az ch d -> f (el az) (ch d)')
    svect_ref = rearrange(svect_ref, 'f el az ch d -> f (el az) (ch d)')
    
    if sv_method == 'ref':
        svect_est = svect_ref
    elif sv_method == 'alg':
        svect_est = svect_alg
    else:
        sv_method_exp_name = eusipco_names_to_exp_name[sv_method]
        results_dict = load_upsampled_svects(path_to_best_models, nObs, seed, sv_method_exp_name)
        svect_est = results_dict['y_pred']  # [nFreq x (nEle x nAzi) x nChan x 1]
        # model_fn = results_dict['model_fn']
    print("svect_est.shape: ", svect_est.shape)
    svect_est = svect_est.reshape(svect_ref.shape)
    discrete_model_dict = {
        'format' : 'freq,doa,chan',
        'coords': coords,
        'svects': svect_est,
        'nObs': nObs,
        'seed': seed,
        'sv_method': sv_method,
        'nfft' : data_dict['params']['nfft'],
        'fs' : data_dict['params']['fs'],
    }
    return discrete_model_dict


if __name__ == "__main__":
    args = parser.parse_args()
    if not any(vars(args).values()):

        # Steering vector models
        # compile the list of models combining the method with nObs and seed
        sv_model_choices = [['ref', 8, 13], ['alg', 8, 13]]
        sv_model_choices += list(itertools.product(sv_methods_choices, sv_nObs_choice, sv_seed_choice))
            
        # run for all the other steering vectors
        for sv_model_name in sv_model_choices:
            
        
            sv_method, nObs, seed = sv_model_name
            print(f"Exporting model for {sv_method} with nObs={nObs} and seed={seed}")
            
            model_save_name = f"{sv_method}_nObs-{nObs}_seed-{seed}.pkl"
            # if already exists, skip
            if (output_dir / model_save_name).exists():
                print(f"Model {model_save_name} already exists. Skipping")
                continue
            
            # Create a string for the file name
            discrete_model_dict = export_model_to_tensor(sv_method, seed, nObs)
            
            # Save the model as pickle
            model_file = output_dir / model_save_name
            with open(model_file, 'wb') as f:
                pickle.dump(discrete_model_dict, f)
            print(f"Saved model to {model_file}")
      
    else:
        
        sv_method = args.sv_method
        nObs = args.nObs
        seed = args.seed

        # Create a string for the file name
        discrete_model_dict = export_model_to_tensor(sv_method, seed, nObs)