from pathlib import Path
import yaml
import pickle

# DNN model stuff
from ml_collections import ConfigDict
from nsteerer.models.nf import NSteerer
from nsteerer.models.pinn import PINNSteerer

from flax.training import checkpoints

# math stuff
from einops import rearrange

def load_config_and_data(model_dir):
    print("# Load config")
    with open(model_dir / "config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        config = ConfigDict(config)

    model_config = config.model
    data_config = config.data
    
    if config['model.regressor'] == 'nf':
        method = 'nf-svect' if config['model.add_svect'] else 'nf'
        print(f"Method: {method}")

    print("# Load data")
    src_atf = model_dir / "Device_ATF_downsampled.pkl"
    with open(src_atf, "rb") as file:
        dset_dict = pickle.load(file)

    assert data_config["fs"] == dset_dict["params"]["fs"]
    nfft = dset_dict["params"]["nfft"]
    fs = dset_dict["params"]["fs"]

    return model_config, data_config, dset_dict, nfft, fs


def load_model(model_config, data_config, nfft, seed, model_dir):
    print("# Load best model")
    if model_config.regressor in ["pinn", "nf", "gpdkl"]:
        if model_config.regressor in ["nf", "gpdkl"]:
            model = NSteerer(model_config=model_config, data_config=data_config, nfft=nfft, seed=seed)
        elif model_config.regressor == "pinn":
            model = PINNSteerer(model_config=model_config, data_config=data_config, nfft=nfft, seed=seed)

        ckpt_dict = checkpoints.restore_checkpoint(ckpt_dir=model_dir / "checkpoints", target=None)
        state = ckpt_dict["state"]
        if model_config.regressor in ["nf", "gpdkl"]:
            model.get_model(state["params"], collocation_pts=ckpt_dict["collocation_pts"])
        elif model_config.regressor == "pinn":
            model.get_model(state["params"])

    elif model_config.regressor == "gpnll":
        model = NSteerer(model_config=model_config, data_config=data_config, nfft=nfft, seed=seed, workdir=model_dir, viz_data=dset_dict["full"])
        ckpt_dict = checkpoints.restore_checkpoint(ckpt_dir=model_dir / "checkpoints", target=None)
        state = ckpt_dict["state"]
        model.get_model(state["params"], collocation_pts=ckpt_dict["collocation_pts"])

    elif model_config.regressor in ["nn", "sp", "sh"]:
        with open(model_dir / "model_function_bin.pkl", "rb") as file:
            model = pickle.load(file)
            
    else:
        raise ValueError(f"Unknown regressor, got {model_config.regressor}")

    return model


def find_model_dir(path_to_best_models, nObs, seed, method):
    model_dir = []
    
    if "nf-subfreq-gw" in method:
        method="nf-subfreq"
        hash_config = "9a521b2597684d40b566607adc663b1aaa9a89ca24157075f6dfdd86f92d3bcf"
    else:
        hash_config = "6183d8dde4ee7eda41c12a08fbe022ade4a711b403df305da8bf0190c791f031"
    
    with open(path_to_best_models, 'r') as f:
        for _, line in enumerate(f):
            curr_model_dir = Path(line.strip())
            method_ = curr_model_dir.parent.name.split("_config_")[1].split("_")[0]
            nObs_ = int(curr_model_dir.parent.name.split("_nGrid-")[1].split("_")[0])
            seed_ = int(curr_model_dir.parent.name.split("_seed-")[1].split("_")[0])

            if nObs_ == nObs and seed == seed_ and method == method_:
                model_dir.append(curr_model_dir)


    if len(model_dir) == 0:
        raise ValueError("No model found with the specified parameters")

    if method == "nf-subfreq" and len(model_dir) > 1:
        # get the model with the correct hash
        for curr_model_dir in model_dir:
            if hash_config in curr_model_dir.parent.name:
                model_dir = [curr_model_dir]
                break
        
    if len(model_dir) > 1:
        raise ValueError("More than one model found with the specified parameters")

    return model_dir[0]


def load_ground_truth_data(path_to_best_models):
    model_dir = find_model_dir(path_to_best_models, 8, 13, 'nf-subfreq')
    model_config, data_config, dset_dict, nfft, fs = load_config_and_data(model_dir)
    model = load_model(model_config, data_config, nfft, 13, model_dir)

    x, y, shapes = model.prepare_data(dset_dict['full'], stage="test", return_original_shape=True)
    # x_obs, y_obs, shapes_obs = model.prepare_data(dset_dict['train_valid'], stage="test", return_original_shape=True)

    pcoords = model.model.proj(x)
    pcoords = pcoords.reshape(-1, pcoords.shape[-1])
    svects = model.model.compute_svect(pcoords[:, :1], pcoords[:, 1:4], pcoords[:, 4:], model.model.fs, model.model.c, model.model.offset_sample)
    svects = svects.reshape(shapes["y"])

    data = {'x': x.reshape(*shapes['x']), 
            'y': y.reshape(*shapes['y']),
            "svects": svects,
            "params": {"nfft": nfft, 'fs' : fs}
           }
    return data


def load_upsampled_svects(path_to_best_models, nObs, seed, method):
    model_dir = find_model_dir(path_to_best_models, nObs, seed, method)
    model_config, data_config, dset_dict, nfft, fs = load_config_and_data(model_dir)
    model = load_model(model_config, data_config, nfft, seed, model_dir)

    x, y, shapes = model.prepare_data(dset_dict['full'], stage="test", return_original_shape=True)
    y_pred = model.predict(x)

    results = {
        "model_fn": model,
        "model_config": model_config,
        "data_config": data_config,
        "method": method,
        "nObs": nObs,
        "seed": seed,
        "y_pred": y_pred,
    }

    return results
