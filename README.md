# ShaMaNS: Sound Source Localization with Spatial Measure and Neural Steerer

## Set up the environment

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Download the models for [this Drive folder](https://drive.google.com/drive/folders/1TNzmW4RWqV4RMq-mVSH2hSedDv3HOtsU?usp=drive_link) and put them in the `data/models/` folder

## Run the experiments

1. Experiment 1: one source, varying SNR
```bash
python main.py --exp-id 1
```

2. Experiment 2: two sources, SNR = 0 dB
```bash
python main.py --exp-id 2
```