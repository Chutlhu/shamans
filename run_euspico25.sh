conda activate python39cuda12

./main.py \
    --results ../results_tmp/talsp_good_baselines/models/ \
    --best-models ../results_tmp/talsp_good_baselines/best_model_interps.txt \
    --nobs 8 --seed 13 --method gpdkl-sph-shlow \
    --eval-method gpdkl-sph-shlow \
    --speech1 /path/to/speech1.wav --speech2 /path/to/speech2.wav \
    --doas 5,55 --snr -5 --no-display