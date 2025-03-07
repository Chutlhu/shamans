#!/bin/bash
# parallel_experiments.sh
# This script launches several runs of the experiment concurrently,
# with user-defined maximum parallel processes and maximum number of seeds.

EXP_ID=3

# Default maximum number of parallel processes and maximum seeds.
MAX_PROCS=${1:-4}    # Defaults to 4 if not provided.
MAX_SEEDS=${2:-10}   # Defaults to 10 if not provided.

# Function to wait until the number of running jobs is less than MAX_PROCS.
wait_for_slot() {
    while [ "$(jobs -r | wc -l)" -ge "$MAX_PROCS" ]; do
        sleep 1
    done
}

echo "Starting experiment 3 with up to $MAX_PROCS parallel processes and $MAX_SEEDS seeds..."

# Launch experiment 3 with seeds from 0 to MAX_SEEDS-1.
for (( seed=0; seed<MAX_SEEDS; seed++ )); do
    wait_for_slot
    python eusipco2025_experiments.py --exp_id "$EXP_ID" --mc_seed "$seed" &
done

# Wait for all background processes from experiment 3 to finish.
wait
echo "Experiment 3 completed."
