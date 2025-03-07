# Justfile

#   just parallel-experiments MAX_PROCS=6 MAX_SEEDS=12
run_exp_parallel EXP_ID MAX_PROCS MAX_SEEDS:
	@echo "Starting experiment ${EXP_ID} with up to ${MAX_PROCS} parallel processes and ${MAX_SEEDS} seeds..."
	# Launch experiment 1 for seeds 0 through MAX_SEEDS-1.
	for seed in $(seq 0 $(($$MAX_SEEDS - 1))); do
	  # Wait until the number of running jobs is less than MAX_PROCS.
	  while [ "$$(jobs -r | wc -l)" -ge "${MAX_PROCS}" ]; do
	    sleep 1;
	  done;
	  # Launch the experiment in background.
	  python my_experiment.py --exp_id 1 --mc_seed "$$seed" &
	done
	wait
	@echo "Experiment 1 completed."