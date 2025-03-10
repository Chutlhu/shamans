from invoke import task

@task
def run(c, exp_id, num_jobs, num_scenes):
    print("Welcome to Shamans: Sound Source Localization with Neural Steerr and Spatial Measure!")
    print(f"Running experiment {exp_id}...")
    c.run(f"./run_parallel_exp.sh {exp_id} {num_jobs} {num_scenes}")