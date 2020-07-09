#!/usr/bin/env python

import os

lambda_cvars = [0.0, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 0.7, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, -1.0]
learning_rates = [0.001, 0.0001, 0.00001, 0.000001]

for lambda_cvar in lambda_cvars:
    for lr in learning_rates:

        all_params_string = "lambda_cvar_"+str(lambda_cvar)+"_lr_"+str(lr)
        job_file = "launched_jobs/%s.job" % all_params_string
        
        with open(job_file, 'w') as fh:
            fh.writelines("#!/bin/bash\n")
            fh.writelines("#SBATCH -J %s.job\n" % all_params_string)
            fh.writelines("#SBATCH -o /share/thorsten/as3354/saferecs/slurm_outputs/%j.out\n")
            fh.writelines("#SBATCH -e /share/thorsten/as3354/saferecs/slurm_outputs/%j.err\n")
            fh.writelines("#SBATCH --tasks-per-node=1\n")
            fh.writelines("#SBATCH --cpus-per-task=1 \n")
            fh.writelines("#SBATCH --cpus-per-task=1 \n")
            fh.writelines("#SBATCH --get-user-env \n")
            fh.writelines("#SBATCH -t 72:00:00 \n")
            fh.writelines("#SBATCH --mem-per-cpu=5000\n")
            fh.writelines("#SBATCH --mem=5000M\n")
            fh.writelines("#SBATCH --partition=mpi-cpus\n")
            fh.writelines("python $HOME/ml-fairness-gym/experiments/movielens_recs_main.py --lambda_cvar %s --lr %s\n" % (lambda_cvar, lr))

        os.system("sbatch --requeue %s" %job_file)