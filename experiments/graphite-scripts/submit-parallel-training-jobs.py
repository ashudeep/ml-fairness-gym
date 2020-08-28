#!/usr/bin/env python

import os
import datetime

lambda_cvars = [0.0, 0.5, 1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0, -1.0, -10.0]
learning_rates = [0.001, 0.0001, 0.0005]
date_time = datetime.datetime.now().strftime('%m%d-%H:%M:%S.%f')[:-4]

for lambda_cvar in lambda_cvars:
    for lr in learning_rates:

        all_params_string = "onehot_lambda_cvar_"+str(lambda_cvar)+"_lr_"+str(lr)+"_"+date_time
        job_file = "launched_jobs/%s.job" % all_params_string

        with open(job_file, 'w') as fh:
            fh.writelines("#!/bin/bash\n")
            fh.writelines("#SBATCH -J %s.job\n" % all_params_string)
            fh.writelines(
                "#SBATCH -o /share/thorsten/as3354/saferecs/slurm_outputs/%s.out\n" % all_params_string)
            fh.writelines(
                "#SBATCH -e /share/thorsten/as3354/saferecs/slurm_outputs/%s.err\n" % all_params_string)
            fh.writelines("#SBATCH --tasks-per-node=1\n")
            fh.writelines("#SBATCH --cpus-per-task=1 \n")
            fh.writelines("#SBATCH --cpus-per-task=1 \n")
            fh.writelines("#SBATCH --get-user-env \n")
            fh.writelines("#SBATCH -t 72:00:00 \n")
            fh.writelines("#SBATCH --mem-per-cpu=5000\n")
            fh.writelines("#SBATCH --mem=5000M\n")
            fh.writelines("#SBATCH --partition=mpi-cpus\n")
            fh.writelines("python $HOME/ml-fairness-gym/experiments/movielens_recs_main.py --lambda_cvar %s --lr %s --expt_name_suffix %s\n" %
                          (lambda_cvar, lr, all_params_string))

        os.system("sbatch --requeue %s" % job_file)
