#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task={n_cpu}
#SBATCH --mem={memory}
#SBATCH --mail-type=END
#SBATCH --mail-user=aritraghosh.iem@gmail.com
#SBATCH --partition={gpu}
#SBATCH --constraint=""
#SBATCH --time={time}
#SBATCH -o /work/arighosh_umass_edu/difa/slurm_outputs/%j.out
#SBATCH -e /work/arighosh_umass_edu/difa/slurm_errors/%j.out

cd /work/arighosh_umass_edu/difa/
source /work/arighosh_umass_edu/venv/difa/bin/activate





python src/train.py\
    --data {data}\
    --problem {problem}\
    --lr {lr}\
    --policy_lr {policy_lr}\
    --policy_base_lr {policy_base_lr}\
    --seed {seed}\
    --imputation_model {imputation_model}\
    --iters {iters}\
    --pretrain_iters {pretrain_iters}\
    --workers 2\
    --batch_size {batch_size}\
    --n_features {n_features}\
    --grad_norm {grad_norm}\
    --weight "{weight}"\
    {fixed_params}\
    --name "${SLURM_JOB_ID}" --nodes "${SLURM_JOB_NODELIST}" --slurm_partition "${SLURM_JOB_PARTITION}"
