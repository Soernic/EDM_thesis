#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100

### -- ask for number of cores (default: 1) --
#BSUB -n 4
#BSUB -R "span[hosts=1]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm -- maximum 24 hours for GPU-queues right now
#BSUB -W 10:00
### request 5GB of system-memory
#BSUB -R "rusage[mem=2GB]"

### -- set the job Name --
#BSUB -J prop11_cv
### -- Specify the output and error file. %J is the job-id --
#BSUB -o batch/logs/prop11_cv%J.out
#BSUB -e batch/logs/prop11_cv%J.err

# -- end of LSF options --

nvidia-smi

source ~/miniconda3/bin/activate torch_env

python3 train_prop_pred.py --name prop11_cv --target_idx 11 --p 1 --num_rounds 3 --state_dim 128 --edge_dim 20 --patience 5