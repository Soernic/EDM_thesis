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
#BSUB -W 24:00
### request 5GB of system-memory
#BSUB -R "rusage[mem=2GB]"

### -- set the job Name --
#BSUB -J prop_mu_edge
### -- Specify the output and error file. %J is the job-id --
#BSUB -o batch/logs/prop_mu_edge%J.out
#BSUB -e batch/logs/prop_mu_edge%J.err

# -- end of LSF options --

nvidia-smi

source ~/miniconda3/bin/activate torch_env

python3 train_prop_pred.py --name prop_mu_edge_100 --target_idx 0 --cutoff_data 10 --p 1 --num_rounds 3 --state_dim 128 --edge_dim 20 --epochs 800

python3 train_prop_pred.py --name prop_mu_edge_90 --target_idx 0 --cutoff_data 5.0991 --p 1 --num_rounds 3 --state_dim 128 --edge_dim 20 --epochs 800

python3 train_prop_pred.py --name prop_mu_edge_80 --target_idx 0 --cutoff_data 4.3927 --p 1 --num_rounds 3 --state_dim 128 --edge_dim 20 --epochs 800

python3 train_prop_pred.py --name prop_mu_edge_70 --target_idx 0 --cutoff_data 3.9266 --p 1 --num_rounds 3 --state_dim 128 --edge_dim 20 --epochs 800

python3 train_prop_pred.py --name prop_mu_edge_60 --target_idx 0 --cutoff_data 3.4910 --p 1 --num_rounds 3 --state_dim 128 --edge_dim 20 --epochs 800

