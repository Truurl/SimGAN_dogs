#!/bin/bash -l

## simgan-training
#SBATCH --job-name=simgan-training
#SBATCH --time=48:00:00

#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgdyplomanci5-gpu-a100

#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu
#SBATCH --mem=60G

#SBATCH -C memfs
#SBATCH --signal=B:TERM@300

module load CUDA/11.3.1 -q
module load GCC/10.3.0 -q
module load OpenMPI/4.1.1 -q
module load matplotlib/3.4.2 -q
module load PyTorch/1.12.1-CUDA-11.3.1 -q

mkdir $MEMFS/dataset
# cp /net/tscratch/people/plgtrurl/datasets/eyes/*.dat $MEMFS/dataset
cp /net/tscratch/people/plgtrurl/datasets/dogs/*.hdf5 $MEMFS/dataset

cd ~/SimGAN/SimGAN_pytorch
# pip install torchvision==0.13.1 h5py wandb==0.13.3 --upgrade
# pip install 

wandb login
export WANDB_DIR=$SCRATCH/wandb_logs


echo 
echo 
echo 
echo 
echo 
echo  "######################### TRAINING STARTS #########################"

python3 main.py ${SLURM_JOB_NAME}
