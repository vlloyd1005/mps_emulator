#!/bin/bash
#SBATCH --job-name=mps_val
#SBATCH --output=/gpfs/projects/MirandaGroup/victoria/cocoa/Cocoa/mps_emu/out/validation_%x_%a_%A.txt
#SBATCH --time=4:00:00
#SBATCH -p short-40core	
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=9
#SBATCH --mail-type=ALL
#SBATCH --mail-user=victoria.lloyd@stonybrook.edu



echo "Available CPUs:  $SLURM_JOB_CPUS_PER_NODE"
# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1
module load slurm

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID
echo Number of task is $SLURM_NTASKS
echo Number of cpus per task is $SLURM_CPUS_PER_TASK

source ~/.bashrc
source /gpfs/projects/MirandaGroup/victoria/miniconda/etc/profile.d/conda.sh
conda init bash
conda deactivate
conda activate vic
# source start_cocoa.sh

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
srun python ./mps_emu/t_comp_val.py