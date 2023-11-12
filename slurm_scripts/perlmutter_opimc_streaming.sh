#!/bin/bash

#SBATCH -A m1641
#SBATCH -C cpu
#SBATCH -t 00:02:00
#SBATCH -q debug
#SBATCH -N 5 
#SBATCH --ntasks-per-node=1
#SBATCH -J m5_opimc_streaming_github_IC
#SBATCH -o /global/homes/w/wadecap/results/jobs/testing_leveled/github/m5_opimc_streaming_github_IC.o
#SBATCH -e /global/homes/w/wadecap/results/jobs/testing_leveled/github/m5_opimc_streaming_github_IC.e

# module use /global/common/software/m3169/perlmutter/modulefiles
module use /global/common/software/m3169/perlmutter/modulefiles

#OpenMP settings:
export OMP_NUM_THREADS=64
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#module load PrgEnv-cray
# module load python/3.9-anaconda-2021.11
module load gcc/11.2.0
module load cmake/3.24.3
module load cray-mpich
module load cray-libsci
#module load openmpi
#module load cudatoolkit/11.0

srun -n 5 ./build/release/tools/mpi-greedimm -i /global/cfs/cdirs/m1641/network-data/Binaries/github_IC_binary.txt -w -k 100 -p -d IC -e 0.13 -o /global/homes/w/wadecap/results/jobs/testing_leveled/github/m5_opimc_streaming_github_IC.json --reload-binary -u --opimc=0 --epsilon-2=0.077
