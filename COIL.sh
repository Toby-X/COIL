#!/bin/bash
#SBATCH --account=stats
#SBATCH --job-name=COIL
#SBATCH -c 24
#SBATCH --time=48:00:00
#SBATCH -o /burg/home/zx2488/COIL/COIL.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zx2488@columbia.edu
#SBATCH --chdir=/burg/home/zx2488/COIL/
module load anaconda/3-2023.09
conda init bash
source ~/.bashrc
conda activate torchbase
python test.py