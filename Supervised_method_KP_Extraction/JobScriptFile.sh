#!/bin/bash



#SBATCH -J job_KP_30
#SBATCH --mail-user=<liuxinyu19930328@gmail.com>
#SBATCH --mail-type=ALL
#SBATCH -e /home/xl23lico/somewhere/Job_Name.err.%j
#SBATCH -o /home/xl23lico/somewhere/Job_Name.out.%j
#SBATCH --mem-per-cpu=250
#SBATCH –t 05:00:00
#SBATCH –n 4
#SBATCH -C nvd4
#SBATCH –-account=kurs00015
#SBATCH –-partition=kurs00015
#SBATCH –-reservation=kurs00015

echo "This is Job $SLURM_JOB_ID"
module load gcc
module load openmpi/gcc intel cuda python/3
cd /home/xl23lico/mastertheses
python3 test01.py
