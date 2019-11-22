#!/bin/bash

## Create directories if they don't exist
curr_dir=$(pwd)
sub_dir=${curr_dir}/Sub_Scripts
logs_dir=${curr_dir}/Logs
err_dir=${curr_dir}/Errors
output=${curr_dir}/Output

mkdir ${sub_dir} 2> /dev/null
mkdir ${logs_dir} 2> /dev/null
mkdir ${err_dir} 2> /dev/null
mkdir ${output} 2> /dev/null

# Number of CPUs
ncpus=12

## Counter
p=101

## Main loop for creation of submission scripts

for file in $@; do

# Remove the submission script if it already exists
rm -rf ${sub_dir}/${p}_pls_100_iters_05_11_2019_16_18_11.sh 2> /dev/null

# Write a new submission script
cat <<EOF >> ${sub_dir}/${p}_pls_100_iters_05_11_2019_16_18_11.sh
#PBS -P 11001558
#PBS -j oe
#PBS -N py_QSRR_IC_${p}
#PBS -q normal
#PBS -l select=1:ncpus=${ncpus}:mem=5GB
#PBS -l walltime=12:00:00
#PBS -e ${err_dir}/${p}_pls_100_iters_05_11_2019_16_18_11.err
#PBS -o ${logs_dir}/${p}_pls_100_iters_05_11_2019_16_18_11.log

cd \$PBS_O_WORKDIR;
np=\$(cat \${PBS_NODEFILE} | wc -l);

# Load python module
module load anaconda/3

# Activate virtual environment
source activate ${curr_dir}/venv/

# Call Python ($ncpus * 2 because HT is activated)
python ${file} 100 ${p} $(($ncpus*2)) pls no > ${output}/${p}_pls_100_iters_05_11_2019_16_18_11.out
EOF

# Submit the script
qsub ${sub_dir}/${p}_pls_100_iters_05_11_2019_16_18_11.sh

# Increment counter
((p++))

done
