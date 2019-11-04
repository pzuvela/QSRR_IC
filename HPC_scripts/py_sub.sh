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
p=1

## Main loop for creation of submission scripts

for file in $@; do

# Remove the submission script if it already exists
rm -rf ${sub_dir}/${p}_xgb_100_iters_04_11_2019_18_10_08.sh 2> /dev/null

# Write a new submission script
cat <<EOF >> ${sub_dir}/${p}_xgb_100_iters_04_11_2019_18_10_08.sh
#PBS -P py_QSRR_IC
#PBS -j oe
#PBS -N py_QSRR_IC_${p}
#PBS -q parallel12
#PBS -l select=1:ncpus=${ncpus}:mem=40GB
#PBS -e ${err_dir}/${p}_xgb_100_iters_04_11_2019_18_10_08.err
#PBS -o ${logs_dir}/${p}_xgb_100_iters_04_11_2019_18_10_08.log

cd \$PBS_O_WORKDIR;
np=\$(cat \${PBS_NODEFILE} | wc -l);

# Load miniconda module
source /etc/profile.d/rec_modules.sh
module load miniconda

# Source the .bashrc file
source ~/.bashrc

# Activate virtual environment
conda activate QSRR_in_IC

# Call Python
python3.6 ${file} 100 ${p} $ncpus xgb no > ${output}/${p}_xgb_100_iters_04_11_2019_18_10_08.out
EOF

# Submit the script
qsub ${sub_dir}/${p}_xgb_100_iters_04_11_2019_18_10_08.sh

# Increment counter
((p++))

done
