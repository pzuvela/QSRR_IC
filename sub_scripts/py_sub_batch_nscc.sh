## Usage:
# sh py_sub_batch.sh ncpus counter_init max_iter method max_files 

if [[ $# -eq 0 ]] ; then
    echo 'Usage: sh py_sub_batch.sh ncpus counter_init max_iter method max_files'
    exit 0
fi

## Input arguments
# ncups - number of CPUs
# counter_init - initial value of the counter for the submission script loop
# max_iter - maximum interations
# method - method (gbr, or xgb)
# max_files - maximum number of jobs to submit at once

# Date & time
dt=$(date '+%d_%m_%Y_%H_%M_%S')

# Replace the number of CPU cores
sed -i "16s@.*@ncpus=$1@" py_sub_nscc.sh

# Replace the initial counter
sed -i "19s@.*@p=$2@" py_sub_nscc.sh

# Replace lines with submission script filenames
sed -i "26s@.*@rm -rf \${sub_dir}/\${p}_$4_$3_iters_$dt.sh 2> /dev/null@" py_sub_nscc.sh
sed -i "29s@.*@cat <<EOF >> \${sub_dir}/\${p}_$4_$3_iters_$dt.sh@" py_sub_nscc.sh

# Replace the lines with log & error file names
sed -i "36s@.*@#PBS -e \${err_dir}/\${p}_$4_$3_iters_$dt.err@" py_sub_nscc.sh
sed -i "37s@.*@#PBS -o \${logs_dir}/\${p}_$4_$3_iters_$dt.log@" py_sub_nscc.sh

# Replace the line with the python "call"
sed -i "49s@.*@python \${file} $3 \${p} \$(( \$ncpus*2 )) $4 no \> \${output}/\${p}_$4_$3_iters_$dt.out@" py_sub_nscc.sh
# sed -i "49s@.*@python \${file} $3 \${p} \$ncpus $4 no" py_sub_nscc.sh

# Replace the line to submit the job sh script
sed -i "53s@.*@qsub \${sub_dir}/\${p}_$4_$3_iters_$dt.sh@" py_sub_nscc.sh

# String with the python filename repeated max_files times
run_string=$(for (( i=1; i<= $5; i++)); do printf "main.py "; done)

# Run the submission script for max_files times
sh py_sub_nscc.sh $run_string
