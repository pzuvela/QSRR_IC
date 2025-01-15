#!/usr/bin/bash

# Usage:
# py_setup.sh user_id work_dir

user_id=$1; work_dir=$2


# Show usage if no input given
if [[ $# -eq 0 ]]; then
     echo "Usage: py_setup.sh user_id work_dir"
     exit 0
fi

## Using Anaconda environment
# Load Anaconda module
module load anaconda/3

# Create conda environment
echo "Creating environment in ${work_dir}..."
conda create --prefix /home/users/nus/${user_id}/${work_dir}/venv python=3.6

# Activate environment
echo "Activating environment /home/users/nus/${user_id}/${work_dir}/venv..."
source activate /home/users/nus/${user_id}/${work_dir}/venv

# Install packages
echo "Installing numpy, scipy, scikit-learn, and xgboost..."
read -n 1 -s -r -p "Press any key to continue..."; echo ""
conda install -c anaconda numpy
read -n 1 -s -r -p "Press any key to continue..."; echo ""
conda install -c anaconda scipy
read -n 1 -s -r -p "Press any key to continue..."; echo ""
conda install -c anaconda pandas
read -n 1 -s -r -p "Press any key to continue..."; echo ""
conda install -c anaconda scikit-learn
read -n 1 -s -r -p "Press any key to continue..."; echo ""
conda install -c anaconda py-xgboost

