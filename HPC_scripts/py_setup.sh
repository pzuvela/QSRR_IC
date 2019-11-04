# Load Python 3.6.4
module load python3.6.4

# Create virtual environment
python3.6 -m venv --system-site-packages ./venv

# Activate virtual environment
source ./venv/bin/activate

# Install packages
easy_install-3.6 Pandas
easy_install-3.6 numpy
easy_install-3.6 scikit-learn
easy_install-3.6 scipy
easy_install-3.6 --upgrade scipy


