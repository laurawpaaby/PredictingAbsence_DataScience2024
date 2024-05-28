sudo apt-get update
sudo apt-get install python3-venv

# install requirements.txt for project
python -m pip install ipykernel
# Usage: source setup.sh

# create virtual environment
python3 -m venv venv_dat_sci24

# activate virtual environment
source ./venv_dat_sci24/bin/activate

python -m ipykernel install --user --name=venv_dat_sci24

# Install requirements
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# INFO message
echo "Successfully installed requirements.txt"

# Deactivate virtual environment
deactivate