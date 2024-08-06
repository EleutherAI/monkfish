#!/usr/bin/env bash

# Set up script to exit on any errors.
set -e

# Miniconda installation path
MINICONDA_PATH="$HOME/miniconda"

# Environment setup
ENV_NAME="monkfish"
PYTHON_VERSION="3.10"
MY_PROJECT_DIR="/runtime/monkfish"

# Check if Miniconda is already installed
if [ ! -d "$MINICONDA_PATH" ]; then
    echo "Downloading Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $MINICONDA_PATH
    echo "Miniconda installed."
else
    echo "Miniconda is already installed."
fi

# Add Conda to PATH and initialize Conda
eval "$($MINICONDA_PATH/bin/conda shell.bash hook)"
conda init

# Update Conda if necessary
echo "Updating Conda..."
conda update -y conda

# Check if the environment already exists
if conda info --envs | grep "^$ENV_NAME\s" > /dev/null; then
    echo "Environment $ENV_NAME already exists."
else
    # Create a Conda environment if it doesn't exist
    echo "Creating Conda environment: $ENV_NAME with Python $PYTHON_VERSION"
    conda create -y -n $ENV_NAME python=$PYTHON_VERSION
    echo "Environment created."
fi

# Activate the environment
echo "Activating environment: $ENV_NAME"
conda activate $ENV_NAME

# Install project dependencies using pip
echo "Installing project dependencies from $MY_PROJECT_DIR"
pip install 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -e $MY_PROJECT_DIR[test]

echo "Setup completed."
