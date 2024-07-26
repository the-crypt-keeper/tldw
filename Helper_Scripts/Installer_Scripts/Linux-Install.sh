#!/bin/bash

# Linux TLDW Installer Script

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install packages based on the package manager
install_package() {
    if command_exists apt-get; then
        sudo apt-get install -y "$1"
    elif command_exists dnf; then
        sudo dnf install -y "$1"
    else
        echo "Unsupported package manager. Please install $1 manually."
        exit 1
    fi
}

# Check and install Python3
if ! command_exists python3; then
    echo "Python3 not found. Installing..."
    install_package python3
fi

# Check and install ffmpeg
if ! command_exists ffmpeg; then
    echo "ffmpeg not found. Installing..."
    install_package ffmpeg
fi

# Prompt for GPU installation
read -p "Do you want to install with GPU support? (y/n): " gpu_support
if [[ $gpu_support == "y" || $gpu_support == "Y" ]]; then
    echo "Please ensure your GPU drivers and CUDA are up to date."
    echo "Visit https://developer.nvidia.com/cuda-downloads for instructions."
    read -p "Press enter to continue when ready..."
fi

# Clone the repository
git clone https://github.com/rmusser01/tldw
cd tldw || echo "tldw directory not found. Now exiting..." && exit

# Create and activate virtual environment
python3 -m venv ./
source ./bin/activate

# Upgrade pip and install wheel
pip install --upgrade pip wheel

# Install PyTorch based on GPU support choice
if [[ $gpu_support == "y" || $gpu_support == "Y" ]]; then
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
else
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
    # Update config.txt to use CPU
    sed -i 's/cuda/cpu/' config.txt
fi

# Install other requirements
pip install -r requirements.txt

echo "Installation completed successfully!"
echo "To activate the virtual environment in the future and run tldw, use the commands:"
echo "        source ./bin/activate"
echo "        python3 summarize.py -gui"
echo "Now starting tldw..."
python3 summarize.py -gui -log DEBUG