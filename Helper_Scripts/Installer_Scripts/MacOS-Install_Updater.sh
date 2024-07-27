#!/bin/bash

# Interactive TLDW macOS Installer and Updater Script

# Set up logging
log_file="$(dirname "$0")/tldw_install_log.txt"
log() {
    echo "$(date): $1" >> "$log_file"
}

log "Starting TLDW installation/update process for macOS"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to ask for permission to install
ask_permission() {
    read -p "Would you like to install $1? (y/n): " choice
    case "$choice" in
        y|Y ) return 0;;
        n|N ) return 1;;
        * ) echo "Invalid input. Please enter 'y' or 'n'."; ask_permission "$1";;
    esac
}

cleanup() {
    log "Performing cleanup"
    # Deactivate virtual environment
    deactivate
}

# Function to install Homebrew
install_homebrew() {
    if ask_permission "Homebrew"; then
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    else
        echo "Homebrew installation skipped. Some features may not work without it."
        log "Homebrew installation skipped by user"
    fi
}

# Function to install packages using Homebrew
install_package() {
    if ! command_exists brew; then
        echo "Homebrew is required to install $1."
        install_homebrew
    fi
    if command_exists brew; then
        if ask_permission "$1"; then
            brew install "$1"
        else
            echo "$1 installation skipped. Some features may not work without it."
            log "$1 installation skipped by user"
        fi
    fi
}

# Check and install Python3
if ! command_exists python3; then
    echo "Python3 is required for this installation."
    install_package python3
fi

# Check and install git
if ! command_exists git; then
    echo "Git is required for this installation."
    install_package git
fi

# Check and install ffmpeg
if ! command_exists ffmpeg; then
    install_package ffmpeg
fi

install_dir="$(dirname "$0")/tldw"

# Check if this is an update or new installation
if [ -d "$install_dir" ]; then
    read -p "TLDW directory found. Do you want to update? (y/n): " update_choice
    if [[ $update_choice == "y" || $update_choice == "Y" ]]; then
        update
    else
        fresh_install
    fi
else
    fresh_install
fi

cleanup
log "Installation/Update process completed"
echo "Installation/Update completed successfully!"
echo "To activate the virtual environment in the future, run: source $install_dir/venv/bin/activate"
echo "To start using TLDW, please refer to the project documentation."

# Functions

update() {
    log "Updating existing installation"
    cd "$install_dir" || exit
    git fetch
    old_version=$(git rev-parse HEAD)
    new_version=$(git rev-parse @{u})
    if [ "$old_version" == "$new_version" ]; then
        echo "TLDW is already up to date."
        log "TLDW is already up to date"
        exit 0
    fi
    echo "Current version: $old_version"
    echo "New version: $new_version"
    read -p "Do you want to proceed with the update? (y/n): " confirm_update
    if [[ $confirm_update == "y" || $confirm_update == "Y" ]]; then
        log "Creating backup"
        cp -R "$install_dir" "${install_dir}_backup_$(date +%Y%m%d)"
        if ! git pull; then
            log "Git pull failed"
            echo "Error: Git pull failed. Please check your internet connection and try again."
            exit 1
        fi
    else
        log "Update cancelled by user"
        echo "Update cancelled."
        exit 0
    fi
    setup_environment
}

fresh_install() {
    log "Starting fresh installation"
    # Prompt for GPU installation
    read -p "Do you want to install with GPU support? (y/n): " gpu_support
    if [[ $gpu_support == "y" || $gpu_support == "Y" ]]; then
        echo "Note: GPU support on macOS is limited. Defaulting to CPU installation."
        gpu_choice="cpu"
    else
        gpu_choice="cpu"
    fi

    # Save GPU choice
    mkdir -p "$install_dir"
    echo "$gpu_choice" > "$install_dir/gpu_choice.txt"

    # Clone the repository
    git clone https://github.com/rmusser01/tldw "$install_dir"
    cd "$install_dir" || exit

    setup_environment
}

setup_environment() {
    log "Setting up environment"
    # Create and activate virtual environment (if it doesn't exist)
    if [ ! -d "venv" ]; then
        python3 -m venv ./venv
    fi
    source ./venv/bin/activate

    # Upgrade pip and install wheel
    pip install --upgrade pip wheel

    # Install PyTorch (CPU version for macOS)
    pip install torch torchvision torchaudio

    # Update config.txt to use CPU
    sed -i '' 's/cuda/cpu/' config.txt

    # Install other requirements
    pip install -r requirements.txt
}

