#!/bin/bash
# Setup script for LLM reproducibility experiments
# Creates virtual environment and installs dependencies

set -e  # Exit on error

echo "==========================================="
echo "LLM Reproducibility Setup"
echo "==========================================="

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "Python version: $PYTHON_VERSION"

# Create virtual environment
VENV_DIR="venv"
echo ""
echo "Creating virtual environment in '$VENV_DIR'..."
python3 -m venv $VENV_DIR

# Activate virtual environment
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "==========================================="
echo "✓ Setup completed successfully!"
echo "==========================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run experiments:"
echo "  jupyter notebook experiments.ipynb"
echo ""
echo "To deactivate when done:"
echo "  deactivate"
echo ""
