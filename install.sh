#!/bin/bash
# Installation script for required Python packages

# Update pip to the latest version
python3 -m pip install --upgrade pip

# Install required Python packages
pip install -r requirements.txt > install.txt

# Download NLTK data
python3 -c "import nltk; nltk.download('punkt')"

# Indicate installation success
echo "All dependencies installed successfully."