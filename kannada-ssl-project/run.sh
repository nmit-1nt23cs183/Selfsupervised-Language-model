#!/bin/bash
# Kannada SSL — Quick Start Script

echo "========================================"
echo "  Kannada SSL — Starting Web App"
echo "========================================"

# Check Python
if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found. Please install Python 3.10+"
    exit 1
fi

# Create directories
mkdir -p data/dataset_kannada uploads checkpoints

# Check requirements
echo "Checking requirements..."
pip install -r requirements.txt -q

# Run the app
echo ""
echo "Starting Flask server on http://localhost:5000"
echo "Press Ctrl+C to stop"
echo ""
python app.py
