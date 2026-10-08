#!/bin/bash
# Script to generate model_data for threeD-generation/obj and collect data

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Activate conda environment
echo "Activating conda environment: RoboTwin"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate RoboTwin

# Change to project directory
cd "${PROJECT_ROOT}"

# Run the Python script that does all the work
python threeD-generation/run_obj_data_collection.py "$@"
