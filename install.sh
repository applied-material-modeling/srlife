#!/bin/bash
set -e

# Parse arguments
BUILD_MOOSE=false
if [[ "$1" == "--with-moose" ]]; then
    BUILD_MOOSE=true
    echo "Installing srlife with MOOSE..."
    # Initialize MOOSE submodule without recursion (avoids large media files)
    git submodule update --init moose
    cd moose
    git checkout master
    cd ../
else
    echo "Installing srlife only..."
fi

# Get repo root
SRLIFE_DIR="$(pwd)"
ENV_NAME="srlifeMoose"
MOOSE_JOBS=32

# Create conda environment
echo "Creating conda environment..."
conda create -y -n moose moose-dev=2025.09.18=mpich

# Activate environment
conda activate $ENV_NAME

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install --user wheel
pip3 install -r requirements.txt

# Build MOOSE (if specified)
if [[ "$BUILD_MOOSE" == "true" ]]; then
    echo "Building MOOSE..."
    cd $SRLIFE_DIR/moose/test
    make -j$MOOSE_JOBS
    
    # Set environment variables
    echo "Installing Thermohydraylics module..."
    cd $SRLIFE_DIR/moose/modules/thermal_hydraulics/
    make -j$MOOSE_JOBS
fi

echo "Installation complete!"
echo ""
if [[ "$BUILD_MOOSE" == "true" ]]; then
    echo "To use:"
    echo "  conda activate $ENV_NAME"
    echo "  export PYTHONPATH=$SRLIFE_DIR/"
    echo "  export MOOSE_THM=$SRLIFE_DIR/moose/modules/thermal_hydraulics/thermal_hydraulics-opt"
else
    echo "To use:"
    echo "  conda activate $ENV_NAME"
fi