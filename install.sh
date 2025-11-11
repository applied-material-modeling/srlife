#!/bin/bash
set -e

# Parse arguments
BUILD_MOOSE=false
if [[ "$1" == "--with-moose" ]]; then
    BUILD_MOOSE=true
    echo "Installing srlife with MOOSE..."
    # Initialize MOOSE submodule
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
# ensure conda shell functions are available in this script
eval "$(conda shell.bash hook)"

# Create the environment only if it doesn't already exist
if conda env list | awk '!/^#/ && NF>0 {print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists; skipping creation."
else
    conda create -n "$ENV_NAME" moose-dev=2025.09.18=mpich
    #TODO:install seacas --- maybe add this to requirements.txt later
    conda install moose-seacas
fi


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
    
    echo "Installing Solid Mechanics module..."
    cd $SRLIFE_DIR/moose/modules/solid_mechanics/
    make -j$MOOSE_JOBS
fi

echo "Installation complete!"
echo ""
if [[ "$BUILD_MOOSE" == "true" ]]; then
    echo "To use add the following to your rc file:"
    echo "  conda activate $ENV_NAME"
    echo "  export PYTHONPATH=$SRLIFE_DIR/"
    echo "  export MOOSE_THM=$SRLIFE_DIR/moose/modules/thermal_hydraulics/thermal_hydraulics-opt"
    echo "  export MOOSE_SM=$SRLIFE_DIR/moose/modules/thermal_hydraulics/solid_mechanics-opt"
    echo "  export MOOSE_MPI=$(which mpirun)"
else
    echo "To use:"
    echo "  conda activate $ENV_NAME"
fi