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

    # Get nemlapp and NEML
    git submodule update --init --recursive nemlapp
else
    echo "Installing srlife only..."
fi

# Get repo root
SRLIFE_DIR="$(pwd)"
ENV_NAME="srlifeMoose"
MOOSE_JOBS=32


# ensure conda shell functions are available in this script
eval "$(conda shell.bash hook)"

# Create the environment only if it doesn't already exist
if conda env list | awk '!/^#/ && NF>0 {print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists; skipping creation."
else
    # Create conda environment
    echo "Creating conda environment..."
    
    conda create -n "$ENV_NAME" moose-dev=2025.09.18=mpich
    #TODO:install seacas --- maybe add this to requirements.txt later
    conda install moose-seacas
fi


# Activate environment
conda activate $ENV_NAME

# This is coming from moose I think. The conda env is adding a space after the -Wl, 
# which is causing problems for the linker. This sed command removes that space. Can be removed if fixed.
export LDFLAGS=$(echo "$LDFLAGS" | sed 's/^-Wl, /-Wl,-O2 /')

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install --user wheel
pip3 install -r requirements.txt




# Build MOOSE (if specified)
if [[ "$BUILD_MOOSE" == "true" ]]; then
    echo "Building with MOOSE..."
    cd $SRLIFE_DIR/moose/test
    make -j$MOOSE_JOBS
    
    # Set environment variables
    echo "Installing Thermohydraylics module..."
    cd $SRLIFE_DIR/moose/modules/thermal_hydraulics/
    make -j$MOOSE_JOBS
    
    echo "Building NEML..."
    # Clean any partial state from a previous failed in-source cmake run
    rm -rf $SRLIFE_DIR/nemlapp/neml/CMakeCache.txt \
           $SRLIFE_DIR/nemlapp/neml/CMakeFiles \
           $SRLIFE_DIR/nemlapp/neml/build
    # Out-of-source build, install into the NEML submodule prefix so libneml.so
    # lands at neml/lib/ where nemlapp's Makefile expects it.
    mkdir -p $SRLIFE_DIR/nemlapp/neml/build
    cd $SRLIFE_DIR/nemlapp/neml/build
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=$SRLIFE_DIR/nemlapp/neml \
          -DENABLE_OPENMP=OFF \
          ..
    make -j$MOOSE_JOBS
    make install

    echo "Building nemlapp..."
    cd $SRLIFE_DIR/nemlapp
    MOOSE_DIR=$SRLIFE_DIR/moose 
    make -j$MOOSE_JOBS
fi

echo "Installation complete!"
echo ""

echo "To use, add the following to your rc file:"
echo "  conda activate $ENV_NAME"
echo "  # This is an adhoc fix for a linker issue with the conda environment. Can be removed if fixed."
echo "  export LDFLAGS=\$(echo \"\$LDFLAGS\" | sed 's/^-Wl, /-Wl,-O2 /')"
echo "  export SRLIFE_DIR="$(pwd)""
echo "  export PYTHONPATH=$SRLIFE_DIR:\$PYTHONPATH"
echo "  export NEML_DIR=$SRLIFE_DIR/nemlapp/neml"
echo "  export LD_LIBRARY_PATH=$SRLIFE_DIR/nemlapp/neml/lib:\$LD_LIBRARY_PATH"
echo "  export NEMLAPP=$SRLIFE_DIR/nemlapp/nemlapp-opt"

if [[ "$BUILD_MOOSE" == "true" ]]; then
    echo "  export MOOSE_DIR=$SRLIFE_DIR/moose"
    echo "  export MOOSE_THM=$SRLIFE_DIR/moose/modules/thermal_hydraulics/thermal_hydraulics-opt"
    echo "  export MOOSE_MPI=\$(which mpirun)"
    echo "  export MOOSE_NPROCS=$MOOSE_JOBS"
fi