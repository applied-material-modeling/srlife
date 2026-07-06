"""Shared helper for invoking MOOSE executables via subprocess."""

import os
import subprocess


def run_moose(input_filename, executable_env_var="MOOSE_THM", mesh_only=False):
    """Run a MOOSE executable on the given input file.

    Reads MOOSE_MPI, MOOSE_NPROCS, and the executable path from environment
    variables. When mesh_only=True, runs the executable directly (no mpirun)
    and appends --mesh-only.

    Args:
        input_filename (str): path to MOOSE input (.i) file
        executable_env_var (str): env var holding moose executable path, 
                                  thermal_hydraulics-opt for THM and
                                  nemlapp-opt for structural solve
        mesh_only (bool): if True, run with --mesh-only and no mpirun
    """
    moose_exec = os.environ.get(executable_env_var, executable_env_var)
    if mesh_only:
        argv = [moose_exec, "-i", input_filename, "--mesh-only"]
        capture = True
    else:
        mpirun = os.environ.get("MOOSE_MPI")
        nprocs = os.environ.get("MOOSE_NPROCS")
        argv = [mpirun, "-n", nprocs, moose_exec, "-i", input_filename]
        capture = False
    try:
        print("Running MOOSE!")
        subprocess.run(argv, check=True, capture_output=capture, text=True)
    except subprocess.CalledProcessError as e:
        print(f"MOOSE returned error {e.returncode}")
        print(f"stderr: {e.stderr}")
