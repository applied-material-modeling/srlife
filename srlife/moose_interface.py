import os
import subprocess
from pathlib import Path
import numpy as np
import netCDF4 as nc
import re
import pyhit
from srlife import receiver 
from srlife.receiver import make_moose_hit_vector


def find_pin_coords(mesh_file, tol=1e-6):
        # Two diametrically-opposite outer-radius nodes on the bottom face.
        # This helps pin the bottom face without restricting the axial expansion
        with nc.Dataset(mesh_file, "r") as exo:
                x = np.array(exo.variables["coordx"][:])
                y = np.array(exo.variables["coordy"][:])
                z = np.array(exo.variables["coordz"][:])
        bot_mask = np.abs(z - z.min()) < tol
        x_bot, y_bot = x[bot_mask], y[bot_mask]
        cx, cy = x_bot.mean(), y_bot.mean()
        dist = np.hypot(x_bot - cx, y_bot - cy)
        outer_mask = np.abs(dist - dist.max()) < tol * 10
        x_outer, y_outer = x_bot[outer_mask], y_bot[outer_mask]
        xa, ya = x_outer[0], y_outer[0]
        idx_b = np.argmax(np.hypot(x_outer - xa, y_outer - ya))
        xb, yb = x_outer[idx_b], y_outer[idx_b]
        z_bottom = float(z.min())
        return (float(xa), float(ya), z_bottom), (float(xb), float(yb), z_bottom)


PANEL_BLOCK_RE = re.compile(r"panel_(\d+)/")


def panels_in_thm_exodus(exo_path):
        # Read panel names in the THM output
        with nc.Dataset(exo_path, "r") as exo:
                raw = exo.variables["eb_names"][:]
        names = ["".join(c.decode() for c in row if c).strip() for row in raw]
        panels = set()
        for name in names:
                m = PANEL_BLOCK_RE.match(name)
                if m:
                        panels.add(int(m.group(1)))
        return sorted(panels)


def read_time_axis(thm_exodus: Path):
        with nc.Dataset(thm_exodus, "r") as exo:
                return list(exo.variables["time_whole"][:])

def discover_tubes(panel: int, out_dir: Path):
        # Find tube IDs by globbing panel_{panel}_tube_*_in.e in out_dir

        pattern = f"panel_{panel}_tube_*_in.e"
        files = sorted(out_dir.glob(pattern))
        if not files:
                raise FileNotFoundError(f"No tube meshes matching {pattern} in {out_dir}")
        ids = []
        for f in files:
                m = re.search(r"_tube_(\d+)_in\.e$", f.name)
                if m:
                        ids.append(int(m.group(1)))
        return sorted(ids)


def create_moose_sm_inputs(moose_thm_filename, out_dir=None):
        """
        Creates MOOSE Solid Mechanics input files for each receiver panel based on the THM results.
        Expects the THM Exodus files to be named {moose_thm_filename}_flowpath_{fp}_exo.e

        Args:
                moose_thm_filename (String): base filename of the MOOSE THM Exodus outputs
                out_dir (String, optional): directory where THM exodus files are located. This is also where the structural input files will be written. Defaults to current working directory.
        """
        out_dir = Path(out_dir) if out_dir is not None else Path.cwd()
        written = []
        for fp in (0, 1):
                exo_path = out_dir / f"{moose_thm_filename}_flowpath_{fp}_exo.e"
                times = read_time_axis(exo_path)
                for panel in panels_in_thm_exodus(exo_path):
                        tubes = discover_tubes(panel, out_dir)
                        root, i_name = build_structural_input(
                                panel, tubes, fp, times, out_dir, moose_thm_filename
                        )
                        input_path = out_dir / i_name
                        pyhit.write(str(input_path), root)
                        written.append(input_path)
        return written

def run_moose_sm_model(moose_input_filename):
        """
        Runs the MOOSE SolidMechanics module 
        Needs environment variables MOOSE_MPI, MOOSE_NPROCS, and DEER to be set.
        TODO: Write a small NEML app and replace the usage of full Deer here

        Args:
          moose_input_filename (String): filename to call moose with

        """
        try:
            print("Running MOOSE!")
            mpirun = os.environ.get("MOOSE_MPI")
            nprocs = os.environ.get("MOOSE_NPROCS")
            # TODO: Write a small NEML app and replace the usage of full Deer here
            moose_sm = os.environ.get("DEER") 
            argv = [mpirun, "-n", nprocs, moose_sm, "-i", moose_input_filename]
            result = subprocess.run(argv,
                                    check=True, capture_output=False, text=True)
        except subprocess.CalledProcessError as e:
            print(f"MOOSE returned error {e.returncode}")
            print(f"stderr: {e.stderr}")

def build_structural_input(panel: int, tubes: list, flowpath: int, times, out_dir: Path, moose_thm_filename: str):
        # Build the structural solution file for receiver panel by panel

        thm_file = f"{moose_thm_filename}_flowpath_{flowpath}_exo.e"
        output_base = f"panel_{panel}_struct_from_fp{flowpath}"
        i_name = f"moose_structural_panel_{panel}_from_fp{flowpath}.i"
        dt = float(times[1] - times[0]) if len(times) > 1 else 1.0
        end_time = float(times[-1])

        root = pyhit.Node(parent=None, hitnode=None, offset=None)

        mesh = root.append("Mesh", construct_side_list_from_node_list="true")
        for t in tubes:
                rmin_id = 100 + t
                blk_id = 10 + t
                mesh.append(f"tube_{t}_mesh",
                            type="FileMeshGenerator",
                            file=f"panel_{panel}_tube_{t}_in.e")
                mesh.append(f"tube_{t}_renum",
                            type="RenameBoundaryGenerator",
                            input=f"tube_{t}_mesh",
                            old_boundary=make_moose_hit_vector(["rmin"]),
                            new_boundary=make_moose_hit_vector([rmin_id]))
                mesh.append(f"tube_{t}_rename",
                            type="RenameBoundaryGenerator",
                            input=f"tube_{t}_renum",
                            old_boundary=make_moose_hit_vector([rmin_id]),
                            new_boundary=make_moose_hit_vector([f"rmin_tube_{t}"]))
                mesh.append(f"tube_{t}_blk_renum",
                            type="RenameBlockGenerator",
                            input=f"tube_{t}_rename",
                            old_block=make_moose_hit_vector(["0"]),
                            new_block=make_moose_hit_vector([blk_id]))
                mesh.append(f"tube_{t}_blk_rename",
                            type="RenameBlockGenerator",
                            input=f"tube_{t}_blk_renum",
                            old_block=make_moose_hit_vector([blk_id]),
                            new_block=make_moose_hit_vector([f"tube_{t}"]))
        mesh.append("combined",
                    type="CombinerGenerator",
                    inputs=make_moose_hit_vector([f"tube_{t}_blk_rename" for t in tubes]))

        prev_input = "combined"
        for t in tubes:
                coord_a, coord_b = find_pin_coords(out_dir / f"panel_{panel}_tube_{t}_in.e")
                mesh.append(f"pin_xy_tube_{t}",
                            type="ExtraNodesetGenerator",
                            input=prev_input,
                            new_boundary=make_moose_hit_vector([f"pin_xy_tube_{t}"]),
                            coord=make_moose_hit_vector(
                                [f"{coord_a[0]:.16g}", f"{coord_a[1]:.16g}", f"{coord_a[2]:.16g}"]))
                prev_input = f"pin_xy_tube_{t}"
                mesh.append(f"pin_y_tube_{t}",
                            type="ExtraNodesetGenerator",
                            input=prev_input,
                            new_boundary=make_moose_hit_vector([f"pin_y_tube_{t}"]),
                            coord=make_moose_hit_vector(
                                [f"{coord_b[0]:.16g}", f"{coord_b[1]:.16g}", f"{coord_b[2]:.16g}"]))
                prev_input = f"pin_y_tube_{t}"

        root.append("GlobalParams",
                    displacements=make_moose_hit_vector(["disp_x", "disp_y", "disp_z"]))

        aux_vars = root.append("AuxVariables")
        aux_vars.append("temp", order="FIRST", family="LAGRANGE")

        user_objs = root.append("UserObjects")
        user_objs.append("tube_temp_soln",
                         type="SolutionUserObject",
                         mesh=thm_file,
                         system_variables="T_solid",
                         execute_on=make_moose_hit_vector(["initial", "timestep_begin"]))

        functions = root.append("Functions")
        for t in tubes:
                functions.append(f"p_from_thm_tube_{t}",
                                 type="PiecewiseMultilinear",
                                 data_file=f"panel_{panel}_fch_tube_{t}_p.dat")

        physics = root.append("Physics")
        sm = physics.append("SolidMechanics")
        qs = sm.append("QuasiStatic")
        qs.append("all",
                  strain="SMALL",
                  new_system="true",
                  eigenstrain_names="eigenstrain",
                  add_variables="true",
                  generate_output=make_moose_hit_vector([
                      "cauchy_stress_xx", "cauchy_stress_yy", "cauchy_stress_zz",
                      "cauchy_stress_yz", "cauchy_stress_xz", "cauchy_stress_xy",
                      "mechanical_strain_xx", "mechanical_strain_yy", "mechanical_strain_zz",
                      "mechanical_strain_yz", "mechanical_strain_xz", "mechanical_strain_xy",
                  ]))

        aux_kernels = root.append("AuxKernels")
        aux_kernels.append("temp_from_thm",
                           type="SolutionAux",
                           variable="temp",
                           solution="tube_temp_soln",
                           from_variable="T_solid",
                           execute_on=make_moose_hit_vector(["initial", "timestep_begin"]))

        bcs = root.append("BCs")
        bcs.append("z_disp",
                   type="DirichletBC",
                   variable="disp_z",
                   boundary=make_moose_hit_vector(["bot"]),
                   value=0.0)
        for t in tubes:
                bcs.append(f"pin_x_tube_{t}",
                           type="DirichletBC",
                           variable="disp_x",
                           boundary=make_moose_hit_vector([f"pin_xy_tube_{t}"]),
                           value=0.0)
                bcs.append(f"pin_y_tube_{t}",
                           type="DirichletBC",
                           variable="disp_y",
                           boundary=make_moose_hit_vector([f"pin_xy_tube_{t}"]),
                           value=0.0)
                bcs.append(f"pin_y2_tube_{t}",
                           type="DirichletBC",
                           variable="disp_y",
                           boundary=make_moose_hit_vector([f"pin_y_tube_{t}"]),
                           value=0.0)
        for t in tubes:
                bcs.append(f"inner_pressure_x_tube_{t}",
                           type="Pressure",
                           variable="disp_x",
                           function=f"p_from_thm_tube_{t}",
                           boundary=f"rmin_tube_{t}")
                bcs.append(f"inner_pressure_y_tube_{t}",
                           type="Pressure",
                           variable="disp_y",
                           function=f"p_from_thm_tube_{t}",
                           boundary=f"rmin_tube_{t}")

        constraints = root.append("Constraints")
        constraints.append("ev_z",
                           type="EqualValueBoundaryConstraint",
                           variable="disp_z",
                           secondary=make_moose_hit_vector(["top"]),
                           penalty=1e7)

        materials = root.append("Materials")
        materials.append("stress",
                         type="CauchyStressFromNEML",
                         database="../srlife/srlife/data/deformation/SiC.xml",
                         model="cares",
                         temperature="temp")
        materials.append("thermal_strain",
                         type="ComputeThermalExpansionEigenstrainNEML",
                         database="../srlife/srlife/data/deformation/SiC.xml",
                         model="cares",
                         temperature="temp",
                         stress_free_temperature=300.0,
                         eigenstrain_name="eigenstrain")

        precond = root.append("Preconditioning")
        precond.append("pc", type="SMP", full="true")

        root.append("Executioner",
                    type="Transient",
                    solve_type="NEWTON",
                    l_max_its=100,
                    l_tol=1e-10,
                    nl_max_its=15,
                    nl_rel_tol=1e-6,
                    nl_abs_tol=1e-8,
                    automatic_scaling="true",
                    compute_scaling_once="false",
                    resid_vs_jac_scaling_param=0.5,
                    petsc_options=make_moose_hit_vector([
                        "-snes_converged_reason", "-ksp_converged_reason",
                        "-snes_linesearch_monitor"]),
                    petsc_options_iname=make_moose_hit_vector([
                        "-pc_type", "-pc_factor_mat_solver_package"]),
                    petsc_options_value=make_moose_hit_vector(["lu", "superlu_dist"]),
                    line_search="none",
                    dt=dt,
                    start_time=0.0,
                    end_time=end_time)

        outputs = root.append("Outputs", print_linear_residuals="false")
        outputs.append("exodus_out",
                       type="Exodus",
                       file_base=output_base,
                       sync_times=make_moose_hit_vector([f"{float(t):.16g}" for t in times]),
                       sync_only="true")
        outputs.append("csv_out", type="CSV", file_base=output_base)

        return root, i_name