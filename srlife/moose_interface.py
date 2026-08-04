"""
This module provides functions to interface between srlife and MOOSE.
It has functions to read MOOSE THM results, create MOOSE structural input files,
run the MOOSE structural model, and compute reliability from MOOSE structural outputs
using srlife's damage models.

"""

import os
import sys
import re
from pathlib import Path
import numpy as np
import pyhit  # pylint: disable=import-error,wrong-import-position
from srlife.receiver import make_moose_hit_vector
from srlife.interface import convert_m_to_mm
from srlife.moose_runner import run_moose

conda_env_dir = os.environ.get("CONDA_PREFIX")
ACCESS = os.getenv("ACCESS", f"{conda_env_dir}/seacas")
sys.path.append(os.path.join(ACCESS, "lib"))
import exodus as exo  # pylint: disable=import-error,wrong-import-position

SQRT2 = np.sqrt(2.0)
# Structural material and NEML model variant -> data/deformation/<name>.xml
MATERIAL_DIR = Path(__file__).resolve().parent / "data" / "deformation"
DEFAULT_MATERIAL = "SiC"
DEFAULT_MATERIAL_MODEL = "cares"


def find_pin_coords(mesh_file, tol=1e-6):
    """
    Two diametrically-opposite outer-radius nodes on the bottom face.
    This helps pin the bottom face without restricting the axial expansion

    Args:
        mesh_file: path to the tube mesh exodus file
        tol: tolerance for coordinate comparisons

    Returns:
        (xa, ya, za), (xb, yb, zb): coordinates of the two pin nodes.
    """
    model = exo.exodus(str(mesh_file), array_type="numpy")
    x, y, z = model.get_coords()
    model.close()
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
FCH_TUBE_BLOCK_RE = re.compile(r"panel_(\d+)/fch_tube_(\d+)$")


def panels_in_thm_exodus(exo_path):
    """
    Reads the THM exodus file and returns a sorted list of panel numbers present in the file.

    Args:
        exo_path: path to the THM exodus file
    Returns:
        A sorted list of panel numbers (integers) found in the exodus file."""
    model = exo.exodus(str(exo_path), array_type="numpy")
    names = [model.get_elem_blk_name(b) for b in model.get_elem_blk_ids()]
    model.close()
    panels = set()
    for name in names:
        m = PANEL_BLOCK_RE.match(name)
        if m:
            panels.add(int(m.group(1)))
    return sorted(panels)


def read_time_axis(thm_exodus: Path):
    """
    Reads the time axis from the THM exodus file.

    Args:
        thm_exodus: Path to the THM exodus file

    Returns:
        A list of time values.
    """
    model = exo.exodus(str(thm_exodus), array_type="numpy")
    times = list(model.get_times())
    model.close()
    return times


def discover_tubes(panel: int, out_dir: Path):
    """
    Find tube IDs by globbing panel_{panel}_tube_*_in.e in out_dir
    Args:
        panel: int, panel number
        out_dir: Path, directory to search for tube meshes
    Returns:
        A sorted list of tube IDs (integers) found in the directory.
    """
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


def extract_thm_pressures_to_dat(thm_exo_path, out_dir):
    """For each panel_X/fch_tube_Y block in the THM exodus, write
    panel_X_fch_tube_Y_p.dat (AXIS Z / AXIS T / DATA, pressure in MPa).

    Args:
       thm_exo_path: path to the THM exodus file
       out_dir: directory where the .dat files will be written
    """
    out_dir = Path(out_dir)
    model = exo.exodus(str(thm_exo_path), array_type="numpy")
    times = model.get_times()
    for blk_id in model.get_elem_blk_ids():
        blk_name = model.get_elem_blk_name(blk_id)
        m = FCH_TUBE_BLOCK_RE.match(blk_name)
        if not m:
            continue
        panel, tube = int(m.group(1)), int(m.group(2))
        conn, num_elem, num_nodes = model.get_elem_connectivity(blk_id)
        conn = np.array(conn, dtype=int).reshape((num_elem, num_nodes))
        z_vals = np.array(
            [
                np.mean([model.get_coord(nid)[2] for nid in elem_nodes])
                for elem_nodes in conn
            ]
        )
        order = np.argsort(z_vals)
        out_path = out_dir / f"panel_{panel}_fch_tube_{tube}_p.dat"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# Pressure from block {blk_name} variable p\n")
            f.write("AXIS Z\n")
            f.write(" ".join(f"{v:.16g}" for v in z_vals[order]) + "\n")
            f.write("AXIS T\n")
            f.write(" ".join(f"{t:.16g}" for t in times) + "\n")
            f.write("DATA\n")
            for step in range(len(times)):
                p_vals = model.get_variable_values(
                    "EX_ELEM_BLOCK", blk_id, "p", step + 1
                )
                f.write(
                    " ".join(f"{v:.16g}" for v in np.array(p_vals)[order] / 1e6) + "\n"
                )
    model.close()


def create_moose_sm_inputs(
    moose_thm_filename,
    material=DEFAULT_MATERIAL,
    material_model=DEFAULT_MATERIAL_MODEL,
    out_dir=None,
):
    """
    Creates MOOSE Solid Mechanics input files for each receiver panel based on the THM results.
    Expects the THM Exodus files to be named {moose_thm_filename}_flowpath_{fp}_exo.e

    Args:
        moose_thm_filename (String): base filename of the MOOSE THM Exodus outputs
        material (String, optional): name of the NEML deformation model file in
        data/deformation/ (without the .xml extension). Defaults to DEFAULT_MATERIAL.
        material_model (String, optional): NEML model variant within that file.
        Defaults to DEFAULT_MATERIAL_MODEL.
        out_dir (String, optional): directory where THM exodus files are located.
        This is also where the structural input files will be written. Defaults to
        current working directory.

    Returns:
        (input_paths, output_exodus_paths): input_paths are the .i files for moose sm
        output_exodus_paths are the .e files for compute_moose_reliability.
    """
    out_dir = Path(out_dir) if out_dir is not None else Path.cwd()
    input_paths = []
    output_exodus_paths = []
    for fp in (0, 1):
        exo_path = out_dir / f"{moose_thm_filename}_flowpath_{fp}_exo.e"
        times = read_time_axis(exo_path)
        extract_thm_pressures_to_dat(exo_path, out_dir)
        for panel in panels_in_thm_exodus(exo_path):
            tubes = discover_tubes(panel, out_dir)
            root, i_name = build_structural_input(
                panel,
                tubes,
                fp,
                times,
                out_dir,
                moose_thm_filename,
                material,
                material_model,
            )
            input_path = out_dir / i_name
            pyhit.write(str(input_path), root)
            input_paths.append(input_path)
            # build_structural_input writes the [Outputs/exodus_out] file_base
            # as panel_{P}_struct_from_fp{F}; MOOSE appends .e
            output_exodus_paths.append(out_dir / f"panel_{panel}_struct_from_fp{fp}.e")
    return input_paths, output_exodus_paths


# Disabling pylint warnings for local variables and statements. This is the main moose file writer
# function, its bound to be long and have many variables.
# pylint: disable=too-many-locals, too-many-statements
def build_structural_input(
    panel: int,
    tubes: list,
    flowpath: int,
    times,
    out_dir: Path,
    moose_thm_filename: str,
    material: str,
    material_model: str,
):
    """
    Build the structural solution file for receiver panel by panel.

    Args:
        panel: int, panel number
        tubes: list, list of tube IDs
        flowpath: int, flow path number
        times: list, time values
        out_dir: Path, directory where the structural input file will be written
        moose_thm_filename: str, base filename of the MOOSE THM Exodus outputs
        material: str, name of the NEML deformation model file in data/deformation/
        material_model: str, NEML model variant within that file

    Returns:
        root: pyhit.Node, used to write the MOOSE input file
        i_name: str, the name of the structural input file
    """
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
        mesh.append(
            f"tube_{t}_mesh",
            type="FileMeshGenerator",
            file=f"panel_{panel}_tube_{t}_in.e",
        )
        mesh.append(
            f"tube_{t}_renum",
            type="RenameBoundaryGenerator",
            input=f"tube_{t}_mesh",
            old_boundary=make_moose_hit_vector(["rmin"]),
            new_boundary=make_moose_hit_vector([rmin_id]),
        )
        mesh.append(
            f"tube_{t}_rename",
            type="RenameBoundaryGenerator",
            input=f"tube_{t}_renum",
            old_boundary=make_moose_hit_vector([rmin_id]),
            new_boundary=make_moose_hit_vector([f"rmin_tube_{t}"]),
        )
        mesh.append(
            f"tube_{t}_blk_renum",
            type="RenameBlockGenerator",
            input=f"tube_{t}_rename",
            old_block=make_moose_hit_vector(["0"]),
            new_block=make_moose_hit_vector([blk_id]),
        )
        mesh.append(
            f"tube_{t}_blk_rename",
            type="RenameBlockGenerator",
            input=f"tube_{t}_blk_renum",
            old_block=make_moose_hit_vector([blk_id]),
            new_block=make_moose_hit_vector([f"tube_{t}"]),
        )
    mesh.append(
        "combined",
        type="CombinerGenerator",
        inputs=make_moose_hit_vector([f"tube_{t}_blk_rename" for t in tubes]),
    )

    prev_input = "combined"
    for t in tubes:
        coord_a, coord_b = find_pin_coords(out_dir / f"panel_{panel}_tube_{t}_in.e")
        mesh.append(
            f"pin_xy_tube_{t}",
            type="ExtraNodesetGenerator",
            input=prev_input,
            new_boundary=make_moose_hit_vector([f"pin_xy_tube_{t}"]),
            coord=make_moose_hit_vector(
                [f"{coord_a[0]:.16g}", f"{coord_a[1]:.16g}", f"{coord_a[2]:.16g}"]
            ),
        )
        prev_input = f"pin_xy_tube_{t}"
        mesh.append(
            f"pin_y_tube_{t}",
            type="ExtraNodesetGenerator",
            input=prev_input,
            new_boundary=make_moose_hit_vector([f"pin_y_tube_{t}"]),
            coord=make_moose_hit_vector(
                [f"{coord_b[0]:.16g}", f"{coord_b[1]:.16g}", f"{coord_b[2]:.16g}"]
            ),
        )
        prev_input = f"pin_y_tube_{t}"

    root.append(
        "GlobalParams",
        displacements=make_moose_hit_vector(["disp_x", "disp_y", "disp_z"]),
    )

    aux_vars = root.append("AuxVariables")
    aux_vars.append("temp", order="FIRST", family="LAGRANGE")

    user_objs = root.append("UserObjects")
    user_objs.append(
        "tube_temp_soln",
        type="SolutionUserObject",
        mesh=thm_file,
        system_variables="T_solid",
        execute_on=make_moose_hit_vector(["initial", "timestep_begin"]),
    )

    functions = root.append("Functions")
    for t in tubes:
        functions.append(
            f"p_from_thm_tube_{t}",
            type="PiecewiseMultilinear",
            data_file=f"panel_{panel}_fch_tube_{t}_p.dat",
        )

    physics = root.append("Physics")
    sm = physics.append("SolidMechanics")
    qs = sm.append("QuasiStatic")
    qs.append(
        "all",
        strain="SMALL",
        new_system="true",
        eigenstrain_names="eigenstrain",
        add_variables="true",
        generate_output=make_moose_hit_vector(
            [
                "cauchy_stress_xx",
                "cauchy_stress_yy",
                "cauchy_stress_zz",
                "cauchy_stress_yz",
                "cauchy_stress_xz",
                "cauchy_stress_xy",
                "mechanical_strain_xx",
                "mechanical_strain_yy",
                "mechanical_strain_zz",
                "mechanical_strain_yz",
                "mechanical_strain_xz",
                "mechanical_strain_xy",
            ]
        ),
    )

    aux_kernels = root.append("AuxKernels")
    aux_kernels.append(
        "temp_from_thm",
        type="SolutionAux",
        variable="temp",
        solution="tube_temp_soln",
        from_variable="T_solid",
        execute_on=make_moose_hit_vector(["initial", "timestep_begin"]),
    )

    bcs = root.append("BCs")
    bcs.append(
        "z_disp",
        type="DirichletBC",
        variable="disp_z",
        boundary=make_moose_hit_vector(["bot"]),
        value=0.0,
    )
    for t in tubes:
        bcs.append(
            f"pin_x_tube_{t}",
            type="DirichletBC",
            variable="disp_x",
            boundary=make_moose_hit_vector([f"pin_xy_tube_{t}"]),
            value=0.0,
        )
        bcs.append(
            f"pin_y_tube_{t}",
            type="DirichletBC",
            variable="disp_y",
            boundary=make_moose_hit_vector([f"pin_xy_tube_{t}"]),
            value=0.0,
        )
        bcs.append(
            f"pin_y2_tube_{t}",
            type="DirichletBC",
            variable="disp_y",
            boundary=make_moose_hit_vector([f"pin_y_tube_{t}"]),
            value=0.0,
        )
    for t in tubes:
        bcs.append(
            f"inner_pressure_x_tube_{t}",
            type="Pressure",
            variable="disp_x",
            function=f"p_from_thm_tube_{t}",
            boundary=f"rmin_tube_{t}",
        )
        bcs.append(
            f"inner_pressure_y_tube_{t}",
            type="Pressure",
            variable="disp_y",
            function=f"p_from_thm_tube_{t}",
            boundary=f"rmin_tube_{t}",
        )

    constraints = root.append("Constraints")
    constraints.append(
        "ev_z",
        type="EqualValueBoundaryConstraint",
        variable="disp_z",
        secondary=make_moose_hit_vector(["top"]),
        penalty=1e7,
    )

    material_db = str(MATERIAL_DIR / f"{material}.xml")
    materials = root.append("Materials")
    materials.append(
        "stress",
        type="CauchyStressFromNEML",
        database=material_db,
        model=material_model,
        temperature="temp",
    )
    materials.append(
        "thermal_strain",
        type="ComputeThermalExpansionEigenstrainNEML",
        database=material_db,
        model=material_model,
        temperature="temp",
        stress_free_temperature=300.0,
        eigenstrain_name="eigenstrain",
    )

    precond = root.append("Preconditioning")
    precond.append("pc", type="SMP", full="true")

    root.append(
        "Executioner",
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
        petsc_options=make_moose_hit_vector(
            [
                "-snes_converged_reason",
                "-ksp_converged_reason",
                "-snes_linesearch_monitor",
            ]
        ),
        petsc_options_iname=make_moose_hit_vector(
            ["-pc_type", "-pc_factor_mat_solver_package"]
        ),
        petsc_options_value=make_moose_hit_vector(["lu", "superlu_dist"]),
        line_search="none",
        dt=dt,
        start_time=0.0,
        end_time=end_time,
    )

    outputs = root.append("Outputs", print_linear_residuals="false")
    outputs.append(
        "exodus_out",
        type="Exodus",
        file_base=output_base,
        sync_times=make_moose_hit_vector([f"{float(t):.16g}" for t in times]),
        sync_only="true",
    )
    outputs.append("csv_out", type="CSV", file_base=output_base)

    return root, i_name


def get_element_temperatures(model, conn, i_step):
    """
    Element-averaged temperature (K) at exodus 1-based step i_step

    Args:
        model: exodus model object
        conn: element connectivity matrix for the block (nelem, 8)
        i_step: 1-based step index in the exodus file
    Returns:
        temperatures: element-averaged temperatures (nelem,)
    """

    temp_all = model.get_variable_values("EX_NODAL", 0, "temp", i_step)
    return np.mean(temp_all[conn - 1], axis=1)


def read_tube_stress_and_temp(model, blk_id, conn, times):
    """Mandel-form Cauchy stress (ntime, nelem, 6) and element-averaged
    temperatures (ntime, nelem) for one HEX8 tube block in a MOOSE
    structural exodus

       Args:
           model: exodus model object
           blk_id: block ID for the tube in the exodus file
           conn: element connectivity matrix for the block (nelem, 8)
           times: list of time values in the exodus file
       Returns:
           mandel_stress: Cauchy stress in Mandel-form
           temperatures: element-averaged temperatures
    """
    n_times = len(times)
    n_elem = conn.shape[0]
    mandel_stress = np.zeros((n_times, n_elem, 6))
    temperatures = np.zeros((n_times, n_elem))
    stress_vars = [
        "cauchy_stress_xx",
        "cauchy_stress_yy",
        "cauchy_stress_zz",
        "cauchy_stress_yz",
        "cauchy_stress_xz",
        "cauchy_stress_xy",
    ]
    mandel_mult = np.array([1.0, 1.0, 1.0, SQRT2, SQRT2, SQRT2])
    for t_idx in range(n_times):
        i_step = t_idx + 1
        for s_idx, var_name in enumerate(stress_vars):
            s_vals = model.get_variable_values(
                "EX_ELEM_BLOCK", blk_id, var_name, i_step
            )
            mandel_stress[t_idx, :, s_idx] = s_vals * mandel_mult[s_idx]
        temperatures[t_idx] = get_element_temperatures(model, conn, i_step)
    return mandel_stress, temperatures


def compute_moose_reliability(
    rec, mat_damage, damage_model, lifetime, moose_sm_output_files, tube_multiplier
):
    """Read MOOSE structural exodus output and returns reliability using srlife's models.

    Args:
        rec: srlife Receiver.
        mat_damage: material damage model variant.
        damage_model: srlife damage model (e.g. PIAModel).
        lifetime: float, hours.
        moose_sm_output_files: list of paths to MOOSE SM exodus files
        tube_multiplier: float, scaling from analysis tubes to actual
            tubes for per-panel

    Returns:
    a dict that the driver script can use:
    - "lifetime": input lifetime in hours
    - "tube_volume": list of per-tube volume flaw reliabilities(VFR)
    - "tube_surface": list of per-tube surface flaw reliabilities (SFR)
    - "tube_combined": list of per-tube combined reliabilities (CR)
    - "panel_volume": list of per-panel VFR
    - "panel_surface": list of per-panel SFR
    - "panel_combined": list of per-panel CR
    - "overall_volume": overall VFR
    - "overall_surface": overall SFR
    - "overall_combined": overall CR

    """
    m3_to_mm3 = convert_m_to_mm(1.0) ** 3
    m2_to_mm2 = convert_m_to_mm(1.0) ** 2

    # One representative tube
    sample_tube = next(iter(next(iter(rec.panels.values())).tubes.values()))
    nt, nz = sample_tube.nt, sample_tube.nz

    volumes = sample_tube.element_volumes() * m3_to_mm3
    surface, normals = sample_tube.surface_elements()

    # element_surface_areas() ships (z, t) per side; surface mask is (r, t, z)
    # with z fastest -- transpose each side so areas align with the mask.
    sa_raw = sample_tube.element_surface_areas()
    half = (nz - 1) * nt
    inner_sa = sa_raw[:half].reshape(nz - 1, nt).T.flatten()
    outer_sa = sa_raw[half:].reshape(nz - 1, nt).T.flatten()
    surface_areas = np.concatenate([inner_sa, outer_sa]) * m2_to_mm2

    # Reorder so surface elements come first -- workaround for srlife's
    # damage.py [:count_surface_elements] slicing in the surface-flaw call.
    sort_order = np.argsort(~surface)
    volumes_r = volumes[sort_order]
    surface_r = surface[sort_order]
    normals_r = normals[sort_order]

    per_panel_tube_results = []
    for sm_exo_path in moose_sm_output_files:
        model = exo.exodus(str(sm_exo_path), array_type="numpy")
        times = np.asarray(model.get_times())
        times_hr = (
            times / 3600.0
        )  # in moose solution the time is in seconds, here we need hours.
        tube_results = []
        for blk_id in model.get_elem_blk_ids():
            conn_flat, num_elem, npe = model.get_elem_connectivity(blk_id)
            if npe != 8:
                continue
            conn = np.array(conn_flat, dtype=int).reshape(num_elem, npe)
            mandel_stress, temperatures = read_tube_stress_and_temp(
                model, blk_id, conn, times
            )
            mandel_stress = mandel_stress[:, sort_order]
            temperatures = temperatures[:, sort_order]
            vol_log_rel = damage_model.calculate_volume_flaw_element_log_reliability(
                times_hr,
                mandel_stress,
                temperatures,
                volumes_r,
                mat_damage,
                lifetime,
            )
            surf_log_rel = damage_model.calculate_surface_flaw_element_log_reliability(
                times_hr,
                mandel_stress,
                surface_r,
                normals_r,
                temperatures,
                surface_areas,
                mat_damage,
                lifetime,
            )
            combined_log_rel = vol_log_rel.copy()
            combined_log_rel[np.where(surface_r)[0]] += surf_log_rel
            tube_results.append(
                {
                    "volume": float(np.sum(vol_log_rel)),
                    "surface": float(np.sum(surf_log_rel)),
                    "combined": float(np.sum(combined_log_rel)),
                }
            )
        model.close()
        per_panel_tube_results.append(tube_results)

    all_vol = np.array([r["volume"] for tubes in per_panel_tube_results for r in tubes])
    all_surf = np.array(
        [r["surface"] for tubes in per_panel_tube_results for r in tubes]
    )
    all_comb = np.array(
        [r["combined"] for tubes in per_panel_tube_results for r in tubes]
    )

    panel_volume, panel_surface, panel_combined = [], [], []
    idx = 0
    for tubes in per_panel_tube_results:
        n = len(tubes)
        panel_volume.append(
            float(np.exp(np.sum(all_vol[idx : idx + n] * tube_multiplier)))
        )
        panel_surface.append(
            float(np.exp(np.sum(all_surf[idx : idx + n] * tube_multiplier)))
        )
        panel_combined.append(
            float(np.exp(np.sum(all_comb[idx : idx + n] * tube_multiplier)))
        )
        idx += n

    return {
        "lifetime": lifetime,
        "tube_volume": [float(v) for v in np.exp(all_vol)],
        "tube_surface": [float(v) for v in np.exp(all_surf)],
        "tube_combined": [float(v) for v in np.exp(all_comb)],
        "panel_volume": panel_volume,
        "panel_surface": panel_surface,
        "panel_combined": panel_combined,
        "overall_volume": float(np.exp(np.sum(all_vol * tube_multiplier))),
        "overall_surface": float(np.exp(np.sum(all_surf * tube_multiplier))),
        "overall_combined": float(np.exp(np.sum(all_comb * tube_multiplier))),
    }
