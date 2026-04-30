# pylint: disable=too-many-lines,too-many-branches
"""
  This module define the data structures used as input and output to the analysis module.

  The require input data can be provided by constructing a Receiver object in python or by 
  loading an HDF5 datafile, which will populate the python class hierarchy.
"""

import itertools
from collections import OrderedDict

import numpy as np
import scipy.interpolate as inter
import h5py

import subprocess
import os
import sys 

# Get absolute paths to moose python modules
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)  # go up one level
moose_root = os.path.join(repo_root, 'moose') # go to moose directory
# Add moose python paths for pyhit
moose_python_paths = [
    os.path.join(moose_root, 'python'),
    os.path.join(moose_root, 'python', 'pyhit'),
    os.path.join(moose_root, 'framework', 'contrib', 'hit'),
]

for path in moose_python_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

import pyhit
from pyhit import moosetree
conda_env_dir = os.environ.get("CONDA_PREFIX")
# Needed to use exodus.py
ACCESS = os.getenv("ACCESS", f"{conda_env_dir}/seacas")
sys.path.append(os.path.join(ACCESS, "lib"))
sys.path.append(os.path.join(ACCESS, "lib64"))
import exodus as exo


from srlife import writers


class Receiver:
    """Basic definition of the tubular receiver geometry.

    A receiver is a collection of panels linked together by
    an elastic spring stiffness.  This stiffness can be a real number,
    "rigid" or "disconnect"

    Panels can be labeled by strings.  By default the names
    are sequential numbers.

    In addition this object stores some required metadata:
      1) The daily cycle period (which can be less than 24 hours
         if the analysis neglects some of the night period)
      2) The number of days (see #1) explicitly represented in the
         analysis results.

    Args:
      period (float): single daily cycle period
      days (int): number of daily cycles explicitly represented
      panel_stiffness (float or string): panel stiffness (float) or "rigid" or "disconnect"
    """

    def __init__(self, period, days, panel_stiffness):
        """Initialize a Receiver object"""
        self.period = period
        self.days = days
        self.panels = OrderedDict()
        self.stiffness = panel_stiffness

        self.flowpaths = OrderedDict()

    def add_flowpath(self, panels_in_path, times, mass_rate, inlet_temp, name=None):
        """Add a flow path to the receiver model

        Args:
            panels_in_path (list):  list of panel names in the path
            times (np.array):       time data
            mass_rate (np.array):   mass flow rate data
            inlet_temp (np.array):  inlet temperature data
            name (Optional[str]):   optional flowpath name
        """
        if not name:
            name = next_name(self.flowpaths.keys())

        for n in panels_in_path:
            if n not in self.panels.keys():
                raise ValueError("Panel %s does not exist in the receiver!" % n)

        self.flowpaths[name] = {
            "panels": panels_in_path,
            "times": times,
            "mass_flow": mass_rate,
            "inlet_temp": inlet_temp,
        }

    def write_vtk(self, basename):
        """Write out the receiver as individual panels with names basename_panelname

        The VTK format is mostly used for additional postprocessing.  The VTK
        format cannot be used for input.

        Args:
          basename (str):  base file name
        """
        for n, panel in self.panels.items():
            panel.write_vtk(basename + "_" + n)

    def close(self, other):
        """Check to see if two objects are nearly equal.

        Primarily used for testing

        Args:
          other (Receiver):      the object to compare against

        Returns:
          bool:   True if the receivers are similar.
        """
        base = (
            np.isclose(self.period, other.period)
            and np.isclose(self.days, other.days)
            and np.isclose(self.stiffness, other.stiffness)
        )
        for name, panel in self.panels.items():
            if name not in other.panels:
                return False
            base = base and panel.close(other.panels[name])

        return base

    @property
    def tubes(self):
        """Shortcut iterator over all tubes

        Returns:
          iterator over panels
        """
        return itertools.chain(
            *(panel.tubes.values() for panel in self.panels.values())
        )

    def set_paging(self, page):
        """Tell tubes to store results on or off disk

        Args:
          page (bool):    if true, page results to disk
        """
        for i, tube in enumerate(self.tubes):
            tube.set_paging(page, i)

    @property
    def ntubes(self):
        """Shortcut for total number of tubes

        Returns:
          int: Number of tubes in all panels
        """
        return len(list(self.tubes))

    @property
    def npanels(self):
        """Number of panels in the receiver

        Returns:
          int:  Number of panels
        """
        return len(self.panels)

    def add_panel(self, panel, name=None):
        """Add a panel object to the receiver

        Args:
          panel (Panel):          panel object
          name (Optional[str]):   panel name, by default follows fixed scheme
        """
        if not name:
            name = next_name(self.panels.keys())

        self.panels[name] = panel

    def save(self, fobj):
        """Save to an HDF5 file

        This saves a Receiver object to the HDF5 format.

        Args:
          fobj (str):  either a h5py file object or a filename
        """
        if isinstance(fobj, str):
            fobj = h5py.File(fobj, "w")

        fobj.attrs["period"] = self.period
        fobj.attrs["days"] = self.days
        fobj.attrs["stiffness"] = self.stiffness

        grp = fobj.create_group("panels")

        for name, panel in self.panels.items():
            sgrp = grp.create_group(name)
            panel.save(sgrp)

        grp = fobj.create_group("flowpaths")
        for name, path in self.flowpaths.items():
            sgrp = grp.create_group(name)
            sgrp.create_dataset("panels", data=path["panels"])
            sgrp.create_dataset("times", data=path["times"])
            sgrp.create_dataset("mass_flow", data=path["mass_flow"])
            sgrp.create_dataset("inlet_temp", data=path["inlet_temp"])

    @classmethod
    def load(cls, fobj):
        """Load a Receiver from an HDF5 file

        A full description of the HDF format is included in the module documentation

        Args:
          fobj (string):  either a h5py file object or a filename

        Returns:
          Receiver: The constructed receiver object.
        """
        if isinstance(fobj, str):
            fobj = h5py.File(fobj, "r")

        res = cls(fobj.attrs["period"], fobj.attrs["days"], fobj.attrs["stiffness"])

        grp = fobj["panels"]

        for name in grp:
            res.add_panel(Panel.load(grp[name]), name)

        if "flowpaths" in fobj:
            grp = fobj["flowpaths"]

            for name in grp:
                res.add_flowpath(
                    list(
                        map(
                            lambda x: x.decode(encoding="UTF-8"),
                            np.copy(grp[name]["panels"]),
                        )
                    ),
                    np.copy(grp[name]["times"]),
                    np.copy(grp[name]["mass_flow"]),
                    np.copy(grp[name]["inlet_temp"]),
                    name=name,
                )

        if "flowpaths" in fobj:
            grp = fobj["flowpaths"]

            for name in grp:
                res.add_flowpath(
                    list(
                        map(
                            lambda x: x.decode(encoding="UTF-8"),
                            np.copy(grp[name]["panels"]),
                        )
                    ),
                    np.copy(grp[name]["times"]),
                    np.copy(grp[name]["mass_flow"]),
                    np.copy(grp[name]["inlet_temp"]),
                    name=name,
                )

        return res

    def create_moose_thm_inlet(
        self, moose_node, name, connectivity, m_dot_in, fluid_inlet_T
    ):
        """
        Create an inlet object for a moose THM analysis.

        Args:
          node (pyhit.Node): the node object to append the inlet to
            typically this would be a node for a flowpath in the Components
            section of MOOSE input file
          name (string): name for inlet, want to include panel info to make it searchable
          connectivity (string): The moose path to the node the inlet connects to
          m_dot_in (double): input mass flow rate to inlet
          fluid_inlet_T (double): input fluid temp for inlet

        Return: None
        """
        print("Creating Moose Inlet!!!")
        moose_node.append(
            name,
            type="InletMassFlowRateTemperature1Phase",
            input=connectivity,
            m_dot=convert_kghr_to_kgs(m_dot_in),
            T=fluid_inlet_T,
        )

    def create_moose_thm_outlet(self, moose_node, name, connectivity, outlet_p):
        """
        Create an outlet object for a moose THM analysis.

        Args:
          node (pyhit.Node): the node object to append the inlet to
            typically this would be a node for a flowpath in the Components
            section of MOOSE input file
          name (string): name for outlet, want to include panel info to make it searchable
          connectivity (string): The moose path to the node the outlet connects to
          outlet_p (double): outlet pressure

        Return: None
        """
        print("Creating Moose outlet!!!")
        moose_node.append(name, type="Outlet1Phase", input=connectivity, p=outlet_p)

    def create_moose_thm_panel_to_panel_connection(
        self, comp_node, panel_node, prev_panel_node, manifold_tube, tube_roughness, pos
    ):
        """
        Create a tube and connection between panels. This connects the
        current panel being built to the previously built panel.

        Args:
          comp_node (pyhit.Node): pyhit node for components section
          panel_node (pyhit.Node): pyhit node for current panel
          prev_panel_node (pyhit.Node): pyhit node for prev panel
          manifold_tube (Tube): helper object for manifold tubes
          tube_roughness (double): tube roughness used in flow simulation
          pos (string): sets top or bottom connection
        """
        # find in/out tubes from panels
        prev_panel_tube_name = f"fch_{prev_panel_node.name}_"
        panel_tube_name = f"fch_{panel_node.name}_"
        if pos == "bot":
            in_out_string = "in"
        else:
            in_out_string = "out"
        prev_panel_tube_name += in_out_string
        panel_tube_name += in_out_string
        # Get in/out from prev panel
        prev_panel_tube = moosetree.find(
            prev_panel_node, func=lambda n: n.name == prev_panel_tube_name
        )
        # Get in/out from this panel
        panel_tube = moosetree.find(
            panel_node, func=lambda n: n.name == panel_tube_name
        )
        if prev_panel_tube is None or panel_tube is None:
            print("CANNOT FIND PANEL in/out tube!!!")
            print(prev_panel_tube_name)
            print(panel_tube_name)
            sys.exit()
        # get height of this panels tube
        in_out_tube_height = float(panel_tube["length"])
        # create tube between panels
        # take hit vector to np.array of floats
        tube_pos = str(prev_panel_tube["position"])
        tube_start = np.fromstring(tube_pos, dtype=float, sep=" ")
        tube_pos = str(panel_tube["position"])
        tube_end = np.fromstring(tube_pos, dtype=float, sep=" ")
        # adjust connector tubes to match correct height for top tubes
        if pos == "top":
            tube_start[2] += in_out_tube_height
            tube_end[2] += in_out_tube_height
        orientation = tube_end - tube_start
        dist = np.sqrt(np.dot(orientation, orientation))
        tube_name = f"fch_{prev_panel_node.name}_to_{panel_node.name}"
        manifold_tube_ir = convert_mm_to_m(manifold_tube.r - manifold_tube.t)
        A_pipe = np.pi * manifold_tube_ir**2
        comp_node.append(
            tube_name,
            type="FlowChannel1Phase",
            position=make_moose_hit_vector(tube_start),
            orientation=make_moose_hit_vector(orientation),
            n_elems=manifold_tube.nz,
            length=dist,
            A=A_pipe,
            D_h=2.0 * manifold_tube_ir,
            roughness=tube_roughness,
            fp="fp",
        )
        # create jcts between tubes
        connectivity = [
            f"{prev_panel_node.name}/{prev_panel_tube_name}:{in_out_string}",
            f"{tube_name}:in",
        ]
        comp_node.append(
            f"jct_{prev_panel_node.name}_to_connector",
            type="VolumeJunction1Phase",
            position=make_moose_hit_vector(tube_start),
            volume=A_pipe * 2.0,
            connections=make_moose_hit_vector(connectivity),
        )
        connectivity = [
            f"{tube_name}:out",
            f"{panel_node.name}/{panel_tube_name}:{in_out_string}",
        ]
        comp_node.append(
            f"jct_connector_to_{panel_node.name}",
            type="VolumeJunction1Phase",
            position=make_moose_hit_vector(tube_end),
            volume=A_pipe * 2.0,
            connections=make_moose_hit_vector(connectivity),
        )

    def create_moose_thm_front_matter(self, moose_root, press, fluid_inlet_T):
        """
        Create front matter before components in MOOSE input file
        This includes GlobalParams, materials, functions, closures, etc

        Args:
          moose_root (pyhit.Node): root node for pyhit moose interface
          press (double): outlet pressure for flowpath
          fluid_inlet_T (double): inlet fluid temp, used to set initial_T
            of all components and fluid

        ToDo: Use solver, fluid, material, etc objects
        """
        # global params
        moose_root.append(
            "GlobalParams",
            initial_p=press,
            initial_vel=0.0,
            initial_T=fluid_inlet_T,
            initial_vel_x=0.0,
            initial_vel_y=0.0,
            initial_vel_z=0.0,
            gravity_vector=make_moose_hit_vector([0, 0, -9.81]),
            rdg_slope_reconstruction="full",
            scaling_factor_1phase=make_moose_hit_vector([1, 1e-2, 1e-4]),
            closures="thm_closure",
        )
        # fp = "fp")
        # material properties
        mat_node = moose_root.append("Materials")

        mat_node.append(
            "tube_mat",
            type="ADGenericConstantMaterial",
            prop_names=make_moose_hit_vector(
                ["density", "specific_heat", "thermal_conductivity"]
            ),
            prop_values=make_moose_hit_vector([3200, 947, 60]),
        )
        # fluid properties
        fluid_node = moose_root.append("FluidProperties")
        fluid_node.append(
            "sco2", type="CO2FluidProperties", allow_imperfect_jacobians="true"
        )
        fluid_node.append(
            "fp",
            type="TabulatedBicubicFluidProperties",
            fp="sco2",
            interpolated_properties=make_moose_hit_vector(
                [
                    "density",
                    "enthalpy",
                    "viscosity",
                    "internal_energy",
                    "k",
                    "cv",
                    "cp",
                    "c",
                    "entropy",
                ]
            ),
            # out_of_bounds_behavior="declare_invalid",
            temperature_min=240,
            temperature_max=999,
            pressure_min=1e7,
            pressure_max=8e7,
            num_T=100,
            num_p=100,
            tolerance=1e-5,
            T_initial_guess=773,
            p_initial_guess=2e7,
            construct_pT_from_ve="true",
            construct_pT_from_vh="true",
            allow_imperfect_jacobians="false",
        )
        # fluid_node.append("fp",
        #                   type="SimpleFluidProperties",
        #                   density0=1500,
        #                   bulk_modulus=2.0e9,
        #                   porepressure_coefficient=0,
        #                   molar_mass=0.0812,
        #                   specific_entropy=0.0,
        #                   thermal_expansion=1.0e-12,
        #                   cv=1008,
        #                   cp=1010,
        #                   viscosity=4.0e-3,
        #                   thermal_conductivity=0.44)

        # Closures
        closure_node = moose_root.append("Closures")
        closure_node.append("thm_closure", type="Closures1PhaseTHM")

        # Functions
        moose_root.append("Functions")
        # user objects needed for flux SolutionFunctions
        moose_root.append("UserObjects")

    def create_moose_thm_back_matter(
        self,
        moose_root,
        flow_path_name,
        target_outlet_T,
        mass_flow_with_t,
        inlet_name,
        inlet_pipe_name,
        outlet_pipe_name,
        start_time,
        end_time,
        dt,
        dtmin,
        dtmax,
        nl_rel_tol,
        nl_abs_tol,
        nl_max_its,
    ):
        """
        Add back matter to moose_root node.
        BCs, controls, postProc, precon, execs, outputs

        Args:
          moose_root (pyhit.Node): pyhit node for the root of moose sim
          flow_path_name (string): flowpath name used to set filename for csv outs
          target_outlet_T (double): target temperature for outlet, used in controls
          mass_flow_with_t (np.array): mass flow rate with time
          inlet_name (string): name of pyhit node for inlet to flowpath
          inlet_pipe_name (string): name of pipe after inlet
          outlet_pipe_name (string): name of pipe before outlet
          start_time (double): sim start time (sec)
          end_time (double): sim end time (sec)
          dt (double): init time step (sec)
          dtmin (double): min time step (sec)
          dtmax (double): max time step (sec)
          nl_rel_tol (double): relative tolerance for newton iter
          nl_abs_tol (double): absolute tolerance for newton iter
          nl_max_its (int): maximum number of iterations for newton solver

        """
        # Controls
        useControls = True
        if useControls:
            m_dot_fun = "m_dot_time_fun"
            func_node = moosetree.find(moose_root, func=lambda n: n.name == "Functions")
            if func_node is None:
                func_node = moose_root.append("Functions")
            times = np.arange(start_time, end_time, dtmax)
            times = np.append(times, end_time)
            func_node.append(
                m_dot_fun,
                type="PiecewiseLinear",
                x=make_moose_hit_vector(times),
                y=make_moose_hit_vector(mass_flow_with_t),
            )
            control_node = moose_root.append("ControlLogic")
            control_node.append(
                "set_inlet_mass_flow",
                type="TimeFunctionComponentControl",
                component=inlet_name,
                parameter="m_dot",
                function=m_dot_fun,
            )

        # post processors
        post_proc_node = moose_root.append("Postprocessors")
        post_proc_node.append(
            "m_dot_inlet",
            type="RealComponentParameterValuePostprocessor",
            component=inlet_name,
            parameter="m_dot",
        )
        post_proc_node.append(
            "path_T_in", type="SideAverageValue", boundary=inlet_pipe_name, variable="T"
        )
        post_proc_node.append(
            "path_T_out",
            type="SideAverageValue",
            boundary=outlet_pipe_name,
            variable="T",
        )
        post_proc_node.append(
            "core_p_in", type="SideAverageValue", boundary=inlet_pipe_name, variable="p"
        )
        post_proc_node.append(
            "core_p_out",
            type="SideAverageValue",
            boundary=outlet_pipe_name,
            variable="p",
        )
        post_proc_node.append(
            "core_delta_p",
            type="ParsedPostprocessor",
            pp_names=make_moose_hit_vector(["core_p_in", "core_p_out"]),
            function=make_moose_hit_vector(["core_p_in - core_p_out"]),
        )

        # PRECONDITIONER
        precon_node = moose_root.append("Preconditioning")
        precon_node.append(
            "pc",
            type="SMP",
            full="true",
            petsc_options_iname=make_moose_hit_vector(["-snes_test_err"]),
            petsc_options_value=make_moose_hit_vector([1.0e-9]),
        )

        # EXECUTIONER
        exec_node = moose_root.append(
            "Executioner",
            type="Transient",
            start_time=start_time,
            dtmin=dtmin,
            dtmax=dtmax,
            end_time=end_time,
            line_search="basic",
            solve_type="NEWTON",
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
            l_max_its=200,
            l_tol=1.0e-10,
            nl_rel_tol=nl_rel_tol,
            nl_abs_tol=nl_abs_tol,
            nl_max_its=nl_max_its,
        )
        exec_node.append(
            "TimeStepper", type="IterationAdaptiveDT", dt=dt, growth_factor=4
        )
        exec_node.append("TimeIntegrator", type="BDF2")
        # OUTPUTS
        sync_times = np.arange(start_time, end_time + dtmax, dtmax)
        output_node = moose_root.append("Outputs", print_linear_residuals="false")
        output_node.append(
            "exo",
            type="Exodus",
            sync_times=make_moose_hit_vector(sync_times),
            sync_only="true",
        )
        output_node.append(
            "console", type="Console", max_rows=1, outlier_variable_norms="false"
        )
        output_node.append(
            "csv",
            type="CSV",
            file_base=f"{flow_path_name}_outs",
            delimiter=",",
            show=make_moose_hit_vector(["core_delta_p", "m_dot_inlet", "path_T_out"]),
        )

    def create_moose_thm_model(
        self,
        moose_filename,
        rec_diam,
        tube_od,
        tube_spacing,
        tube_roughness,
        manifold_tube,
        outlet_p,
        target_outlet_T,
        start_time,
        end_time,
        dt,
        nl_rel_tol,
        nl_abs_tol,
    ):
        """
        Create a moose THM input file for the receiver. This file
        can then be used with MOOSE to run the Thermal Hydraulics analysis.

        This assumes the receiver is fully defined and ready for analysis

        NOTE: To work here we must assume panel ordering. This allows
        us to put the panels/tubes in correct positions

        Args:
            moose_filename (string): filename used for the moose input
            rec_diam (double): diameter of receiver (THIS SHOULD BE MEMBER OF REC)
            tube_od (double): diameter of receiver tubes
            tube_spacing (double): space between tubes, for edge spacing to tubes
            tube_roughness (double): tube roughness used in flow simulation
            manifold_tube (Tube): srlife tube objects for manifold pipes
            outlet_p (double): outlet pressure for flow
            target_outlet_T (double): target fluid temp at outlet
            start_time (double): sim start time
            end_time (double): sim end time
            dt (double): init time step
            nl_rel_tol (double): relative tolerance for newton iter
            nl_abs_tol (double): absolute tolerance for newton iter

        Returns: None
        """
        print("Creating Moose Model!!!")
        # simulation specifics
        dtmax = 3600
        dtmin = 1
        nl_max_its = 40
        # length of in/out tubes to panels
        panel_in_out_length = 0.1

        # This loop will create a moose input file for each flowpath in receiver
        panel_delta_theta = 2.0 * np.pi / self.npanels
        rec_radius = 0.5 * convert_mm_to_m(rec_diam)
        edge_spacing_angle = convert_mm_to_m(tube_spacing + 0.5 * tube_od) / (
            rec_radius
        )
        filenames = []
        for path_key, flowpath in self.flowpaths.items():
            # for each flowpath in receiver
            # make pyhit root
            # loop over panels and call their functions
            moose_root = pyhit.Node(parent=None, hitnode=None, offset=None)
            # add moose front matter: Global params, fluid props, closures, functions, etc
            self.create_moose_thm_front_matter(
                moose_root, outlet_p, flowpath["inlet_temp"][0]
            )
            # Component loop: This will create all components needed in MOOSE
            comp_node = moose_root.append("Components")
            # NOTE: MOOSE THM cannot do groups of groups
            # so I cannot create a flowpath group
            print(f"path: {path_key}")
            connectivity = (
                f"panel_{flowpath['panels'][0]}/fch_panel_{flowpath['panels'][0]}_in:in"
            )
            inlet_name = f"inlet_panel_{flowpath['panels'][0]}"
            self.create_moose_thm_inlet(
                comp_node,
                inlet_name,
                connectivity,
                flowpath["mass_flow"][0],
                flowpath["inlet_temp"][0],
            )
            for iPanel, panel_name in enumerate(flowpath["panels"]):
                print(f"panel: {panel_name}")
                # for each panel in flowpath
                # get theta start and delta theta
                panel_node = comp_node.append(f"panel_{panel_name}")
                panel = self.panels[panel_name]
                panel_theta_start = (
                    int(panel_name) * panel_delta_theta + edge_spacing_angle
                )
                panel_theta_end = (
                    int(panel_name) + 1
                ) * panel_delta_theta - edge_spacing_angle
                panel.create_panel_components(
                    panel_node,
                    panel_theta_start,
                    panel_theta_end,
                    rec_radius,
                    tube_roughness,
                    manifold_tube,
                    panel_in_out_length,
                )
                if iPanel != 0:
                    # if we are not in first panel, connect panels
                    prev_panel_name = f"panel_{flowpath['panels'][iPanel-1]}"
                    prev_panel_node = moosetree.find(
                        comp_node, func=lambda n: n.name == prev_panel_name
                    )
                    if iPanel % 2 == 0:
                        # even panel, connect bot
                        pos = "bot"
                        self.create_moose_thm_panel_to_panel_connection(
                            comp_node,
                            panel_node,
                            prev_panel_node,
                            manifold_tube,
                            tube_roughness,
                            pos,
                        )
                    else:
                        # odd panel, connect top
                        pos = "top"
                        self.create_moose_thm_panel_to_panel_connection(
                            comp_node,
                            panel_node,
                            prev_panel_node,
                            manifold_tube,
                            tube_roughness,
                            pos,
                        )
            connectivity = (
                f"panel_{flowpath['panels'][-1]}/fch_panel_{flowpath['panels'][-1]}_"
            )
            if len(flowpath["panels"]) % 2:
                # if there are even number of panels in path
                outlet_pos = "out:out"
            else:
                # odd number of panels in path
                outlet_pos = "in:in"
            connectivity += outlet_pos
            outlet_name = f"outlet_panel_{flowpath['panels'][-1]}"
            self.create_moose_thm_outlet(comp_node, outlet_name, connectivity, outlet_p)
            # create moose back matter: BCs, controls, postProcessors,
            # precon, execs, outputs
            inlet_pipe_name = (
                f"panel_{flowpath['panels'][0]}/fch_panel_{flowpath['panels'][0]}_in:in"
            )
            outlet_pipe_name = connectivity
            flow_path_name = f"flowpath_{path_key}"
            self.create_moose_thm_back_matter(
                moose_root,
                flow_path_name,
                target_outlet_T,
                convert_kghr_to_kgs(flowpath["mass_flow"]),
                inlet_name,
                inlet_pipe_name,
                outlet_pipe_name,
                start_time,
                end_time,
                dt,
                dtmin,
                dtmax,
                nl_rel_tol,
                nl_abs_tol,
                nl_max_its,
            )
            moose_flow_path_filename = f"{moose_filename}_{flow_path_name}.i"
            pyhit.write(moose_flow_path_filename, moose_root)
            filenames.append(moose_flow_path_filename)
        return filenames

    def run_moose_thm_model(self, moose_input_filename):
        """
        Runs the MOOSE THM model using inputs

        Args:
          moose_exec (String): moose executable path and name
          moose_input_filename (String): filename to call moose with

        """
        try:
            print("Running MOOSE!")
            mpirun = os.environ.get("MOOSE_MPI")
            nprocs = os.environ.get("MOOSE_NPROCS")
            moose_thm = os.environ.get("MOOSE_THM")
            argv = [mpirun, "-n", nprocs, moose_thm, "-i", moose_input_filename]
            result = subprocess.run(argv,
                                    check=True, capture_output=False, text=True)
        except subprocess.CalledProcessError as e:
            print(f"MOOSE returned error {e.returncode}")
            print(f"stderr: {e.stderr}")

    def get_moose_thm_results(self, moose_input_filenames):
        """
        This will hop into MOOSE exodus file and get
        flow tube nodal values of pressure and
        heat tube nodal values of temp
        Then load them into srlife Tube objects

        Agrs:
          moose_input_filename (list(string)): name of moose input file used to get output
            There will be one filename per flowpath
        """
        # Steps:
        # 1) read MOOSE data and get dicts
        #    using exodus.py from SEACAS
        # 2) load MOOSE data into Tubes
        #    using add_results
        # probably better to do this through tube objects
        # loop over tubes, get results from name/dict
        # have tube add results
        # can loop over panels in rec (get panel_node)
        # then tube in panels, find heat_tube? or just use name from there?
        # name = {panel_node.name}/heat_tube_i for i in panel.tubes
        for iPath, flowpath in enumerate(self.flowpaths.values()):
            # for each flowpath in model
            input_filename = moose_input_filenames[iPath]
            output_filename = input_filename[0:-2] + "_exo.e"
            # read temp and pressure results from output exodus
            times, press_results, temp_results = read_moose_thm_exodus_file(
                output_filename
            )
            for panel_name in flowpath["panels"]:
                # for each panel in receiver
                panel_str = f"panel_{panel_name}/"
                panel = self.panels[panel_name]
                for iTube, tube in enumerate(panel.tubes.values()):
                    # for each tube in panel
                    flow_tube_name = f"{panel_str}fch_tube_{iTube}"
                    press_res_time_space = press_results[flow_tube_name]
                    heat_tube_name = f"{panel_str}heat_tube_{iTube}:0"
                    heat_res_time_space = temp_results[heat_tube_name]
                    # load MOOSE results into Tube object
                    tube.load_moose_thm_results(
                        times, press_res_time_space, heat_res_time_space
                    )


def read_moose_thm_exodus_file(moose_output_filename):
    """
    Read an exodus file and return results
    In particular we get dictionary of elem block name and results.
    Flow tube nodal pressure results and
    Heat tube nodal temperature results

    Args:
      moose_output_filename (string): name of moose exodus output file

    Returns:
      times (list): list of analysis times
      press_results (dict): {elem_block_name, nodal_pressures} pressure results
      temp_results (dict): {elem_block_name, nodal_temp} temp results
    """
    model = exo.exodus(moose_output_filename, array_type="numpy")
    times = model.get_times()
    pressure_dict = {}
    temp_dict = {}

    for iElemBlk in model.get_elem_blk_ids():
        name = model.get_elem_blk_name(iElemBlk)
        if name[0:5] == "panel":
            if "fch_tube" in name:
                pressure_results = []
                for iStep in range(len(times)):
                    # NOTE: exodus.py indexes times steps from 1
                    press_data = model.get_variable_values(
                        "EX_ELEM_BLOCK", iElemBlk, "p", iStep + 1
                    )
                    pressure_results.append(press_data)
                pressure_dict[name] = np.array(pressure_results)
            if "heat_tube" in name:
                connect, _, _ = model.get_elem_connectivity(iElemBlk)
                nodes = np.unique(connect)
                temp_results = []
                nodeCoord = []
                for node in nodes:
                    x, y, z = model.get_coord(node)
                    nodeCoord.append([x, y, z])

                for iStep in range(len(times)):
                    # NOTE: exodus.py indexes times steps from 1
                    temp_data = model.get_variable_values(
                        "EX_NODAL", iElemBlk, "T_solid", iStep + 1
                    )
                    temp_results.append(temp_data[nodes - 1])
                temp_dict[name] = [np.array(nodeCoord), np.array(temp_results)]
    return times, pressure_dict, temp_dict


class Panel:
    """Basic definition of a panel in a tubular receiver.

    A panel is a collection of Tube object linked together by
    an elastic spring stiffness.  This stiffness can be a real number,
    a string "disconnect" or a string "rigid"

    Tubes in the panel can be labeled by strings.  By default the
    names are sequential numbers.

    Args:
      stiffness:                        manifold spring stiffness
    """

    def __init__(self, stiffness, ntubes_actual=None):
        """Initialize the panel"""
        self.tubes = OrderedDict()
        self.stiffness = stiffness

    def write_vtk(self, basename):
        """Write out the panels as individual tubes with names basename_tubename

        Args:
          basename (string): base file name
        """
        for n, tube in self.tubes.items():
            tube.write_vtk(basename + "_" + n)

    def close(self, other):
        """Check to see if two objects are nearly equal.

        Primarily used for testing

        Args:
          other (Panel): the object to compare against

        Returns:
          bool: true if the panels are sufficiently similar
        """
        base = np.isclose(self.stiffness, other.stiffness)
        for name, tube in self.tubes.items():
            if name not in other.tubes:
                return False
            base = base and tube.close(other.tubes[name])

        return base

    @property
    def ntubes(self):
        """Number of tubes in the panel

        Returns:
          int:  number of tubes in the panel
        """
        return len(self.tubes)

    @property
    def ntubes_actual(self):
        """Number of actual tubes in the panel

        Returns:
            int:    number of tubes in the full panel
        """
        total = 0
        for t in self.tubes.values():
            total += t.multiplier

        return total

    def add_tube(self, tube, name=None):
        """Add a tube object to the panel

        Args:
          tube (Tube): tube object
          name (Optional[str]): Tube name, defaults to fixed scheme.
        """
        if not name:
            name = next_name(self.tubes.keys())

        self.tubes[name] = tube

    def save(self, fobj):
        """Save to an HDF5 file

        Args:
          fobj (h5py.Group):  h5py group
        """
        fobj.attrs["stiffness"] = self.stiffness

        grp = fobj.create_group("tubes")

        for name, tube in self.tubes.items():
            sgrp = grp.create_group(name)
            tube.save(sgrp)

    @classmethod
    def load(cls, fobj):
        """Load from an HDF5 file

        Args:
          fobj (h5py.Group):  h5py group containing the panel
        """
        res = cls(fobj.attrs["stiffness"])

        grp = fobj["tubes"]

        for name in grp:
            res.add_tube(Tube.load(grp[name]), name)

        return res

    def create_moose_thm_split_connector_tube(
        self,
        panel_node,
        iTube,
        manifold_tube,
        tube_roughness,
        x_tube_prev,
        y_tube_prev,
        x_tube,
        y_tube,
        tube_height,
        panel_in_out_length,
    ):
        """
        Create thm tubes to connect tubes within a panel when the connector
        tube must be split for panel in/out tubes. This also creates the jcts
        and panel in/out pipes

        Args:
          panel_node (pyhit.Node): panel node that tubes live on
          manifold_tube (Tube): tube object describing the manifold pipes
          tube_roughness (double): roughness of tube for flow simulation
          x_tube_prev (double): x position of tube before current
          y_tube_prev (double): x position of tube before current
          x_tube (double): x position of current tube
          y_tube (double): x position of current tube
          tube_height (double): height of current tube to make connection at
            top and bot
          is_center_tube (bool): whether or not this tube is in center of panel.
            if it is, we will create and join panel in/out tubes
          panel_in_out_length (double): length of panel in/out tubes
        """
        # Assuming straight path between tubes
        tube_to_tube_vect = np.array([x_tube - x_tube_prev, y_tube - y_tube_prev])
        midpoint = 0.5 * np.array([x_tube + x_tube_prev, y_tube + y_tube_prev])
        dist = np.sqrt(np.dot(tube_to_tube_vect, tube_to_tube_vect))
        manifold_tube_ir = convert_mm_to_m(manifold_tube.r - manifold_tube.t)
        # bottom tube
        bot_connector_tube_1_name = f"fch_tube_{iTube-1}_to_in_bot"
        panel_node.append(
            bot_connector_tube_1_name,
            type="FlowChannel1Phase",
            position=make_moose_hit_vector([x_tube_prev, y_tube_prev, 0.0]),
            orientation=make_moose_hit_vector(
                [x_tube - x_tube_prev, y_tube - y_tube_prev, 0.0]
            ),
            n_elems=manifold_tube.nz,
            length=0.5 * dist,
            A=np.pi * manifold_tube_ir**2,
            D_h=2.0 * manifold_tube_ir,
            roughness=tube_roughness,
            fp="fp",
        )
        # find and add to jct for prev tube
        jct_name = f"jct_tube_{iTube-1}_bot"
        add_tube_to_moose_thm_jct(
            panel_node,
            jct_name,
            panel_node.name + "/" + bot_connector_tube_1_name,
            "in",
        )
        # second half of bottom tube
        bot_connector_tube_2_name = f"fch_tube_in_to_{iTube}_bot"
        panel_node.append(
            bot_connector_tube_2_name,
            type="FlowChannel1Phase",
            position=make_moose_hit_vector([midpoint[0], midpoint[1], 0.0]),
            orientation=make_moose_hit_vector(
                [x_tube - x_tube_prev, y_tube - y_tube_prev, 0.0]
            ),
            n_elems=manifold_tube.nz,
            length=0.5 * dist,
            A=np.pi * manifold_tube_ir**2,
            D_h=2.0 * manifold_tube_ir,
            roughness=tube_roughness,
            fp="fp",
        )
        # find and add to jct for this tube
        jct_name = f"jct_tube_{iTube}_bot"
        add_tube_to_moose_thm_jct(
            panel_node,
            jct_name,
            panel_node.name + "/" + bot_connector_tube_2_name,
            "out",
        )
        # create panel in tube and junction
        panel_in_tube_name = f"fch_{panel_node.name}_in"
        A_pipe = np.pi * manifold_tube_ir**2
        panel_node.append(
            panel_in_tube_name,
            type="FlowChannel1Phase",
            position=make_moose_hit_vector(
                [midpoint[0], midpoint[1], -panel_in_out_length]
            ),
            orientation=make_moose_hit_vector([0.0, 0.0, 1.0]),
            n_elems=manifold_tube.nz,
            length=panel_in_out_length,
            A=A_pipe,
            D_h=2.0 * manifold_tube_ir,
            roughness=tube_roughness,
            fp="fp",
        )
        connectivity = [
            f"{panel_node.name}/{panel_in_tube_name}:out",
            f"{panel_node.name}/{bot_connector_tube_1_name}:out",
            f"{panel_node.name}/{bot_connector_tube_2_name}:in",
        ]
        panel_node.append(
            f"jct_{panel_node.name}_t_in",
            type="VolumeJunction1Phase",
            position=make_moose_hit_vector([midpoint[0], midpoint[1], 0]),
            volume=A_pipe * 2.0,
            connections=make_moose_hit_vector(connectivity),
        )

        # first half of top tube
        top_connector_tube_1_name = f"fch_tube_{iTube-1}_to_out_top"
        panel_node.append(
            top_connector_tube_1_name,
            type="FlowChannel1Phase",
            position=make_moose_hit_vector([x_tube_prev, y_tube_prev, tube_height]),
            orientation=make_moose_hit_vector(
                [x_tube - x_tube_prev, y_tube - y_tube_prev, 0.0]
            ),
            n_elems=manifold_tube.nz,
            length=0.5 * dist,
            A=A_pipe,
            D_h=2.0 * manifold_tube_ir,
            roughness=tube_roughness,
            fp="fp",
        )
        # find and add to jct for prev tube
        jct_name = f"jct_tube_{iTube-1}_top"
        add_tube_to_moose_thm_jct(
            panel_node,
            jct_name,
            panel_node.name + "/" + top_connector_tube_1_name,
            "in",
        )
        # second half of top tube
        top_connector_tube_2_name = f"fch_tube_out_to_{iTube}_top"
        panel_node.append(
            top_connector_tube_2_name,
            type="FlowChannel1Phase",
            position=make_moose_hit_vector([midpoint[0], midpoint[1], tube_height]),
            orientation=make_moose_hit_vector(
                [x_tube - x_tube_prev, y_tube - y_tube_prev, 0.0]
            ),
            n_elems=manifold_tube.nz,
            length=0.5 * dist,
            A=A_pipe,
            D_h=2.0 * manifold_tube_ir,
            roughness=tube_roughness,
            fp="fp",
        )
        # find and add to jct for this tube
        jct_name = f"jct_tube_{iTube}_top"
        add_tube_to_moose_thm_jct(
            panel_node,
            jct_name,
            panel_node.name + "/" + top_connector_tube_2_name,
            "out",
        )
        # create panel out tube and junction
        panel_out_tube_name = f"fch_{panel_node.name}_out"
        panel_node.append(
            panel_out_tube_name,
            type="FlowChannel1Phase",
            position=make_moose_hit_vector([midpoint[0], midpoint[1], tube_height]),
            orientation=make_moose_hit_vector([0.0, 0.0, 1.0]),
            n_elems=manifold_tube.nz,
            length=panel_in_out_length,
            A=A_pipe,
            D_h=2.0 * manifold_tube_ir,
            roughness=tube_roughness,
            fp="fp",
        )
        connectivity = [
            f"{panel_node.name}/{top_connector_tube_1_name}:out",
            f"{panel_node.name}/{top_connector_tube_2_name}:in",
            f"{panel_node.name}/{panel_out_tube_name}:in",
        ]
        panel_node.append(
            f"jct_{panel_node.name}_t_out",
            type="VolumeJunction1Phase",
            position=make_moose_hit_vector([midpoint[0], midpoint[1], tube_height]),
            volume=A_pipe * 2.0,
            connections=make_moose_hit_vector(connectivity),
        )

    def create_moose_thm_connector_tube(
        self,
        panel_node,
        iTube,
        manifold_tube,
        tube_roughness,
        x_tube_prev,
        y_tube_prev,
        x_tube,
        y_tube,
        tube_height,
        panel_in_out_length,
        is_center_tube,
    ):
        """
        Create thm tubes to connect tubes within a panel. It also creates in/out
        tubes when the tube is at panel centerline.

        Args:
          panel_node (pyhit.Node): panel node that tubes live on
          manifold_tube (Tube): tube object describing the manifold pipes
          tube_roughness (double): roughness of tube for flow simulation
          x_tube_prev (double): x position of tube before current
          y_tube_prev (double): x position of tube before current
          x_tube (double): x position of current tube
          y_tube (double): x position of current tube
          tube_height (double): height of current tube to make connection at
            top and bot
          is_center_tube (bool): whether or not this tube is in center of panel.
            if it is, we will create and join panel in/out tubes
          panel_in_out_length (double): length of panel in/out tubes
        """
        # Assuming straight path between tubes
        tube_to_tube_vect = np.array([x_tube - x_tube_prev, y_tube - y_tube_prev])
        dist = np.sqrt(np.dot(tube_to_tube_vect, tube_to_tube_vect))
        manifold_tube_ir = convert_mm_to_m(manifold_tube.r - manifold_tube.t)
        # bottom tube
        bot_connector_tube_name = f"fch_tube_{iTube-1}_to_{iTube}_bot"
        panel_node.append(
            bot_connector_tube_name,
            type="FlowChannel1Phase",
            position=make_moose_hit_vector([x_tube_prev, y_tube_prev, 0.0]),
            orientation=make_moose_hit_vector(
                [x_tube - x_tube_prev, y_tube - y_tube_prev, 0.0]
            ),
            n_elems=manifold_tube.nz,
            length=dist,
            A=np.pi * manifold_tube_ir**2,
            D_h=2.0 * manifold_tube_ir,
            roughness=tube_roughness,
            fp="fp",
        )
        # find and add to jct for prev tube
        jct_name = f"jct_tube_{iTube-1}_bot"
        add_tube_to_moose_thm_jct(
            panel_node, jct_name, panel_node.name + "/" + bot_connector_tube_name, "in"
        )
        # find and add to jct for this tube
        jct_name = f"jct_tube_{iTube}_bot"
        add_tube_to_moose_thm_jct(
            panel_node, jct_name, panel_node.name + "/" + bot_connector_tube_name, "out"
        )

        # top tube
        top_connector_tube_name = f"fch_tube_{iTube-1}_to_{iTube}_top"
        panel_node.append(
            top_connector_tube_name,
            type="FlowChannel1Phase",
            position=make_moose_hit_vector([x_tube_prev, y_tube_prev, tube_height]),
            orientation=make_moose_hit_vector(
                [x_tube - x_tube_prev, y_tube - y_tube_prev, 0.0]
            ),
            n_elems=manifold_tube.nz,
            length=dist,
            A=np.pi * manifold_tube_ir**2,
            D_h=2.0 * manifold_tube_ir,
            roughness=tube_roughness,
            fp="fp",
        )
        # find and add to jct for prev tube
        jct_name = f"jct_tube_{iTube-1}_top"
        add_tube_to_moose_thm_jct(
            panel_node, jct_name, panel_node.name + "/" + top_connector_tube_name, "in"
        )
        # find and add to jct for this tube
        jct_name = f"jct_tube_{iTube}_top"
        add_tube_to_moose_thm_jct(
            panel_node, jct_name, panel_node.name + "/" + top_connector_tube_name, "out"
        )

        if is_center_tube:
            panel_in_tube_name = f"fch_{panel_node.name}_in"
            panel_node.append(
                panel_in_tube_name,
                type="FlowChannel1Phase",
                position=make_moose_hit_vector([x_tube, y_tube, -panel_in_out_length]),
                orientation=make_moose_hit_vector([0.0, 0.0, 1.0]),
                n_elems=manifold_tube.nz,
                length=panel_in_out_length,
                A=np.pi * manifold_tube_ir**2,
                D_h=2.0 * manifold_tube_ir,
                roughness=tube_roughness,
                fp="fp",
            )
            # find and add to jct for this tube
            jct_name = f"jct_tube_{iTube}_bot"
            add_tube_to_moose_thm_jct(
                panel_node, jct_name, panel_node.name + "/" + panel_in_tube_name, "out"
            )
            panel_out_tube_name = f"fch_{panel_node.name}_out"
            panel_node.append(
                panel_out_tube_name,
                type="FlowChannel1Phase",
                position=make_moose_hit_vector([x_tube, y_tube, tube_height]),
                orientation=make_moose_hit_vector([0.0, 0.0, 1.0]),
                n_elems=manifold_tube.nz,
                length=panel_in_out_length,
                A=np.pi * manifold_tube_ir**2,
                D_h=2.0 * manifold_tube_ir,
                roughness=tube_roughness,
                fp="fp",
            )
            # find and add to jct for this tube
            jct_name = f"jct_tube_{iTube}_top"
            add_tube_to_moose_thm_jct(
                panel_node, jct_name, panel_node.name + "/" + panel_out_tube_name, "in"
            )

    def create_panel_components(
        self,
        panel_node,
        panel_theta_start,
        panel_theta_end,
        rec_radius,
        tube_roughness,
        manifold_tube,
        panel_in_out_length,
    ):
        """
        Create the create components of the panel in a MOOSE THM
        input file. Uses pyhit to create MOOSE .i files

        NOTE: We have assumed the location of the panel based on its name

        Args:
          panel_node (pyhit.Node): the node for this panel in moose components
          panel_theta_start (double): rec theta at start of panel (radians)
          panel_theta_end (double): rec_theta at end of panel (radians)
          rec_radius (double): radius of receiver
          tube_roughness (double): roughness of tube for flow simulation
          manifold_tube (Tube): tube object created for manifold tubes
          panel_in_out_length (double): length of panel in/out tubes
        Returns: None
        """
        print("Creating Moose Panel!!!")
        # set tube thetas based on number of tubes analyzed
        thetas = np.linspace(panel_theta_start, panel_theta_end, self.ntubes)
        tube_xs = rec_radius * np.cos(thetas)
        tube_ys = rec_radius * np.sin(thetas)
        panel_center_theta = 0.5 * (panel_theta_start + panel_theta_end)
        # This function will create a 3D mesh for the tubes
        # to be used for heat transfer
        # This is done once per panel, but could be done once per receiver
        # if we assume that the tubes are the same on all panels
        # and the mesh is also the same
        for iTube, tube in enumerate(self.tubes.values()):
            x_tube = tube_xs[iTube]
            y_tube = tube_ys[iTube]
            tube.create_moose_thm_tube_and_jcts(
                panel_node, iTube, x_tube, y_tube, tube_roughness
            )
            if iTube != 0:
                # if this is not the first tube
                # connect it to previous tube
                tube_height = convert_mm_to_m(tube.h)
                x_tube_prev = tube_xs[iTube - 1]
                y_tube_prev = tube_ys[iTube - 1]
                if (
                    thetas[iTube - 1] < panel_center_theta < thetas[iTube]
                ):
                    # this connector crosses centerline
                    self.create_moose_thm_split_connector_tube(
                        panel_node,
                        iTube,
                        manifold_tube,
                        tube_roughness,
                        x_tube_prev,
                        y_tube_prev,
                        x_tube,
                        y_tube,
                        tube_height,
                        panel_in_out_length,
                    )

                elif thetas[iTube] == panel_center_theta:
                    # this tube is at panel centerline
                    # this is normal tubeToTube connection
                    self.create_moose_thm_connector_tube(
                        panel_node,
                        iTube,
                        manifold_tube,
                        tube_roughness,
                        x_tube_prev,
                        y_tube_prev,
                        x_tube,
                        y_tube,
                        tube_height,
                        panel_in_out_length,
                        True,
                    )
                else:
                    # this is normal tubeToTube connection
                    self.create_moose_thm_connector_tube(
                        panel_node,
                        iTube,
                        manifold_tube,
                        tube_roughness,
                        x_tube_prev,
                        y_tube_prev,
                        x_tube,
                        y_tube,
                        tube_height,
                        panel_in_out_length,
                        False,
                    )


def next_name(names):
    """Determine the next numeric string name based on a list

    Args:
      names (list): list of current names (string)
    """
    curr_ints = []
    for name in names:
        try:
            curr_ints.append(int(name))
        except ValueError:
            continue

    if len(curr_ints) == 0:
        return str(0)
    return str(max(curr_ints) + 1)


def make_moose_hit_vector(list_in):
    """
    Convert a python list into a moose hit vector

    Args: list_in (list): the python list to convert
    Returns: a vector in moose hit form
    """
    return "'" + " ".join(str(v) for v in list_in) + "'"


def convert_mm_to_m(mm_in):
    """
    Unit conversion from mm to m

    Args: mm_in (double): qty to convert
    """
    return mm_in / 1000


def convert_m_to_mm(m_in):
    """
    Unit conversion from m to mm

    Args: m_in (double): qty to convert
    """
    return m_in * 1000


def convert_kghr_to_kgs(kghr_in):
    """
    Unit conversion from kg/hr to kg/sec

    Args: kghr_in (double): qty to convert
    """
    return kghr_in / 3600


def convert_Pa_to_MPa(Pa_in):
    """
    Unit conversion from Pa to MPa

    Args: Pa_in (double): qty to convert
    """
    return Pa_in / 10**6


def convert_Wmm2_to_Wm2(Wmm2_in):
    """
    Unit conversion from W/mm^2 to W/m^2

    Args: Wmm2_in (double): qty to convert
    """
    return Wmm2_in * 1000**2


def add_tube_to_moose_thm_jct(panel_node, jct_name, tube_name, tube_in_out):
    """
    Add a tube to a junctions connection attribute in moose THM model.
    Find the jct on given node, append tube_name to its connections attr

    NOTE: may want to make this just be generic

    Args:
      panel_node (pyhit.Node): node to search for junction within
      jct_name (string): name of jct to add to
      tube_name (string): name of tube to add to jct connections
      tube_in_out (string): specifies tube_name:in or :out connection
    """
    jct_node = moosetree.find(panel_node, func=lambda n: n.name == jct_name)
    if jct_node is None:
        print(f"COULD NOT FIND JCT!!! {jct_name}")
        # BPMToDo: better error handling here
        sys.exit()
    connect = jct_node["connections"]
    connect = make_moose_hit_vector([connect, tube_name + ":" + tube_in_out])
    jct_node["connections"] = connect


class Tube:
    """Geometry, boundary conditions, and results for a single tube.

    The basic tube geometry is defined by an outer radius, thickness, and
    height.

    Results are given at fixed times and
    on a regular polar grid defined by a number of
    r, theta, and z increments.  The grid points are then deduced by
    linear subdivision in r between the outer radius and the
    outer radius - t, 0 to 2 pi, and 0 to the tube height.

    Result fields are general and provided by a list of names.
    The receiver package uses the metal temperatures, stresses,
    mechanical strains, and inelastic strains.

    Analysis results can be provided over the full 3D grid (default),
    a single 2D plane (identified by a height), or a single 1D line
    (identified by a height and a theta position)

    Boundary conditions may be provided in two ways, either
    as fluid conditions or net heat fluxes.  These are defined
    in the HeatFluxBC or ConvectionBC objects below.

    Args:
      outer_radius (float): tube outer radius
      thickness (float): tube thickness
      height (float): tube height
      nr (int): number of radial increments
      nt (int): number of circumferential increments
      nz (int): number of axial increments
      T0 (Optional[float]): initial temperature
      page (Optional[bool]): store results on disk if True
      multiplier (Optional[int]): number of tubes represented by this actual model, defaults to 1
    """

    def __init__(
        self,
        outer_radius,
        thickness,
        height,
        nr,
        nt,
        nz,
        T0=0.0,
        page=False,
        multiplier=1,
    ):
        """Initialize the tube"""
        self.r = outer_radius
        self.t = thickness
        self.h = height

        self.nr = nr
        self.nt = nt
        self.nz = nz

        self.abstraction = "3D"

        self.times = []
        self.results = {}
        self.quadrature_results = {}
        self.axial_results = {}

        self.outer_bc = None
        self.inner_bc = None
        self.pressure_bc = None

        self.T0 = T0
        self.page = page
        self.page_prefix = ""

        self.multiplier_val = multiplier

        # BPM: for use with MOOSE (default)
        self.center_point = [0, 0, 0]
        self.tube_yaw = 0.0

    @property
    def multiplier(self):
        """
        Number of actual tubes represented by this model

        Returns:
            int:    tube multiplier
        """
        return self.multiplier_val

    def copy_results(self, other):
        """Copy the results fields from one tube to another

        Parameters:
          other:      other tube object
        """
        self.results = other.results
        self.quadrature_results = other.quadrature_results
        self.axial_results = other.axial_results

    def set_paging(self, page, i):
        """Set the value of the page parameter

        Parameters:
          page:       if true store results on disk
          i:          tube number to use
        """
        self.page = page
        self.page_prefix = str(i) + "_"

    @property
    def ndim(self):
        """Number of problem dimensions

        Returns:
          int:  tube dimension
        """
        if self.abstraction == "3D":
            return 3
        elif self.abstraction == "2D":
            return 2
        elif self.abstraction == "1D":
            return 1
        else:
            raise ValueError("Tube abstraction unknown!")

    @property
    def dim(self):
        """Actual problem discretization

        Returns:
          tuple(int): tuple giving the fixed grid discretization
        """
        if self.abstraction == "3D":
            return (self.nr, self.nt, self.nz)
        elif self.abstraction == "2D":
            return (self.nr, self.nt, 1)
        elif self.abstraction == "1D":
            return (self.nr, 1, 1)
        else:
            raise ValueError("Tube abstraction unknown!")

    @property
    def mesh(self):
        """Calculate the problem mesh (should only be needed for I/O)

        Returns:
          Results of np.meshgrid over the problem discretization
        """
        r = np.linspace(self.r - self.t, self.r, self.nr)
        if self.ndim > 1:
            t = np.linspace(0, 2 * np.pi, self.nt + 1)[: self.nt]
        else:
            t = [self.angle]
        if self.ndim > 2:
            z = np.linspace(0, self.h, self.nz)
        else:
            z = [self.plane]

        return np.meshgrid(*[r, t, z], indexing="ij")

    def surface_elements(self):
        """Return an indication of which are surface elements and
        their normal vectors.

        Returns:
            logical indexing array, True if on surface
            np.array of surface normals (zeros for solids)
        """
        if self.ndim == 1:
            # Surface elements
            surface = np.zeros((self.nr - 1,), dtype=bool)
            surface[0] = True
            surface[-1] = True

            # Surface normals
            n = np.array([1.0, 0, 0])
            normals = np.zeros((self.nr - 1, 3))
            normals[0] = -n
            normals[-1] = n

        elif self.ndim == 2:
            # Surface elements
            r = np.zeros((self.nr - 1,), dtype=bool)
            r[0] = True
            r[-1] = True
            theta = np.ones((self.nt,), dtype=bool)
            surface = np.outer(r, theta).flatten()

            # Surface normals
            t = np.linspace(0, 2 * np.pi, self.nt)
            ns = np.vstack([np.cos(t), np.sin(t), np.zeros_like(t)]).T
            normals = np.zeros((self.nr - 1, self.nt, 3))
            normals[0] = -ns
            normals[-1] = ns
            normals = normals.reshape((-1, 3))

        elif self.ndim == 3:
            # Surface elements
            r = np.zeros((self.nr - 1,), dtype=bool)
            r[0] = True
            r[-1] = True
            theta = np.ones((self.nt,), dtype=bool)
            z = np.ones((self.nz - 1,), dtype=bool)
            surface = np.outer(np.outer(r, theta), z).flatten()

            # Surface normals
            t = np.linspace(0, 2 * np.pi, self.nt)
            ns = np.vstack([np.cos(t), np.sin(t), np.zeros_like(t)]).T
            normals = np.zeros((self.nr - 1, self.nt, self.nz - 1, 3))
            normals[0] = -ns[:, None]
            normals[-1] = ns[:, None]
            normals = normals.reshape((-1, 3))

        else:
            raise ValueError("Internal error: tube dimension is %i" % self.ndim)

        return surface, normals

    def element_surface_areas(self):
        """Calculate the element surface areas

        Returns:
          np.array with each element area
        """
        if self.ndim == 1:
            return self._surfacearea1d()
        elif self.ndim == 2:
            return self._surfacearea2d()
        elif self.ndim == 3:
            return self._surfacearea3d()
        else:
            raise ValueError("Internal error: tube dimension is %i" % self.ndim)

    def _surfacearea1d(self):
        """
        1D surface area calculator
        """
        # Discretizing along radial direction
        r = np.linspace(self.r - self.t, self.r, self.nr)
        surface_area = np.zeros(self.nr - 1)
        surface_area[0] = 2 * np.pi * r[0] * self.h
        surface_area[-1] = 2 * np.pi * r[-1] * self.h
        surface_area = np.concatenate((surface_area[0], surface_area[-1]))
        surface_area = surface_area.flatten()

        return surface_area

    def _surfacearea2d(self):
        """
        2D surface area calculator
        """
        # Discretizing along radial and tangential direction
        t = np.linspace(0, 2 * np.pi, self.nt + 1)
        r = np.linspace(self.r - self.t, self.r, self.nr)
        theta = np.diff(t)

        surface_area = np.zeros((self.nr - 1, self.nt))
        surface_area[0, :] = theta * r[0] * self.h
        surface_area[-1, :] = theta * r[-1] * self.h
        surface_area = np.concatenate((surface_area[0, :], surface_area[-1, :]))
        surface_area = surface_area.flatten()

        return surface_area

    def _surfacearea3d(self):
        """
        3D surface area calculator
        """
        # Discretizing along radial, tangential and axial direction
        t = np.linspace(0, 2 * np.pi, self.nt + 1)
        r = np.linspace(self.r - self.t, self.r, self.nr)
        z = np.linspace(0, self.h, self.nz)
        theta = np.diff(t)
        heights = np.diff(z)

        surface_area = np.zeros((self.nr - 1, self.nz - 1, self.nt))
        surface_area[0, :, :] = heights[:, np.newaxis] * theta[np.newaxis, :] * r[0]
        surface_area[-1, :, :] = heights[:, np.newaxis] * theta[np.newaxis, :] * r[-1]
        surface_area = np.concatenate((surface_area[0, :, :], surface_area[-1, :, :]))
        surface_area = surface_area.flatten()

        return surface_area

    def element_volumes(self):
        """Calculate the element volumes

        Returns:
          np.array with each element volume
        """
        if self.ndim == 1:
            return self._volume1d()
        elif self.ndim == 2:
            return self._volume2d()
        elif self.ndim == 3:
            return self._volume3d()
        else:
            raise ValueError("Internal error: tube dimension is %i" % self.ndim)

    def _volume1d(self):
        """
        1D volume calculator
        """
        r = np.linspace(self.r - self.t, self.r, self.nr)
        return np.pi * (r[1:] ** 2.0 - r[:-1] ** 2.0) * self.h

    def _volume2d(self):
        """
        2D volume calculator
        """
        r = np.linspace(self.r - self.t, self.r, self.nr)
        t = np.linspace(0, 2 * np.pi, self.nt + 1)
        theta = np.diff(t)

        a = np.outer(2 * r[:-1], np.sin(theta / 2))
        b = np.outer(2 * r[1:], np.sin(theta / 2))
        edge = r[1:] - r[:-1]

        h = np.sqrt(edge[:, None] ** 2.0 - ((b - a) / 2) ** 2.0)

        base = 0.5 * (a + b) * h

        return (base * self.h).flatten()

    def _volume3d(self):
        """
        3D volume calculator
        """
        r = np.linspace(self.r - self.t, self.r, self.nr)
        t = np.linspace(0, 2 * np.pi, self.nt + 1)
        z = np.linspace(0, self.h, self.nz)
        theta = np.diff(t)

        a = np.outer(2 * r[:-1], np.sin(theta / 2))
        b = np.outer(2 * r[1:], np.sin(theta / 2))
        edge = r[1:] - r[:-1]

        h = np.sqrt(edge[:, None] ** 2.0 - ((b - a) / 2) ** 2.0)

        base = 0.5 * (a + b) * h

        heights = np.diff(z)

        return np.einsum("k,ij", heights, base).flatten()

    def write_vtk(self, fname):
        """Write to a VTK file

        The tube VTK files are only used for output and
        postprocessign

        Args:
          fname (string): base filename
        """
        writer = writers.VTKWriter(self, fname)
        writer.write()

    def make_2D(self, height):
        """Abstract the tube as 2D

        Reduce to a 2D abstraction by slicing the tube at the
        indicated height

        Args:
          height (float): the height at which to slice
        """
        if height < 0.0 or height > self.h:
            raise ValueError("2D slice height must be within the tube height")

        self.abstraction = "2D"
        self.plane = height

    def make_1D(self, height, angle):
        """Abstract the tube as 1D

        Reduce to a 1D abstraction along a ray given by the provided
        height and angle.

        Args:
          height (float): the height of the ray
          angle (float): the angle, in radians
        """
        if height < 0.0 or height > self.h:
            raise ValueError("Ray height must be within the tube height")

        self.abstraction = "1D"
        self.plane = height
        self.angle = angle

    def close(self, other):
        """Check to see if two objects are nearly equal.

        Primarily used for testing

        Args:
          other (Tube): the object to compare against

        Returns:
          bool: true if the tubes are similar
        """
        base = (
            np.isclose(self.r, other.r)
            and np.isclose(self.t, other.t)
            and np.isclose(self.h, other.h)
            and (self.nr == other.nr)
            and (self.nt == other.nt)
            and (self.nz == other.nz)
            and (np.allclose(self.times, other.times))
        )

        for name, data in self.results.items():
            if name not in other.results:
                return False
            base = base and np.allclose(data, other.results[name])

        if self.outer_bc:
            if not other.outer_bc:
                return False
            base = base and self.outer_bc.close(other.outer_bc)

        if self.inner_bc:
            if not other.inner_bc:
                return False
            base = base and self.inner_bc.close(other.inner_bc)

        if self.pressure_bc:
            if not other.pressure_bc:
                return False
            base = base and self.pressure_bc.close(other.pressure_bc)

        base = base and self.abstraction == other.abstraction
        if self.abstraction == "2D" or self.abstraction == "1D":
            base = base and np.isclose(self.plane, other.plane)
        if self.abstraction == "1D":
            base = base and np.isclose(self.angle, other.angle)

        base = base and np.isclose(self.T0, other.T0)

        return base

    @property
    def ntime(self):
        """Number of time steps

        Returns:
          int:  number of time steps
        """
        return len(self.times)

    def set_times(self, times):
        """Set the times at which data is provided

        All results arrays must provide data at these
        discrete times.

        Args:
          times (np.array): time values
        """
        for _, res in self.results.items():
            if res.shape[0] != len(times):
                raise ValueError(
                    "Cannot change times to provided values, will be"
                    " incompatible with existing results"
                )
        self.times = times

    def add_results(self, name, data):
        """Add a node point result field

        Args:
          name (str): parameter set name
          data (np.array): actual results data
        """
        self._check_rdim(data.shape)
        self.results[name] = self._setup_memmap(name + "_node", data.shape)
        self.results[name][:] = data[:]

    def add_blank_results(self, name, shape):
        """Add a blank node point result field

        Args:
          name (str): parameter set name
          shape (tuple): required shape
        """
        self._check_rdim(shape)
        self.results[name] = self._setup_memmap(name + "_node", shape)

    def add_quadrature_results(self, name, data):
        """Add a result at the quadrature points

        Args:
          name (str): parameter set name
          data (np.array): actual results data
        """
        if data.shape[0] != self.ntime and name != "ghost_temperature":
            raise ValueError("Quadrature data must have time axis first!")
        self.quadrature_results[name] = self._setup_memmap(name + "_quad", data.shape)
        self.quadrature_results[name][:] = data[:]

    def add_blank_quadrature_results(self, name, shape):
        """Add a blank quadrature point result field

        Args:
          name (str): parameter set name
          shape (tuple): required shape
        """
        if shape[0] != self.ntime:
            raise ValueError("Quadrature data must have time axis first!")
        self.quadrature_results[name] = self._setup_memmap(name + "_quad", shape)

    def add_axial_results(self, name, data):
        """Add a result distributed over the tube height at nz points

        Args:
            name (str): results name
            data (np.array): data to store
        """
        if data.shape != (self.ntime, self.nz) and name[0:5] != "fluid":
            raise ValueError("Axial result field must have shape (ntime, nz)!")
        self.axial_results[name] = self._setup_memmap(name + " _axial", data.shape)
        self.axial_results[name][:] = data[:]

    def add_blank_axial_results(self, name):
        """Add a blank axial results field

        Args:
            name (str): name of field
        """
        self.axial_results[name] = self._setup_memmap(
            name + "_axial", (self.ntime, self.nz)
        )

    def _setup_memmap(self, name, shape):
        """Map array to disk if required

        Args:
          name:   field name
          shape:  required shape
        """
        if self.page:
            return np.memmap(
                self.page_prefix + name + ".dat",
                dtype=np.float64,
                mode="w+",
                shape=shape,
            )
        else:
            return np.zeros(shape)

    def _check_rdim(self, shape):
        """Verify the dimensions of a results array

        Make sure the results array aligns with the correct dimension for the
        abstraction

        Args:
          shape (tuple): input shape

        Raises:
          ValueError: If the data array shape is not correct for the problem dimensions
        """
        if self.abstraction == "3D":
            if shape != (self.ntime, self.nr, self.nt, self.nz):
                raise ValueError("Data array shape must equal ntime x nr x nt x nz!")
        elif self.abstraction == "2D":
            if shape != (self.ntime, self.nr, self.nt):
                raise ValueError("Data array shape must equal ntime x nr x nt!")
        elif self.abstraction == "1D":
            if shape != (self.ntime, self.nr):
                raise ValueError("Data array shape must equal ntime x nr!")
        else:
            raise ValueError(
                "Internal error: unknown abstraction type %s" % self.abstraction
            )

    def set_bc(self, bc, loc):
        """Set the inner or outer heat flux BC

        Args:
          bc (ThermalBC):  boundary condition object
          loc (string): location -- either "inner" or "outer" wall
        """
        if loc == "inner":
            if not np.isclose(bc.r, self.r - self.t) or not np.isclose(bc.h, self.h):
                raise ValueError("Inner BC radius must match inner tube radius!")
            self.inner_bc = bc
        elif loc == "outer":
            if not np.isclose(bc.r, self.r) or not np.isclose(bc.h, self.h):
                raise ValueError("Outer BC radius must match outer tube radius!")
            self.outer_bc = bc
        else:
            raise ValueError("Wall location must be either inner or outer")

    def set_pressure_bc(self, bc):
        """Set the pressure boundary condition

        Args:
          bc (PressureBC):  boundary condition object
        """
        self.pressure_bc = bc

    def save(self, fobj):
        """Save to an HDF5 file

        Args:
          fobj (h5py.Group):  h5py group to save to
        """
        fobj.attrs["r"] = self.r
        fobj.attrs["t"] = self.t
        fobj.attrs["h"] = self.h

        fobj.attrs["nr"] = self.nr
        fobj.attrs["nt"] = self.nt
        fobj.attrs["nz"] = self.nz

        fobj.attrs["multiplier"] = self.multiplier_val

        fobj.attrs["multiplier"] = self.multiplier_val

        fobj.attrs["abstraction"] = self.abstraction
        if self.abstraction == "2D" or self.abstraction == "1D":
            fobj.attrs["plane"] = self.plane
        if self.abstraction == "1D":
            fobj.attrs["angle"] = self.angle

        fobj.create_dataset("times", data=self.times)

        grp = fobj.create_group("results")
        for name, result in self.results.items():
            grp.create_dataset(name, data=result)

        grp = fobj.create_group("quadrature_results")
        for name, result in self.quadrature_results.items():
            grp.create_dataset(name, data=result)

        grp = fobj.create_group("axial_results")
        for name, result in self.axial_results.items():
            grp.create_dataset(name, data=result)

        if self.outer_bc:
            grp = fobj.create_group("outer_bc")
            self.outer_bc.save(grp)

        if self.inner_bc:
            grp = fobj.create_group("inner_bc")
            self.inner_bc.save(grp)

        if self.pressure_bc:
            grp = fobj.create_group("pressure_bc")
            self.pressure_bc.save(grp)

        fobj.attrs["T0"] = self.T0

    @classmethod
    def load(cls, fobj):
        """Load from an HDF5 file

        Parameters:
          fobj (h5py.Group):  h5py to load from
        """
        if "multiplier" in fobj.attrs:
            mult = fobj.attrs["multiplier"]
        else:
            mult = 1

        res = cls(
            fobj.attrs["r"],
            fobj.attrs["t"],
            fobj.attrs["h"],
            fobj.attrs["nr"],
            fobj.attrs["nt"],
            fobj.attrs["nz"],
            T0=fobj.attrs["T0"],
            multiplier=mult,
        )

        res.abstraction = fobj.attrs["abstraction"]
        if res.abstraction == "2D" or res.abstraction == "1D":
            res.plane = fobj.attrs["plane"]
        if res.abstraction == "1D":
            res.angle = fobj.attrs["angle"]

        res.set_times(np.copy(fobj["times"]))

        grp = fobj["results"]
        for name in grp:
            res.add_results(name, np.copy(grp[name]))

        grp = fobj["quadrature_results"]
        for name in grp:
            res.add_quadrature_results(name, np.copy(grp[name]))

        if "axial_results" in fobj:
            grp = fobj["axial_results"]
            for name in grp:
                res.add_axial_results(name, np.copy(grp[name]))

        if "axial_results" in fobj:
            grp = fobj["axial_results"]
            for name in grp:
                res.add_axial_results(name, np.copy(grp[name]))

        if "outer_bc" in fobj:
            res.set_bc(ThermalBC.load(fobj["outer_bc"]), "outer")

        if "inner_bc" in fobj:
            res.set_bc(ThermalBC.load(fobj["inner_bc"]), "inner")

        if "pressure_bc" in fobj:
            res.set_pressure_bc(PressureBC.load(fobj["pressure_bc"]))

        return res

    def create_moose_thm_3D_tube_mesh(self, panel_node, tube_num, x_tube, y_tube):
        """
        Creates a moose input file for tubes on this panel.
        This means making an input file and executing moose
        with --mesh-only flag

        Args:
          panel_node (pyhit.Node): pyhit node object for this panel
          tube_num (int): number designating tube within panel
          x_tube (double): x coordinate of tube base
          y_tube (double): y coordinate of tube base

        Returns:
          tube_mesh_filename (string): filename of this panel's tube mesh
        """
        # angle to rotate tube and apply flux BCs to outer face
        # tube centerline should be perp to its x,y vector
        # this generator splits tube outer face at x = 0
        # so we want to rotate another 90 deg to get sunFaceing
        # ASSUMES that center of rec is at 0,0,0
        tube_angle = np.arctan2(y_tube, x_tube) * 180 / np.pi
        # set tube center point and yaw
        self.center_point = [x_tube, y_tube, 0.0]
        # but the yaw of the tube should be just the perp angle
        self.tube_yaw = tube_angle * np.pi / 180 - np.pi / 2

        # create new root for this file
        tube_mesh_root = pyhit.Node(parent=None, hitnode=None, offset=None)
        mesh_node = tube_mesh_root.append("Mesh")
        # add ring mesh
        ring_name = "ring_2d"
        r_outer = convert_mm_to_m(self.r)
        h = convert_mm_to_m(self.h)
        mesh_node.append(
            ring_name + "_mesh",
            type="AnnularMeshGenerator",
            nr=self.nr - 1,
            nt=self.nt,
            rmin=convert_mm_to_m(self.r - self.t),
            rmax=r_outer,
        )
        # split outer boundary of tube
        mesh_node.append(
            ring_name,
            type="PatchSidesetGenerator",
            boundary="rmax",
            n_patches=2,
            input=ring_name + "_mesh",
        )
        # turn ring mesh into tube
        mesh_node.append(
            "tube",
            type="AdvancedExtruderGenerator",
            input=ring_name,
            heights=make_moose_hit_vector([1, 1, h]),
            num_layers=make_moose_hit_vector([0, 0, (self.nz - 1)]),
            direction=make_moose_hit_vector([0, 0, 1]),
            bottom_boundary="bot",
            top_boundary="top",
        )

        # rotate tube
        mesh_node.append(
            "tube_rot",
            type="TransformGenerator",
            input="tube",
            transform="ROTATE",
            vector_value=make_moose_hit_vector([0, 0, tube_angle]),
        )
        # transform tube
        mesh_node.append(
            "tube_pos",
            type="TransformGenerator",
            input="tube_rot",
            transform="TRANSLATE",
            vector_value=make_moose_hit_vector([x_tube, y_tube, 0.0]),
        )
        tube_mesh_root.append("Outputs", exodus="true")
        # write tube input to a file
        tube_mesh_moose_input = f"{panel_node.name}_tube_{tube_num}"
        pyhit.write(tube_mesh_moose_input + ".i", tube_mesh_root)
        # run moose to generate tube mesh
        try:
            moose_exec = os.environ.get("MOOSE_THM", "MOOSE_THM")
            result = subprocess.run([moose_exec, "-i", tube_mesh_moose_input + ".i", "--mesh-only"],
                                    check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"MOOSE returned error {e.returncode}")
            print(f"stderr: {e.stderr}")

        # if this runs, then output file will be below
        tube_mesh_file = f"{tube_mesh_moose_input}_in.e"
        # now I want to write flux bc data to this exodus file
        self.write_flux_bc_to_3d_tube_mesh(tube_mesh_file)
        # then create a solution user object and a function to use it
        moose_root = (panel_node.parent).parent
        user_obj_node = moosetree.find(
            moose_root, func=lambda n: n.name == "UserObjects"
        )
        if user_obj_node is None:
            # is we could not find it, make it
            user_obj_node = moose_root.append("UserObjects")
        user_obj_node.append(
            f"{tube_mesh_moose_input}_SolObj",
            type="SolutionUserObject",
            mesh=tube_mesh_file,
            system_variables="flux",
        )
        func_node = moosetree.find(moose_root, func=lambda n: n.name == "Functions")
        if func_node is None:
            # is we could not find it, make it
            func_node = moose_root.append("Functions")
        func_node.append(
            f"flux_{tube_mesh_moose_input}_SolFn",
            type="SolutionFunction",
            solution=f"{tube_mesh_moose_input}_SolObj",
        )
        return tube_mesh_file

    def write_flux_bc_to_3d_tube_mesh(self, tube_mesh_file):
        """
        Writes flux bcs defined in srlife to 3D tube mesh in exodus
        This allows moose to generate a solution function to apply time
        and space dependent flux bcs

        Args:
          tube_mesh_file: exodus filename of tube to write to
        """
        model = exo.exodus(tube_mesh_file, array_type="numpy", mode="a")
        exo.add_variables(model, nodal_vars=["flux"])
        # then loop over side set nodes with on rmax_0
        rmax_0_id = 2
        _, flux_bc_nodes = model.get_side_set_node_list(rmax_0_id)
        if model.get_side_set_name(rmax_0_id) != "rmax_0":
            sys.exit("PROBLEM!!!")
        # we only want to operate on each node once
        flux_bc_nodes = np.unique(flux_bc_nodes)
        flux_data = np.zeros((len(self.times), model.num_nodes()))
        for node in flux_bc_nodes:
            # get coords in MOOSE x,y,z coordinates (expects node index)
            x, y, z = model.get_coord(node)
            coord = np.array([x, y, z])
            # get srlife tube coords (note: could just map coords)
            _, theta, z = self.map_moose_coords_to_srlife_tube_coords(coord)
            z_tube = convert_m_to_mm(z)
            for iTime, time in enumerate(self.times):
                # Get the correct value of flux
                flux_data[iTime, node - 1] = convert_Wmm2_to_Wm2(
                    self.outer_bc.flux(time, theta, z_tube)
                )
        # make all times and variables in exodus
        for iTime, time in enumerate(self.times):
            model.put_time(iTime + 1, time * 3600)
            model.put_node_variable_values("flux", iTime + 1, flux_data[iTime])
        # print(model.get_node_variable_values("flux", 2))
        model.close()

    def map_moose_coords_to_srlife_tube_coords(self, coords):
        """
        Check if a set of moose node coords is on the outer boundary
        of the tube

        Args:
          coodrs (list): [x, y, z] in moose analysis coordinates
        Returns:
          r (double): tube radial coord
          theta (double): tube theta coord
          z (double): tube z coord
        """
        # transform to tube coords
        # translate via self.center_point
        # rotate by self.tube_yaw
        c_yaw = np.cos(self.tube_yaw)
        s_yaw = np.sin(self.tube_yaw)
        R = np.array([[c_yaw, s_yaw, 0], [-s_yaw, c_yaw, 0], [0, 0, 1]])
        coord_tube = R @ np.array(coords - self.center_point)
        x_tube = coord_tube[0]
        y_tube = coord_tube[1]
        z_tube = coord_tube[2]
        # radial coords
        r = np.sqrt(x_tube**2 + y_tube**2)
        theta = np.arctan2(y_tube, x_tube)
        if theta < 0.0:
            # we want 0,2pi, not -pi,pi
            theta += 2.0 * np.pi
        return r, theta, z_tube

    def create_moose_thm_tube_and_jcts(
        self, panel_node, tube_num, x_tube, y_tube, tube_roughness
    ):
        """
        Create a tube node in a moose input file using pyhit.

        Args:
          panel_node (pyhit.Node): panel node in moose input file
          tube_num (int): number designating tube within panel
          x_tube (double): x position of tube
          y_tube (double): y position of tube
          tube_roughness (double): roughness of tube for flow simulation
        """
        tube_ir = convert_mm_to_m(self.r - self.t)
        tube_fch_name = f"fch_tube_{tube_num}"
        A_pipe = np.pi * (tube_ir) ** 2
        panel_node.append(
            tube_fch_name,
            type="FlowChannel1Phase",
            position=make_moose_hit_vector([x_tube, y_tube, 0]),
            orientation=make_moose_hit_vector([0, 0, 1]),
            n_elems=(self.nz - 1),
            length=convert_mm_to_m(self.h),
            A=A_pipe,
            D_h=2.0 * (tube_ir),
            roughness=tube_roughness,
            fp="fp",
        )
        # we are just initializing these now
        # will tie in with connectors
        panel_node.append(
            f"jct_tube_{tube_num}_bot",
            type="VolumeJunction1Phase",
            position=make_moose_hit_vector([x_tube, y_tube, 0]),
            volume=A_pipe * 2.0,
            connections=make_moose_hit_vector(
                [f"{panel_node.name}/fch_tube_{tube_num}:in"]
            ),
        )
        panel_node.append(
            f"jct_tube_{tube_num}_top",
            type="VolumeJunction1Phase",
            position=make_moose_hit_vector([x_tube, y_tube, convert_mm_to_m(self.h)]),
            volume=A_pipe * 2.0,
            connections=make_moose_hit_vector(
                [f"{panel_node.name}/fch_tube_{tube_num}:out"]
            ),
        )
        # make 3D tube for heat transfer
        panel_tube_mesh_filename = self.create_moose_thm_3D_tube_mesh(
            panel_node, tube_num, x_tube, y_tube
        )
        # add tube heat structure and heat transfer
        # NOTE: since tube mesh is at correct pos, do not need to set position here
        heat_tube_name = f"heat_tube_{tube_num}"
        panel_node.append(
            heat_tube_name,
            type="HeatStructureFromFile3D",
            file=panel_tube_mesh_filename,
            position=make_moose_hit_vector([0.0, 0.0, 0.0]),
        )
        panel_node.append(
            f"heat_transfer_tube_{tube_num}",
            type="HeatTransferFromHeatStructure3D1Phase",
            flow_channels=f"{panel_node.name}/{tube_fch_name}",
            hs=f"{panel_node.name}/{heat_tube_name}",
            boundary=f"{panel_node.name}/{heat_tube_name}:rmin",
            P_hf=2 * np.pi * tube_ir,
        )
        # add flux BCs for 3D tube
        moose_root = (panel_node.parent).parent
        bc_node_name = "BCs"
        bc_node = moosetree.find(moose_root, func=lambda n: n.name == bc_node_name)
        if bc_node is None:
            # is we could not find it, make it
            bc_node = moose_root.append("BCs")
        bc_func_name = f"flux_{panel_node.name}_tube_{tube_num}_SolFn"
        bc_node.append(
            f"flux_{panel_node.name}_tube_{tube_num}",
            type="FunctionNeumannBC",
            variable="T_solid",
            boundary=f"{panel_node.name}/{heat_tube_name}:rmax_0",
            function=bc_func_name,
        )

    def heat_flux_data_to_moose_thm_data(self, func_name):
        """
        Take tube HeatFluxBC.data and parse into PiecewiseMultilinear
        compliant file for use with MOOSE

        Args:
          func_name (String): name of function for this tube on this panel

        Returns:
          axis_t (list): list of times in seconds
          axis_x (list): list of x coordinates of data in MOOSE coords
          axis_y (list): list of y coordinates of data in MOOSE coords
          axis_z (list): list of z coordinates of data in MOOSE coords
          data (list(list)): array of heat flux values in W/m^2
            [time, nx, ny, nz]
        """
        bc = self.outer_bc
        if bc is None:
            print("NO FLUX BC ON TUBE???")
        t = self.times * 3600
        # map srlife node index to tube coords
        r = convert_mm_to_m(self.r)
        thetas = np.linspace(np.pi, 0, self.nt)
        h = convert_mm_to_m(self.h)
        x_hats = np.array([r * np.cos(theta) for theta in thetas])
        y_hats = np.array([r * np.sin(theta) for theta in thetas])
        z_hats = np.linspace(0, h, self.nz)
        # map tube coords to MOOSE THM coords
        x = np.zeros_like(x_hats)
        data = np.zeros([len(t), len(x), len(z_hats)])
        for iTime, time in enumerate(t):
            hour_data = np.zeros([len(x), len(z_hats)])
            for iX, x_hat in enumerate(x_hats):
                theta = thetas[iX]
                y_hat = y_hats[iX]
                for iZ, z_hat in enumerate(z_hats):
                    x[iX], _, z = self.map_tube_coords_to_moose_thm_coords(
                        x_hat, y_hat, z_hat
                    )
                    hour_data[iX, iZ] = convert_Wmm2_to_Wm2(
                        bc.flux(time / 3600, theta, convert_m_to_mm(z))
                    )
            # now I need to sort everything to be monotonically increasing
            x_order = np.argsort(x)
            x = x[x_order]
            z_order = np.argsort(z_hats)
            z_hats = z_hats[z_order]
            hour_data = hour_data[x_order, :]
            hour_data = hour_data[:, z_order]
            data[iTime] = hour_data
        return t, x, z_hats, data

    def map_tube_coords_to_moose_thm_coords(self, x_hat, y_hat, z_hat):
        """
        Takes tube coordinates and maps them to MOOSE THM simulation coords
        x = xc + R@x_hat

        Args:
          x_hat (double): tube x coordinates
          y_hat (double): tube y coordinates
          z_hat (double): tube z coordinates
        Returns:
          x (double): moose thm x coords
          y (double): moose thm y coords
          z (double): moose thm z coords
        """
        coord_tube = np.array([x_hat, y_hat, z_hat])
        # rotation matrix
        c_yaw = np.cos(self.tube_yaw)
        s_yaw = np.sin(self.tube_yaw)
        R = np.array([[c_yaw, -s_yaw, 0], [s_yaw, c_yaw, 0], [0, 0, 1]])
        # transformation
        coord = R @ coord_tube + np.array(self.center_point)
        x = coord[0]
        y = coord[1]
        z = coord[2]
        return x, y, z

    def map_nodes_moose_to_srlife(self, coords):
        """
        Map nodal coordinates to nodal indices for result data coming from
        MOOSE to be loaded into srlife Tube objects.
        Tube stores results as [time, nr, nt, nz] structs
        Data from MOOSE is [time, all_tube_nodes]

        Args:
          coords (list[double]): nodal coordinates

        Returns:
          r_ind, t_ind, z_ind (int): indices for current node from MOOSE
            into srlife data structure
        """
        # transform to tube coords
        # translate via self.center_point
        # rotate by self.tube_yaw
        c_yaw = np.cos(self.tube_yaw)
        s_yaw = np.sin(self.tube_yaw)
        R = np.array([[c_yaw, s_yaw, 0], [-s_yaw, c_yaw, 0], [0, 0, 1]])
        coord_tube = R @ np.array(coords - self.center_point)
        x = coord_tube[0]
        y = coord_tube[1]
        z = coord_tube[2]
        # radial coords
        r = np.sqrt(x**2 + y**2)
        # correct with tube rotation
        theta = np.pi + np.arctan2(y, x)
        # helper q's
        t = convert_mm_to_m(self.t)
        ir = convert_mm_to_m(self.r - self.t)
        h = convert_mm_to_m(self.h)
        r_hat = r - ir

        # indices
        r_ind = int(round((self.nr - 1) / t * r_hat))
        t_ind = int(round((self.nt) / (2 * np.pi) * theta))
        z_ind = int(round((self.nz - 1) / h * z))
        if t_ind == self.nt:
            # have to correct for wrapped angle
            t_ind = 0
        return r_ind, t_ind, z_ind

    def load_moose_thm_results(self, times, press_data, temp_data):
        """
        Load solution data from MOOSE THM simulation into tube result data structures

        Args:
          times (list): list of analysis times
          press_data (np.array): Array nodal values for pressure along tube length
            [time, node]
         temp_data (list(np.array)): list of two arrays
            1 - nodal coordinates by index [node]
            2 - nodal values for temperature in tube with time [time, node]
        """
        # take times from seconds to hours
        times = np.array([it / 3600 for it in times])
        # set tube pressure BC
        # downsample pressure data to a single value per time
        # also convert from Pa to MPa
        press = np.array(
            [convert_Pa_to_MPa(np.average(press_step)) for press_step in press_data]
        )

        bc = PressureBC(times, press)
        self.set_pressure_bc(bc)
        # add temp results to tube
        coords = temp_data[0]
        temps = temp_data[1]
        # loop through data
        self.set_times(times)
        tube_nod_temps = np.zeros([len(times), self.nr, self.nt, self.nz])
        repeats = 0
        coord_catch = set()
        for iCoord, coord in enumerate(coords):
            # for each node, get index mapping
            r_ind, t_ind, z_ind = self.map_nodes_moose_to_srlife(coord)
            coord_tup = (r_ind, t_ind, z_ind)
            if coord_tup in coord_catch:
                print("ERROR!!!!")
                print(coord_tup)
                repeats += 1
                print(f"{repeats} repeated coords!!!")
            else:
                coord_catch.add(coord_tup)
            for iStep, temp in enumerate(temps):
                # for each timestep, get temp of this node
                tube_nod_temps[iStep, r_ind, t_ind, z_ind] = temp[iCoord]
        self.add_results("temperature", tube_nod_temps)


def _vector_interpolate(base, data):
    """Interpolate as a vector

    Args:
      base (function): base interpolation method
      data (np.array): data to interpolate
    """
    res = np.zeros(data[0].shape)

    # pylint: disable=not-an-iterable
    for ind in np.ndindex(*res.shape):
        res[ind] = base([d[ind] for d in data])

    return res


def _make_ifn(base):
    """Helper to deal with getting both a scalar and a vector input

    Args:
      base (function): base interpolation method

    Returns:
      function: a function that interpolates a vector using base at each component
    """

    def ifn(mdata):
        """
        Interpolation function that handles both the scalar and vector cases
        """
        allscalar = all(map(np.isscalar, mdata))
        anyscalar = any(map(np.isscalar, mdata))
        if allscalar:
            return base(mdata)
        elif anyscalar:
            shapes = [a.shape for a in mdata if not np.isscalar(a)]
            # Could check they are all the same, but eh
            shape = shapes[0]
            ndata = [np.ones(shape) * d for d in mdata]
            return _vector_interpolate(base, ndata)
        else:
            return _vector_interpolate(base, ndata)

    return ifn


class PressureBC:
    """Stores information about the tube internal pressure

    Simple class to store tube pressure, assumed to be constant
    in space and just vary with time.

    Args:
      times (np.array): times throughout load cycle
      data (np.array):  pressure values
    """

    def __init__(self, times, data):
        """Initialize the PressureBC"""
        self.times = times
        if self.times.shape != data.shape:
            raise ValueError("Times and data should have the same shape!")
        self.data = data

        self.ifn = inter.interp1d(self.times, self.data)

    @classmethod
    def load(cls, fobj):
        """Load from an HDF5 file

        Args:
          fobj (h5py.Group):  h5py group to load from
        """
        return cls(np.copy(fobj["times"]), np.copy(fobj["data"]))

    def save(self, fobj):
        """Save to an HDF5 file

        Args:
          fobj (h5py.Group):  h5py group to save to
        """
        fobj.create_dataset("times", data=self.times)
        fobj.create_dataset("data", data=self.data)

    @property
    def ntime(self):
        """Number of time steps

        Returns:
          int:      number of time steps in the pressure history definition
        """
        return len(self.times)

    def pressure(self, t):
        """Return the pressure as a function of time

        Args:
          t (float): time

        Returns:
          float: internal pressure at that time
        """
        return self.ifn(t)

    def close(self, other):
        """Test method for comparing BCs

        Args:
          other (PressureBC): other object

        Returns:
          bool: true if the objects are sufficiently similar
        """
        return np.allclose(self.times, other.times) and np.allclose(
            self.data, other.data
        )


class ThermalBC:
    """Superclass for thermal boundary conditions.

    Currently just here to handle dispatch for HDF files
    """

    @classmethod
    def load(cls, fobj):
        """Load from an HDF5 file

        Args:
          fobj (h5py.Group): h5py group to load from
        """
        if fobj.attrs["type"] == "HeatFlux":
            return HeatFluxBC.load(fobj)
        elif fobj.attrs["type"] == "Convective":
            return ConvectiveBC.load(fobj)
        elif fobj.attrs["type"] == "FixedTemp":
            return FixedTempBC.load(fobj)
        elif fobj.attrs["type"] == "FilmCoefficientConvective":
            return FilmCoefficientConvectiveBC.load(fobj)
        else:
            raise ValueError("Unknown BC type %s" % fobj.attrs["type"])

    # pylint: disable=no-member
    def _generate_surface_mesh(self):
        """Make the finite difference mesh for the BC

        Generate the appropriate finite difference surface
        mesh for a particular problem

        Returns:
          np.array:   discrete times
          np.array:   theta coordinates
          np.array:   z coordinates
        """
        ts = np.linspace(0, 2 * np.pi, self.nt + 1)[:-1]
        zs = np.linspace(0, self.h, self.nz)
        return self.times, ts, zs

    def _generate_ifn(self, data):
        """Generate an interpolation function for the given data array

        Args:
          data (np.array):  (ntime, ntheta, nz) shaped array

        Returns:
          function: appropriate interpolation function
        """
        base = inter.RegularGridInterpolator(
            self._generate_surface_mesh(),
            data,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        return _make_ifn(base)


class HeatFluxBC(ThermalBC):
    """
    A net heat flux on the radius of a tube.  Positive is heat input,
    negative is heat output.

    These conditions are defined on the surface of a tube at fixed
    times given by
    a radius and a height.  The radius is not used in defining the
    BC but is used to ensure the BC is consistent with the Tube object.

    The heat flux is given on a regular grid of theta, z points each defined
    but a number of increments.  This grid need not agree with the Tube
    solid grid.

    Args:
      radius (float): boundary condition application radius
      height (float): tube height
      nt (int): number of circumferential increments
      nz (int): number of axial increments
      times (np.array): heat flux times
      data (np.array): heat flux data
    """

    def __init__(self, radius, height, nt, nz, times, data):
        self.r = radius
        self.h = height

        self.nt = nt
        self.nz = nz

        self.times = times

        if data.shape != (len(self.times), nt, nz):
            raise ValueError("Heat flux shape must equal ntime x ntheta x nz!")

        self.data = data

        self.ifn = self._generate_ifn(self.data)

    @property
    def ntime(self):
        """Number of time steps

        Returns:
          int:  number of time steps
        """
        return len(self.times)

    def flux(self, t, theta, z):
        """Flux as a function of time, angle, and height

        Args:
          t (float): time
          theta (float): angle
          z (float): height

        Returns:
          float:  the flux value at this time and location
        """

        res = self.ifn([t, theta, z])
        return res

    def save(self, fobj):
        """Save to an HDF5 file

        Args:
          fobj (h5py.Group): h5py group to save to
        """
        fobj.attrs["type"] = "HeatFlux"
        fobj.attrs["r"] = self.r
        fobj.attrs["h"] = self.h

        fobj.attrs["nt"] = self.nt
        fobj.attrs["nz"] = self.nz

        fobj.create_dataset("times", data=self.times)
        fobj.create_dataset("data", data=self.data)

    @classmethod
    def load(cls, fobj):
        """Load from an HDF5 file

        Args:
          fobj (h5py.Group): h5py group to load from
        """
        return cls(
            fobj.attrs["r"],
            fobj.attrs["h"],
            fobj.attrs["nt"],
            fobj.attrs["nz"],
            np.copy(fobj["times"]),
            np.copy(fobj["data"]),
        )

    def close(self, other):
        """Check to see if two objects are nearly equal.

        Primarily used for testing

        Args:
          other (HeatFluxBC): the object to compare against

        Returns:
          bool: returns true if the objects are similar
        """
        return (
            np.isclose(self.r, other.r)
            and np.isclose(self.h, other.h)
            and (self.nt == other.nt)
            and (self.nz == other.nz)
            and np.allclose(self.times, other.times)
            and np.allclose(self.data, other.data)
        )


class FixedTempBC(ThermalBC):
    """Fixed temperature BC.

    These conditions are defined on the surface of a tube at fixed
    times given by
    a radius and a height.  The radius is not used in defining the
    BC but is used to ensure the BC is consistent with the Tube object.

    The heat flux is given on a regular grid of theta, z points each defined
    but a number of increments.  This grid need not agree with the Tube
    solid grid.

    Args:
      radius (float): boundary condition application radius
      height (float): tube height
      nt (int): number of circumferential increments
      nz (int): number of axial increments
      times (np.array): fixed temperature times
      data (np.array): fixed temperature data
    """

    def __init__(self, radius, height, nt, nz, times, data):
        self.r = radius
        self.h = height

        self.nt = nt
        self.nz = nz

        self.times = times

        if data.shape != (len(self.times), nt, nz):
            raise ValueError(
                "Discrete temperature shape must equal ntime x ntheta x nz!"
            )

        self.data = data

        self.ifn = self._generate_ifn(self.data)

    @property
    def ntime(self):
        """Number of time steps

        Returns:
          int:  number of time steps
        """
        return len(self.times)

    def temperature(self, t, theta, z):
        """Return the temperature at a given time and position

        Args:
          t (float): time
          theta (float): angle
          z (float):  height

        Returns:
          float: Fixed temperature at given time/location
        """
        return self.ifn([t, theta, z])

    def save(self, fobj):
        """Save to an HDF5 file

        Args:
          fobj (h5py.Group):  h5py group to save to
        """
        fobj.attrs["type"] = "FixedTemp"
        fobj.attrs["r"] = self.r
        fobj.attrs["h"] = self.h

        fobj.attrs["nt"] = self.nt
        fobj.attrs["nz"] = self.nz

        fobj.create_dataset("times", data=self.times)
        fobj.create_dataset("data", data=self.data)

    @classmethod
    def load(cls, fobj):
        """Load from an HDF5 file

        Args:
          fobj (h5py.Group): h5py group to load from
        """
        return cls(
            fobj.attrs["r"],
            fobj.attrs["h"],
            fobj.attrs["nt"],
            fobj.attrs["nz"],
            np.copy(fobj["times"]),
            np.copy(fobj["data"]),
        )

    def close(self, other):
        """Check to see if two objects are nearly equal.

        Primarily used for testing

        Args:
          other (FixedTempBC): the object to compare against

        Returns:
          bool: true if the objects are similar
        """
        return (
            np.isclose(self.r, other.r)
            and np.isclose(self.h, other.h)
            and (self.nt == other.nt)
            and (self.nz == other.nz)
            and np.allclose(self.times, other.times)
            and np.allclose(self.data, other.data)
        )


class FilmCoefficientConvectiveBC(ThermalBC):
    """A convective BC on the ID of a tube, this version provides the film coefficient directly

    Args:
        radius (float):     radius of application
        height (float):     height of tube
        nz (int):           number of divisions along height
        fluid_T (np.array): fluid temperature
        film (np.array):    film coefficient data
    """

    def __init__(self, radius, height, nz, fluid_T, film):
        self.r = radius
        self.h = height

        self.nz = nz

        self.fluid_T = fluid_T
        self.film = film

        if fluid_T.shape != (nz,) or film.shape != (nz,):
            raise ValueError(
                "Film coefficient and fluid temperature data must have size (nz,)"
            )

        zs = np.linspace(0, self.h, self.nz)
        self.ifn_fluid = inter.interp1d(zs, fluid_T)
        self.ifn_film = inter.interp1d(zs, film)

    def fluid_temperature(self, t, z):
        """Return the fluid temperature at a given time and position

        Args:
            t (float): time
            z (float): height
        """
        return self.ifn_fluid(z)

    def film_coefficient(self, t, z):
        """Return the film coefficient at a given time and position

        Args:
          t (float): time
          z (float): height

        Return:
          float: film coefficient at this location and time
        """
        return self.ifn_film(z)

    def save(self, fobj):
        """Save to an HDF5 file

        Args:
          fobj (h5py.Group): h5py group to save to
        """
        fobj.attrs["type"] = "FilmCoefficientConvective"
        fobj.attrs["r"] = self.r
        fobj.attrs["h"] = self.h

        fobj.attrs["nz"] = self.nz

        fobj.create_dataset("fluid_T", data=self.fluid_T)
        fobj.create_dataset("film", data=self.film)

    @classmethod
    def load(cls, fobj):
        """Load from an HDF5 file

        Args:
          fobj (h5py.Group): h5py group to load from
        """
        return cls(
            fobj.attrs["r"],
            fobj.attrs["h"],
            fobj.attrs["nz"],
            np.copy(fobj["fluid_T"]),
            np.copy(fobj["film"]),
        )

    def close(self, other):
        """Check to see if two objects are nearly equal.

        Primarily used for testing

        Args:
          other (ConvectiveBC): the object to compare against

        Returns:
          bool: true if sufficiently similar
        """
        return (
            np.isclose(self.r, other.r)
            and np.isclose(self.h, other.h)
            and (self.nz == other.nz)
            and np.allclose(self.fluid_T, other.fluid_T)
            and np.allclose(self.film, other.film)
        )


class ConvectiveBC(ThermalBC):
    """A convective BC on the surface of a tube defined by a radius and height.

    The radius is not used in defining the BC, but is used to check
    consistency with the Tube object.

    This condition is defined axially by a fluid temperature at
    fixed times on a fixed grid of z points defined by a number of
    increments.

    Args:
      radius (float): radius of application
      height (float): height of fluid temperature info
      nz (int): number of axial increments
      times (np.array): data times
      data (np.array): actual fluid temperature data
    """

    def __init__(self, radius, height, nz, times, data):
        self.r = radius
        self.h = height

        self.nz = nz

        self.times = times

        if data.shape != (len(self.times), nz):
            raise ValueError("Fluid temperature data shape must equal " "ntime x nz!")

        self.data = data

        zs = np.linspace(0, self.h, self.nz)
        base = inter.RegularGridInterpolator(
            (self.times, zs),
            self.data,
            bounds_error=False,
            fill_value=None,
            method="linear",
        )

        self.ifn = _make_ifn(base)

    @property
    def ntime(self):
        """Number of time steps

        Returns:
          int:  number of discrete time steps
        """
        return len(self.times)

    def fluid_temperature(self, t, z):
        """Return the fluid temperature at a given time and position

        Args:
          t (float): time
          z (float): height

        Return:
          float: fluid temperature at this location and time
        """
        return self.ifn([t, z])

    def save(self, fobj):
        """Save to an HDF5 file

        Args:
          fobj (h5py.Group): h5py group to save to
        """
        fobj.attrs["type"] = "Convective"
        fobj.attrs["r"] = self.r
        fobj.attrs["h"] = self.h

        fobj.attrs["nz"] = self.nz

        fobj.create_dataset("times", data=self.times)

        fobj.create_dataset("data", data=self.data)

    @classmethod
    def load(cls, fobj):
        """Load from an HDF5 file

        Args:
          fobj (h5py.Group): h5py group to load from
        """
        return cls(
            fobj.attrs["r"],
            fobj.attrs["h"],
            fobj.attrs["nz"],
            np.copy(fobj["times"]),
            np.copy(fobj["data"]),
        )

    def close(self, other):
        """Check to see if two objects are nearly equal.

        Primarily used for testing

        Args:
          other (ConvectiveBC): the object to compare against

        Returns:
          bool: true if sufficiently similar
        """
        return (
            np.isclose(self.r, other.r)
            and np.isclose(self.h, other.h)
            and (self.nz == other.nz)
            and np.allclose(self.times, other.times)
            and np.allclose(self.data, other.data)
        )
