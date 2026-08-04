"""
  This module has all of the helper functions needed to run an srlife analysis.
  It is meant to allow a simple script interface to define a problem and run the
  analysis.

  One such example is defined in test/test_interface.py
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# srlife specific functions
from srlife import (
    receiver,
    solverparams,
)


# UNIT CONVERSION FUNCTIONS
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


def convert_kWm2_to_Wmm2(kWm2_in):
    """
    Unit conversion from kW/m^2 to W/mm^2

    Args: kWm2_in (double): qty to convert
    """
    return kWm2_in * 1000e-6


def convert_Pa_to_MPa(Pa_in):
    """
    Unit conversion from Pa to MPa

    Args: Pa_in (double): qty to convert
    """
    return Pa_in * 1e-6


def convert_MPa_to_Pa(MPa_in):
    """
    Unit conversion from MPa to Pa

    Args: MPa_in (double): qty to convert
    """
    return MPa_in * 1e6


def convert_C_to_K(C_in):
    """
    Unit conversion from C to K

    Args: C_in (double): qty to convert
    """
    return C_in + 273.15


def read_month_day_hour_flux_file(flux_file_name, data_shape, z_offset_in):
    """
    Read and process the flux data from a solarPILOT file

    Args:
      flux_file_name (string): path and name of flux file from solarpilot
      data_shape (list[int]): list with n_rows and n_cols of flux data
      z_offset_in (double): z offset between srlife model and solarpilot model
        this will shift z data that is read up or down

    Returns:
      z_data (np.array(double)): vector with z positions from flux sampling
      theta_data (np.array(double)): vector with theta positions from flux sampling
      flux_data (np.array(double)): matrix with flux data at sampled positions
    """
    # read flux data
    # massage into format
    # output
    row, col = data_shape
    z_data = np.zeros([row, 1])
    theta_data = np.zeros([1, col])
    flux_data = np.zeros([row, col])
    try:
        theta_data = np.loadtxt(
            flux_file_name, delimiter=",", skiprows=6, usecols=range(1, col + 1)
        )[0]
        z_data = np.loadtxt(
            flux_file_name, delimiter=",", skiprows=7, usecols=range(0, 1)
        )
        z_data += z_offset_in
        flux_data = np.loadtxt(
            flux_file_name, delimiter=",", skiprows=7, usecols=range(1, col + 1)
        )
        # Data is on cylinder, so 180 will repeat
        theta_data = np.append(theta_data, -180)
        flux_data = np.hstack((flux_data, flux_data[:, 0][:, None]))
    except FileNotFoundError:
        # if we cannot find the file
        # print a message and just return all zeros
        print(
            f"File could not be found!!! Returning all zeros\n filename: {flux_file_name}"
        )
    # convert height to mm
    z_data = convert_m_to_mm(z_data)
    return z_data, theta_data, flux_data


def get_flux_interpolators_from_data_files(
    times, start_time, end_time, month, day, flux_data_dir, flux_data_shape, z_offset
):
    """
    Read flux files and create interpolator functions for each month, day, hour

    Args:
      times (list(int)): list of index of hours used for analysis
        Could probably remove this if (end_time - start_time) == len(times)
      start_time (int): first hour to read flux data
      end_time (int): last hour to read flux data
      month (int): month to read flux data
      day (int): day to read flux data
      flux_data_dir (string): path to flux data files
      flux_data_shape (list(int)): n_rows and n_cols of flux data in flux files
      z_offset_in (double): z offset between srlife model and solarpilot model
        this will shift z data that is read up or down

    Returns:
      flux_interpolators_by_hour (list(RegularGridInterpolators)): list with the interpolating
        functions for each hour in times.
    """
    print("Making flux interpolators from data")
    # Time zero is NOT analyzed, so we do not make a fluxFn for that
    flux_interpolators_by_hour = [0] * (len(times) - 1)
    for i_count, i_time in enumerate(range(start_time, end_time + 1)):
        flux_filename = f"{flux_data_dir}/data_{month}_d{day}_hr{int(i_time)}.csv"
        z_pts, theta_pts, flux_pts = read_month_day_hour_flux_file(
            flux_filename, flux_data_shape, z_offset
        )
        interpFun = RegularGridInterpolator(
            (z_pts, theta_pts),
            flux_pts,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        flux_interpolators_by_hour[i_count] = interpFun
    return flux_interpolators_by_hour


def apply_tube_flux_bcs(
    tube,
    tube_absorbance,
    tube_rec_theta,
    flux_interpolators_by_hour,
):
    """
    Actually apply the heat flux to the tube across width and length

    Args:
      tube (srlife.Tube): srlife object for this tube
      tube_z (list[double]): list of z coordinates of tube nodes
      tube_theta (list[double]): list of theta coordinates of tube nodes
      tube_absorbance (double): absorbance of tubes
      tube_rec_theta (double): receiver theta value for this tube
      flux_interpolators_by_hour (list[RegularGridInterpolator]):
        list of interpolating functions for heat flux on tubes
    """
    num_steps = len(flux_interpolators_by_hour)
    tube_theta = np.linspace(0, 360, tube.nt + 1)[:-1]
    tube_z = np.linspace(0, tube.h, tube.nz)
    # add extra time step for time zero
    tube_flux = np.zeros([num_steps + 1, len(tube_theta), len(tube_z)])
    for i_hour in range(num_steps):
        # for each hour of flux data
        # sample tube at tube_rec_theta and z
        for i_z, z in enumerate(tube_z):
            # for each zPt
            pt_z_theta = [z, tube_rec_theta]
            rec_flux = convert_kWm2_to_Wmm2(
                flux_interpolators_by_hour[i_hour](pt_z_theta)
            )
            # apply across crown so only sunSide gets flux
            for i_theta, theta in enumerate(tube_theta):
                # for each thetaPt around tube
                if 0 < theta < 180:
                    tube_flux[i_hour + 1, i_theta, i_z] = rec_flux[0] * np.cos(
                        (theta - 90) * np.pi / 180
                    )
                else:
                    tube_flux[i_hour + 1, i_theta, i_z] = 0.0
    # make bc object for tube with heat flux data
    # NOTE: this is done here and not in cerate_tube because if we initialize
    # bc data to np.zeros, flux will always be zero unless we reload from a file
    tube.set_bc(apply_heatflux_bc(tube, tube_flux * tube_absorbance, "outer"), "outer")
    # while doing this, also apply convective BCs to inner face
    tube.set_bc(apply_convective_bc(tube, "inner"), "inner")


def apply_convective_bc(tube, loc):
    """
    Defines and applies a convective BC to a surface of a tube
    specificed by loc

    Args:
      tube (srlife.Tube): srlife object for this tube
      loc (string): "inner" or "outer" to specify which surface BC is applied to

    """
    nz = tube.nz
    if loc == "outer":
        r = tube.r
    else:
        r = tube.r - tube.t
    return receiver.ConvectiveBC(
        r, tube.h, nz, tube.times, np.zeros((len(tube.times), nz))
    )


def apply_heatflux_bc(tube, data, loc):
    """
    Defines and applied a heat flux BC to a surface of a tube object
    specified by loc

    Args:
      tube (srlife.Tube): srlife object for this tube
      data (list[double]): 3d list of heat flux bcs [hour, theta, z]
      loc (string): "inner" or "outer" to specify which surface BC is applied to

    """
    if loc == "outer":
        r = tube.r
    else:
        r = tube.r - tube.t
    nt = tube.nt
    nz = tube.nz
    return receiver.HeatFluxBC(r, tube.h, nt, nz, tube.times, data)


def calc_and_write_tube_flux_bcs(
    rec,
    ass_tube_per_panel,
    num_panels,
    flux_interpolators_by_hour,
    tube_absorbance,
):
    """
    Applies flux boundary conditions to analyzed tubes. Flux value is interpolated based on
    centerline tube location. It is applied along the tube face as a cosine distribution
    with max at sun-facing centerline to zero at 90 deg in either direction.

    Args:
      rec (Receiver): the receiver object that the tubes belong to
      ass_tube_per_panel (int): the assumed number of tubes per panel, or analyzed number of tubes
      num_panels (int): number of panels in the receiver #BPMToDo: remove this and get from rec
      flux_interpolators_by_hour (list(RegularGridInterpolators)): list of flux interpolators
        on receiver by hour of analysis
      tube_z (list(double)): z nodes points along tube
      tube_theta (list(double)): theta nodes points along tube
      tube_absorbance (double): absorbance of tubes
      write_tube_flux_to_csv (bool): flag to dump flux data per tube to a csv file to debug

    Returns: None
    """
    print("Assigning flux BCs to tubes")
    for panel_key, panel in rec.panels.items():
        # in each panel of rec
        for tube_key, tube in panel.tubes.items():
            print(f"Panel: {panel_key} Tube: {tube_key}")
            # in each tube of panel
            tube_rec_theta = 180 - (360 * int(panel_key) / num_panels)
            if ass_tube_per_panel == 1:
                # if only one tube, place at center of panel
                tube_rec_theta += 360 / (2 * num_panels)
            else:
                # if more than one tube, evenly space them including edges
                tube_rec_theta -= (
                    360 / num_panels * int(tube_key) / (ass_tube_per_panel - 1)
                )
            if tube_rec_theta == -180:
                # wrap data around tube
                tube_rec_theta = 180
            apply_tube_flux_bcs(
                tube,
                tube_absorbance,
                tube_rec_theta,
                flux_interpolators_by_hour,
            )


def set_rec_flow_paths(rec, panel_flow_path, mass_flow_per_path, T_in_per_path):
    """
    Set flowpaths within a receiver object

    Args:
      rec (Receiver): the receiver object to set flowpaths for
      panel_flow_path (list[string]): panel names indicating the flowpath. Can have a list
        of lists if more that one flowpath
      mass_flow_per_path (list[double]): list with entries of initial mass flow rate for
        each path
      T_in_per_path (list[double]): list with entries of inlet temp for each path
    """
    print("Setting receiver flowpaths")
    T_in_per_path = convert_C_to_K(T_in_per_path)
    # add flowpaths to receiver
    for i_path, flow_path_panels in enumerate(panel_flow_path):
        # for each path in given list
        print(f"Path {i_path}")
        flow_path = [0] * len(flow_path_panels)
        for i_panel, panel in enumerate(flow_path_panels):
            # for each panel in path
            print(f"Panel: {panel}")
            flow_path[i_panel] = panel
            panel = rec.panels[panel]
            for tube in panel.tubes.values():
                # for each tube in panel
                times = tube.times
        # initialize mass flow for path
        mass_flow = mass_flow_per_path[i_path]
        # convert mass flow from kg/s to kg/hr
        mass_flow *= 3600
        # set inlet temp
        T_in_flow_path = T_in_per_path[i_path] * np.ones_like(times)
        rec.add_flowpath(flow_path, times, mass_flow, T_in_flow_path)


def check_outlet_temps_and_step_mass_flow(
    rec, path, path_key, T_out_target, pct_err_outlet_temp, breaker
):
    """
    Check outlet temp of all tubes from last panel in flowpath.
    Average their value and use it to step mass flow rate.

    Args:
      rec (srlife.Receiver): Receiver object who owns the flowpath we are
        optimizing
      path (OrderedDict): dictionary of flowpath information
      path_key (string): name of flowpath
      T_out_target (double): target outlet temperature
      pct_err_outler_temp (double): acceptable error % for outlet temp (for convergence)
      breaker (list[bool]): check whether all flowpaths have converged
    """
    # All tubes in last panel go to manifold, so estimate
    # temp as avg of all tubes output
    panels = path.get("panels")
    T_in_path = path["inlet_temp"][1:]
    last_panel = rec.panels[panels[-1]]
    num_tubes = last_panel.ntubes
    num_time_steps = next(iter(last_panel.tubes.values())).ntime
    T_out_path = np.zeros([num_tubes, num_time_steps - 1])
    for i_tube, tube in enumerate(last_panel.tubes.values()):
        # for tubes in the last panel of the flow path
        # results for tube are (time, nz) shape
        # want all time except 0 and the last zPos
        T_out_path[i_tube] = tube.axial_results.get("fluid_temperature")[1:, -1]
    T_out_path = T_out_path.mean(axis=0)
    print(f"T_out = {T_out_path - 273.15}")
    if (
        any(T_out_path > T_out_target * (1 + pct_err_outlet_temp / 100)) is False
        and any(T_out_path < T_out_target * (1 - pct_err_outlet_temp / 100)) is False
    ):
        # we have converged this flowpath
        breaker[int(path_key)] = True
    else:
        mass_flow = path.get("mass_flow")
        mass_flow[1:] = (
            mass_flow[1:] * (T_out_path - T_in_path) / (T_out_target - T_in_path)
        )
        mass_flow[0] = mass_flow[1]
        rec.flowpaths[path_key]["mass_flow"] = mass_flow
        print(f"Mass flow: {mass_flow/3600}")


def optimize_mass_flow_rate_per_path(
    rec, rec_filename, T_out_target, pct_err_outlet_temp, solver, save_heat_to_vtu
):
    """
    Iteratively calculate the ideal mass flow rate for each path such that the
    outlet temperature matches a target temperature.

    Args:
      rec (Receiver): srlife Receiver object we are optimizing flow for
      rec_filename (string): file name for the receiver object to optimize
      T_out_target (double): target output temperature
      pct_err_outlet_temp (double): acceptable percentage of error for the outlet temp
      solver (managers.SolutionManager): srlife solver object
      save_heat_to_vtu (bool): whether or not to save vtu files of results
    """
    print("Optimizing mass flow rate to match outlet temp")
    T_out_target = convert_C_to_K(T_out_target)
    # make sure receiver is up to date
    solver.receiver = rec
    # Mass flow optimization
    # limit of error on outlet temp
    max_mass_flow_iter = 5
    for i_opt in range(max_mass_flow_iter):
        print(f"Iteration = {i_opt}")
        breaker = [False] * len(rec.flowpaths)
        if i_opt > 0:
            for path_key, path in rec.flowpaths.items():
                print(f"FlowPath: {path_key}")
                # check last panel tube outlet temps
                check_outlet_temps_and_step_mass_flow(
                    rec, path, path_key, T_out_target, pct_err_outlet_temp, breaker
                )
            if save_heat_to_vtu:
                for panel_key, panel in rec.panels.items():
                    for tube_key, tube in panel.tubes.items():
                        tube.write_vtk(f"ht-tube-{panel_key}-{tube_key}")

        if all(breaker):
            print(f"Converged!!! solved in {i_opt} iterations")
            if save_heat_to_vtu:
                for panel_key, panel in rec.panels.items():
                    for tube_key, tube in panel.tubes.items():
                        tube.write_vtk(f"ht-tube-{panel_key}-{tube_key}")
            break
        # solve heat transfer problem
        solver.solve_heat_transfer()
        filename = rec_filename + "_ht_iter_" + str(i_opt) + ".hdf5"
        rec.save(filename)


def calc_fluid_velocity(tube_mass_flow, fluid_rho, tube_Dh):
    """
    Calculate the velocity of the fluid in a tube based on mass flow rate

    Args:
      tube_mass_flow (double): mass flow rate in tube
      fluid_rho (double): fluid density
      tube_Dh (double): hydraulic diameter of tube
    """
    return tube_mass_flow / fluid_rho * (4.0 / (np.pi * tube_Dh**2))


def calc_reynolds_number(fluid_rho, fluid_vel, fluid_mu, tube_Dh):
    """
    Calculate the Reynolds number for tube flow

    Args:
      fluid_rho (double): fluid density
      fluid_vel (double): fluid velocity
      fluid_mu (double): fluid dynamic viscosity
      tube_Dh (double): tube hydraulic diameter

    """
    return fluid_rho * fluid_vel * tube_Dh / fluid_mu


def calc_fd(Re, tube_eta, tube_Dh):
    """
    Calculate Darcy friction factor from:
    Zigrange and Sylvester 1985, A review of explicit friction factor equations.
    J of Energy Resources Technology)
    Equation 13

    Args:
      Re (double): Reynolds number
      tube_eta (double): tube roughness
      tube_Dh (double): tube hydraulic diameter
    Returns:
      fd (double): darcy friciton factor
    """
    if Re < 4000:
        # laminar flow
        fd = 64 / Re
    else:
        fd = (
            1
            / (
                -2
                * np.log10(
                    (tube_eta / tube_Dh) / 3.7
                    - 5.02 / Re * np.log10((tube_eta / tube_Dh) / 3.7 + 13 / Re)
                )
            )
            ** 2
        )
    return fd


def calc_friction_p_loss(fluid_rho, fluid_mu, tube_mass_flow, tube_dict, delta_l):
    """
    Calculate pressure loss from friction in tubes

    Args:
      fluid_rho (double): fluid density in tubes
      fluid_mu (double): fluid dynamic visco in tubes
      tube_mass_flow (double): mass flow rate in tube
      tube_dict (dict): dictionary of tube specs
      delta_l (double): length of tube

    Returns:
      friction_p_loss (double): pressure loss from friction
    """
    # tube roughness
    tube_eta = tube_dict["eta"]
    tube_Dh = convert_mm_to_m(tube_dict["od"] - 2 * tube_dict["t"])
    fluid_vel = calc_fluid_velocity(tube_mass_flow, fluid_rho, tube_Dh)
    Re = calc_reynolds_number(fluid_rho, fluid_vel, fluid_mu, tube_Dh)
    fd = calc_fd(Re, tube_eta, tube_Dh)
    return fd * (0.5 * fluid_rho) * (fluid_vel**2) / tube_Dh * delta_l


def calc_manifold_bend_loss(
    max_mass_flow, manifold_dict, manifold_rho, manifold_mu, num_bends
):
    """
    Calculate pressure loss from flow through bends in manifold tubes

    Args:
      max_mass_flow (double): mass flow in manifold
      manifold_dict (dict): dictionary with manifold tube specs
      manifold_rho (double): fluid density in manifold
      manifold_mu (double): dynamic viscosity in manifold
      num_bends (int): total number of bends in manifold tubes
    """
    manifold_eta = manifold_dict["eta"]
    manifold_pipe_t = convert_mm_to_m(manifold_dict["t"])
    manifold_pipe_id = convert_mm_to_m(manifold_dict["od"] - 2 * manifold_pipe_t)
    manifold_pipe_bend_radius = convert_mm_to_m(manifold_dict["bend_radius"])
    manifold_eta = manifold_dict["eta"]
    manifold_Dh = manifold_pipe_id
    if manifold_eta / manifold_Dh > 0.0001:
        K0 = 0.42
    elif manifold_eta / manifold_Dh == 0:
        K0 = 0.21
    else:
        K0 = 0.21 * (1 + 1000 * manifold_eta / manifold_Dh)

    manifold_vel = calc_fluid_velocity(max_mass_flow, manifold_rho, manifold_Dh)
    Re = calc_reynolds_number(manifold_rho, manifold_vel, manifold_mu, manifold_Dh)
    manifold_fd = calc_fd(Re, manifold_eta, manifold_Dh)
    K = (
        K0 ** ((manifold_pipe_id / manifold_pipe_bend_radius) ** 0.5)
        + 0.5 * np.pi * (manifold_pipe_bend_radius / manifold_pipe_id) * manifold_fd
    )
    return num_bends * K * manifold_rho * manifold_vel**2


def calc_path_total_p_loss(
    rec,
    path,
    path_key,
    delta_T,
    tube_dict,
    manifold_dict,
    fluid,
    rec_circumference,
    num_panels,
):
    """
    Calculate the total pressure loss in a flowpath from flow friction,
    head loss, and flow through pipe bends.

    Args:
      rec (Receiver): receiver object to analyze
      path (dict): flowpath to get pressure loss through
      path_key (string): name of flowpath
      delta_T (double): temp step through flowpath
      tube_dict (dict): dictionary of usefule tube info
      manifold_dict (dict): dict of useful info for manifold tubes
      fluid (thermalfluid.ThermalFluidMaterial): working fluid in receiver
      rec_circumference (double): circumference of receiver
      num_panels (int): number of panels in receiver

    Returns:
      total_p_loss (double): total pressure loss in flowpath
    """
    g = 9.81  # m/s^2
    # NOTE: this function works mostly with meters
    act_tube_per_panel = tube_dict["ass_tube_per_panel"] * tube_dict["tube_mult"]
    num_bends = num_panels * manifold_dict["bends_per_panel"]

    inlet_temp = path["inlet_temp"][0]
    # get outlet avg outlet temp from each tube in last panel
    # then average down again to single value
    outlet_temp = np.mean(
        [
            tube.axial_results.get("fluid_temperature")[:, -1][1:-1].mean()
            for tube_key, tube in rec.panels[path["panels"][-1]].tubes.items()
        ]
    )
    max_mass_flow = path["mass_flow"].max() / 3600  # kg/s
    tube_mass_flow = max_mass_flow / act_tube_per_panel
    flow_path_length = convert_mm_to_m(tube_dict["h"]) * len(path["panels"])
    temp_steps = np.arange(inlet_temp, outlet_temp, delta_T)
    rho_with_T = fluid.rho(temp_steps) * 1e9  # kg/m^3
    mu_with_T = fluid.mu(temp_steps) * 1000 / 3600  # Pa-s

    # Friction Loss
    # pos spacing of temp change on tubes
    delta_l = flow_path_length / np.size(temp_steps)
    friction_p_loss = sum(
        calc_friction_p_loss(rho, mu, tube_mass_flow, tube_dict, delta_l)
        for rho, mu in zip(rho_with_T, mu_with_T)
    )
    # Head Loss
    avg_temp = 0.5 * (inlet_temp + outlet_temp)
    head_p_loss = fluid.rho(avg_temp) * g * convert_mm_to_m(tube_dict["h"])
    # Manifold Loss
    manifold_friction_p_loss = calc_friction_p_loss(
        fluid.rho(avg_temp) * 1e9,
        fluid.mu(avg_temp) * 1000 / 3600,
        max_mass_flow,
        manifold_dict,
        (len(path["panels"]) - 1) * (rec_circumference / num_panels),
    )
    manifold_bend_loss = calc_manifold_bend_loss(
        max_mass_flow,
        manifold_dict,
        fluid.rho(avg_temp) * 1e9,
        fluid.mu(avg_temp) * 1000 / 3600,
        num_bends,
    )
    return friction_p_loss + head_p_loss + manifold_bend_loss + manifold_friction_p_loss


def calc_p_loss_from_flows_temps(rec, tube_dict, manifold_dict, fluid, outlet_p):
    """
    Calculate the pressure loss across the receiver based on the results
    of a flow simulation. This captures friction loss, head loss, and bend losses
    in the tubes and manifold pipes of the receiver

    Args:
      rec (Receiver): the receiver object to calculate pressure loss
      tube_dict (dict): a dictionary containing info about rec tubes
      manifold_dict (dict): a dictionary containing info about manifold tubes
      fluid (thermalfluid.ThermalFluidMaterial): thermal material of working fluid
      outlet_p (double): pressure at flowpath outlets
    """
    print("Calculating pressure loss from flow at temps")
    # receiver details
    num_panels = len(rec.panels)
    # back out rec_diam
    rec_diam = (
        tube_dict["ass_tube_per_panel"]
        * tube_dict["tube_mult"]
        * num_panels
        * (tube_dict["od"] + tube_dict["spacing"])
        / np.pi
    )
    rec_circumference = np.pi * convert_mm_to_m(rec_diam)
    print(f"Rec diam = {convert_mm_to_m(rec_diam)}")
    # This is number of tempsteps along tube
    delta_T = 5
    flow_path_p_loss = np.zeros(len(rec.flowpaths))
    for path_key, path in rec.flowpaths.items():
        # for each flow path in rec
        print(f"Flow path: {path_key}")
        total_p_loss = calc_path_total_p_loss(
            rec,
            path,
            path_key,
            delta_T,
            tube_dict,
            manifold_dict,
            fluid,
            rec_circumference,
            num_panels,
        )
        flow_path_p_loss[int(path_key)] = total_p_loss
        print(f"Total pressure loss= {(total_p_loss/1e6)} MPa")
        print(f"Inlet pressure= {(outlet_p + total_p_loss/1e6)} MPa")
    return flow_path_p_loss


def update_tube_pressure_bcs(rec, inlet_p_per_path, outlet_p):
    """
    Set tube pressure BCs based on pressure loss calc after thm solve

    Args:
      rec (Receiver): receiver object to set the pressure for
      inlet_p_per_path (np.array): array of inlet pressures for each flowpath
      outlet_p (double): outlet pressure, same for all paths
    """
    # set tube pressures and temperatures
    for path_key, path in rec.flowpaths.items():
        # for each flowpath in the model
        tube_pressures = np.linspace(
            inlet_p_per_path[int(path_key)], outlet_p, len(path["panels"]) + 1
        )[:-1]
        for i_panel, panel_key in enumerate(path["panels"]):
            # for each panel in flowpath
            for tube in rec.panels[panel_key].tubes.values():
                # for each tube in panel
                # get tube times
                times = tube.times
                pressure = np.ones_like(times)
                pressure[0] = 0.0
                tube_pressure = tube_pressures[i_panel] * pressure
                tube_pressure_bc = receiver.PressureBC(times, tube_pressure)
                tube.set_pressure_bc(tube_pressure_bc)


def cycle_tube_pressure_bcs(tubes_dict, num_cycles, cyclic_times):
    """
    Set pressure bcs on internal face of tube from flow

    Args:
      tubes_dict (dict): rec.panels[panel].tubes dictionary tubes from a panel
      tube_pressure (list[double]): pressure values for tubes in a given panel with height
      pressure (list[double]): properly sized array of pressure data to define bc
      times (list[int]): list of analysis times

    """
    for tube in tubes_dict.values():
        # NOTE: as soon as I cycle results
        # ghost temp, fluid temp, fluid velocity results are no longer good
        # so lets delete themo
        try:
            tube.quadrature_results.pop("ghost_temperature")
            tube.axial_results.pop("fluid_temperature")
            tube.axial_results.pop("fluid_velocity")
        except KeyError:
            # We do not need to do anything here, results are already gone
            pass
        # for each tube in panel
        press_bc = tube.pressure_bc
        pressure = press_bc.data
        tube_pressure = np.tile(pressure[1:], num_cycles)
        tube_pressure = np.append(0, tube_pressure)
        tube_pressure_bc = receiver.PressureBC(cyclic_times, tube_pressure)
        tube.set_pressure_bc(tube_pressure_bc)


def set_and_downsample_tube_temp_bcs(
    tubes_dict,
    set_init_T_to_inlet_T,
    inlet_T,
    cyclic_times,
    num_cycles,
    analysis_type,
    loc,
):
    """
    Handle specific details for temperature BCs in a tube.
    Set the initial temperature
    Find and set the appropriate temp for 2d analysis

    Args:
      tubes_dict (dict): rec.panels[panel].tubes dictionary tubes from a panel
      set_init_T_to_inlet_T (bool): whether to initialize tube temps as the
        inlet temperature
      inlet_T (double): inlet temperature for tubes
      cyclic_times (list[int]): list of times across all cycles of analysis
      num_cycles (int): number of cycles in analysis
      analysis_type (string): "2d" or "3d" to handle turn on or off downsampling
        of temperature results for simpler analysis
      loc (string): "max_T" or "max_avg_T" to set where temp is sampled in 2d analysis
    """
    for tube in tubes_dict.values():
        # for each tube in panel
        T = tube.results["temperature"]
        if set_init_T_to_inlet_T:
            T[0] = inlet_T
            tube.T0 = inlet_T
        _, _, _, r4 = np.shape(T)
        # this will tile stack all temp results num_cycles times
        # It does not account for cycle heuristic
        T_0 = np.array([T[0]])
        T = np.tile(T[1:], (num_cycles, 1, 1, 1))
        T = np.append(T_0, T, axis=0)
        tube.results["temperature"] = T
        tube.set_times(cyclic_times)
        T_3d = tube.results["temperature"]
        if analysis_type == "2d":
            if loc == "max_T":
                T_max = 0
                T_max_h_index = -1
                for i_temp in range(1, r4 - 1):
                    if np.max(T_3d[:, :, :, i_temp]) > T_max:
                        T_max = np.max(T_3d[:, :, :, i_temp])
                        T_max_h_index = i_temp
                tube.make_2D(tube.h / (r4 - 1) * T_max_h_index)
            elif loc == "max_avg_T":
                T_3d_avg = np.average(T_3d, axis=0)
                T_max = 0
                T_max_h_index = -1
                for i_temp in range(1, r4 - 1):
                    if np.max(T_3d_avg[:, :, i_temp]) > T_max:
                        T_max = np.max(T_3d_avg[:, :, i_temp])
                        T_max_h_index = i_temp
                tube.make_2D(tube.h / (r4 - 1) * T_max_h_index)
            tube.results["temperature"] = T_3d[:, :, :, T_max_h_index]


def process_single_panel_analysis(rec, struct_output_dict, single_panel_analysis_id):
    """
    Remove all panels that will not be analyzed

    Args:
      rec (Receiver): object we are analyzing
      struct_output_dict (dict): dictionary with output information
        save_struct_to_vtu (bool): whether or not to write files to vtu
        st_fname (string): filename for structure
        tube_fname (string): filename for tube
      single_panel_analysis_is (string): name of panel to analyze
    """
    # file name mods
    st_fname = (
        f"sing_panel_{single_panel_analysis_id}_{struct_output_dict['st_filename']}"
    )
    struct_output_dict["st_filename"] = st_fname
    tube_fname = (
        f"sing_panel_{single_panel_analysis_id}_{struct_output_dict['tube_filename']}"
    )
    struct_output_dict["tube_filename"] = tube_fname
    # update model to use just one panel
    single_panel_model = rec.panels[single_panel_analysis_id]
    rec.panels.clear()
    rec.flowpaths.clear()
    rec.add_panel(single_panel_model)


def save_structural_results(rec_struct, st_fname, tube_fname):
    """
    Save results from analysis to files

    Args:
      rec_struct (Receiver): receiver object we analyzed
      st_fname (string): hdf5 filename for structure
      tube_fname (string): base filename for tube vtus
    """
    # post process
    rec_struct.save(st_fname + ".hdf5")
    for panel_key, panel in rec_struct.panels.items():
        for tube_key, tube in panel.tubes.items():
            tube.write_vtk(f"{tube_fname}_{panel_key}_{tube_key}")


def run_struct_analysis(
    rec_filename,
    set_init_T_to_inlet_T,
    loc,
    analysis_type,
    is_single_panel_analysis,
    single_panel_analysis_id,
    num_cycles,
    solver,
    struct_output_dict,
):
    """
    Runs the structural analysis on the receiver with a given set of thermal
    hydraulics results.

    Args:
      rec_filename (string): filename of the receiver to analyze
      set_init_T_to_inlet_T (bool): whether to initialize tube temps as the
        inlet temperature
      loc (string): "max_T" or "max_avg_T" used to select location to downsample
        analysis domain if lower dimensional analysis is done
      analysis_type (string): "2d" or "3d" used to decide whether full analysis
        or downsampled analysis is used
      is_single_panel_analysis (bool): whether a single panel is analyzed or the
        full receiver
      single_panel_analysis_id (string): name of single panel to analyze if
        is_single_panel_analysis == True
      num_cycles (double): number of times to repeat load cycles for the analysis
      solver (managers.SolutionManager): system solver for the receiver
      struct_output_dict (dict): dictionary with info about output for st files
        save_to_vtu (bool): save files to vtu output
        st_filename (string): hdf5 filename for structural output
        tube_filename (string): filename for vtk output of tube analysis

    ToDos:
      rename loc to better name, make it and analysis_type bools?
    """
    # passing filename here because we may change receiver significantly
    # i.e. removing many panels
    rec_struct = receiver.Receiver.load(rec_filename + ".hdf5")
    rec_struct.days *= num_cycles
    inlet_T = rec_struct.flowpaths["0"]["inlet_temp"][0]

    # set tube pressures and temperatures
    for path in rec_struct.flowpaths.values():
        # for each flowpath in the model
        for panel_key in path["panels"]:
            # for each panel in flowpath
            # times we actually analyze receiver for thm
            first_tube = next(iter(rec_struct.panels[panel_key].tubes.values()))
            analysis_times = first_tube.times[1:]
            # create cyclic time for life
            cyclic_times = np.tile(analysis_times, num_cycles)
            for i_cyc in range(num_cycles):
                cyclic_times[
                    i_cyc * len(analysis_times) : (i_cyc + 1) * len(analysis_times)
                ] += (i_cyc * rec_struct.period)
            # initial steps for pressure and time
            cyclic_times = np.append([0], cyclic_times)
            pressure = np.ones_like(cyclic_times)
            pressure[0] = 0.0

            cycle_tube_pressure_bcs(
                rec_struct.panels[panel_key].tubes,
                num_cycles,
                cyclic_times,
            )
            set_and_downsample_tube_temp_bcs(
                rec_struct.panels[panel_key].tubes,
                set_init_T_to_inlet_T,
                inlet_T,
                cyclic_times,
                num_cycles,
                analysis_type,
                loc,
            )
    # remove unnecessary panels if needed
    if is_single_panel_analysis:
        process_single_panel_analysis(
            rec_struct, struct_output_dict, single_panel_analysis_id
        )
    solver.receiver = rec_struct
    # run structural problem
    solver.solve_structural()
    if struct_output_dict["save_to_vtu"]:
        save_structural_results(
            rec_struct,
            struct_output_dict["st_filename"],
            struct_output_dict["tube_filename"],
        )
    rec_struct.save(rec_filename + "_struct.hdf5")


def create_receiver(tube_dict, num_days, times, period, panel_k, num_panels, results):
    """
    Create an srlife.Receiver object based on user defined inputs

    Args:
      tube_dict (dict): Python dictionary that holds useful tube definitions
        "od", "t", "h", "nr", "nt", "nz", "spacing", "T0", "tube_k",
        "eta", "tube_mult", "ass_tube_per_panel"
      num_days (int): number of days to analyze
      times (list[int]): list of index of hours to analyze
      period (int): number of hours to analyze in a day
      panel_k (double): panel stiffness used in SystemSolver object
      num_panels (int): number of panels
      results (list): set of results for tube objects
    """
    # array of 1's with leading and trailing zeros multiplied by input pressure
    rec = receiver.Receiver(period, num_days, panel_k)
    for _ in range(num_panels):
        rec.add_panel(create_panel(tube_dict, times, results))
    return rec


def create_panel(tube_dict, times, results):
    """
    Create an srlife.Panel object to add to a receiver

    Args:
      tube_dict (dictionary): Python dictionary that holds useful tube definitions
      times (list[int]): list of index of hours to analyze
      results (list): list of results for tube objects

    Returns:
      panel (srlife.Panel): panel object
    """
    panel = receiver.Panel(tube_dict["tube_k"])
    for _ in range(tube_dict["ass_tube_per_panel"]):
        panel.add_tube(create_tube(tube_dict, times, results))
    return panel


def create_tube(tube_dict, times, results):
    """
    Create an srlife.Tube object to add to panel

    Args:
      tube_dict (dictionary): Python dictionary that holds useful tube definitions
      times (list[int]): list of index of hours to analyze
      results (list): list of results for tube object

    Returns:
      tube (srlife.Tube): tube object
    """
    tube = receiver.Tube(
        0.5 * tube_dict["od"],
        tube_dict["t"],
        tube_dict["h"],
        tube_dict["nr"],
        tube_dict["nt"],
        tube_dict["nz"],
        tube_dict["T0"],
    )
    tube.set_times(times)
    tube.multiplier_val = tube_dict["tube_mult"]
    tube.T0 = tube_dict["T0"]
    # initialize pressure data
    pressure_data = np.ones(len(times)) * 1.0
    pressure_data[0] = 0.0
    pressure_data[-1] = 0.0
    pressure = receiver.PressureBC(times, pressure_data)
    tube.set_pressure_bc(pressure)
    for res in results:
        tube.add_results(
            res,
            np.zeros((len(times), tube_dict["nr"], tube_dict["nt"], tube_dict["nz"])),
        )
    return tube


def sample_parameters(num_threads, verbose, rtol, atol):
    """
    Create a set of srlife.SolverParams to be used by srlife solvers

    Args:
      num_threads (int): number of threads for analysis
      verbose (bool): whether solvers should print verbose info
      rtol (double): relative tol for solver convergence
      atol (double): abs tol for solver convergence
    Returns: None
    """
    params = solverparams.ParameterSet()

    params["nthreads"] = num_threads
    params["progress_bars"] = True
    # If true store results on disk (slower, but less memory)
    params["page_results"] = False

    params["thermal"]["miter"] = 200
    params["thermal"]["verbose"] = verbose
    params["thermal"]["steady"] = True
    params["thermal"]["substep"] = 2  # 10

    params["thermal"]["solid"]["rtol"] = rtol
    params["thermal"]["solid"]["atol"] = atol
    params["thermal"]["solid"]["miter"] = 200
    params["thermal"]["solid"]["verbose"] = verbose
    params["thermal"]["solid"]["substep"] = 2  # 10

    params["thermal"]["fluid"]["rtol"] = rtol
    params["thermal"]["fluid"]["atol"] = atol
    params["thermal"]["fluid"]["miter"] = 200
    params["thermal"]["fluid"]["verbose"] = verbose
    params["thermal"]["fluid"]["substep"] = 2  # 100

    params["structural"]["rtol"] = rtol
    params["structural"]["atol"] = atol
    params["structural"]["miter"] = 50
    params["structural"]["verbose"] = verbose

    params["system"]["rtol"] = rtol
    params["system"]["atol"] = atol
    params["system"]["miter"] = 10
    params["system"]["verbose"] = verbose

    # If true store results on disk (slower, but less memory)
    params["page_results"] = False

    return params
