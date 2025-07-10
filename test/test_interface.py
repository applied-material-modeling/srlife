import unittest
import tempfile

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import multiprocess
import subprocess
import h5py

from srlife import (
    receiver,
    thermal,
    structural,
    system,
    damage,
    library,
    managers,
    interface,
)

header = b"\n\n\n\n\n\n"

fake_data = b"""-,180,135,90,45,0,-45,-90,-135,
0,1,1,1,1,1,1,1,1
1,1,1,1,1,1,1,1,1
2,1,1,1,1,1,1,1,1
3,1,1,1,1,1,1,1,1
"""

nr_unit = 2
nt_unit = 4
nz_unit = 4
tube_dict_unit = {
    "od": 21.3,
    "t": 1.25,
    "h": 3,
    "nr": nr_unit,
    "nt": nt_unit,
    "nz": nz_unit,
    "spacing": 1,
    "T0": 300,
    "tube_k": "rigid",
    "eta": 0.1,
    "tube_mult": 58.5,
    "ass_tube_per_panel": 2,
}


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


class InterfaceUnitTests(unittest.TestCase):
    """
    Run unit tests on some funcitons of interface.py
    """

    def test_file_not_found(self):
        flux_filename = "no_file_existing.csv"
        r = 10
        c = 17
        data_shape = [r, c]
        z_offset_in = 0.0
        z_data, theta_data, flux_data = interface.read_month_day_hour_flux_file(
            flux_filename, data_shape, z_offset_in
        )
        print(theta_data)
        self.assertTrue(np.array_equal(z_data, np.zeros([r, 1])))
        self.assertTrue(np.array_equal(theta_data, np.zeros([1, c])))
        self.assertTrue(np.array_equal(flux_data, np.zeros([r, c])))

    def test_fake_file(self):
        # Create a mock CSV file content to test the parsing
        with tempfile.NamedTemporaryFile(delete=True) as f:
            f.write(header)
            f.write(fake_data)
            f.seek(0)
            z_data, theta_data, flux_data = interface.read_month_day_hour_flux_file(
                f.name, [4, 8], 0
            )
        # spot check data
        print(z_data)
        self.assertEqual(z_data[2], 2000)
        self.assertEqual(theta_data[-1], -180)
        self.assertEqual(flux_data[2, 5], 1)

    def test_apply_tube_flux_bcs(self):
        # test applying BCs via interpolator
        times = np.array([0, 1])
        tube = interface.create_tube(tube_dict_unit, times, [])
        tube_absorbance = 1.0
        tube_rec_theta = 12
        z_pts = np.linspace(10, 20, nz_unit)
        theta_pts = np.linspace(15, -15, nt_unit)
        data = np.array([[(2 * ix + 4 * iy) for ix in z_pts] for iy in theta_pts])
        interpFun = RegularGridInterpolator(
            (z_pts, theta_pts),
            data,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        tube_rec_theta = 4
        interface.apply_tube_flux_bcs(
            tube, tube_absorbance, tube_rec_theta, [interpFun]
        )
        # spot check tube BCs
        self.assertAlmostEqual(tube.outer_bc.data[0, 0, 2], interpFun([0.0, 3.0]))
        self.assertEqual(tube.outer_bc.data[0, 3, 3], 0.0)

    def test_calc_fluid_velocity(self):
        mfr = 10
        rho = 5
        Dh = 3
        vel = mfr / rho * (4.0 / (np.pi * Dh**2))
        self.assertAlmostEqual(interface.calc_fluid_velocity(mfr, rho, Dh), vel)

    def test_calc_reynolds_number(self):
        mfr = 28
        rho = 5
        mu = 5
        Dh = 3
        vel = mfr / rho * (4.0 / (np.pi * Dh**2))
        Re = rho * vel * Dh / mu
        self.assertAlmostEqual(interface.calc_reynolds_number(rho, vel, mu, Dh), Re)

    def test_calc_fd(self):
        Re = [2000, 10000]
        eta = 0.2
        Dh = 0.012
        for iRe in Re:
            fd = interface.calc_fd(iRe, eta, Dh)
            fd_test = calc_fd(iRe, eta, Dh)
            self.assertAlmostEqual(fd, fd_test, places=12)

    def test_calc_friction_p_loss(self):
        mfr = 0.001
        rho = 50
        mu = 0.004
        delta_l = 1
        p_loss = interface.calc_friction_p_loss(rho, mu, mfr, tube_dict_unit, delta_l)
        test_val = 26.09267992590423
        self.assertAlmostEqual(p_loss, test_val)

    def test_calc_manifold_bend_loss(self):
        mfr = 0.001
        rho = 1550
        mu = 0.004
        manifold_tube_dict = {
            "od": 21.3,
            "t": 1.25,
            "bend_radius": 18,
            "bends_per_panel": 4,
            "eta": 0.01,
        }
        num_bends = 4

        p_loss = interface.calc_manifold_bend_loss(
            mfr, manifold_tube_dict, rho, mu, num_bends
        )
        test_val = 0.12351409567895541
        self.assertAlmostEqual(p_loss, test_val)

    def test_create_receiver(self):
        num_days = 1
        times = np.array([0, 1, 2])
        period = 2
        panel_k = "disconnect"
        num_panels = 12
        results = []
        rec = interface.create_receiver(
            tube_dict_unit, num_days, times, period, panel_k, num_panels, results
        )
        self.assertEqual(num_panels, rec.npanels)
        for panel in rec.panels.values():
            self.assertEqual(tube_dict_unit["ass_tube_per_panel"], panel.ntubes)

    def test_cycle_tube_pressure_bcs(self):
        times = np.array([0, 1])
        tube_pressure = 12
        pressure = np.ones_like(times)*tube_pressure
        pressure[0] = 0.0
        tube = interface.create_tube(tube_dict_unit, times, [])
        tube_pressure_bc = receiver.PressureBC(times, pressure)
        tube.set_pressure_bc(tube_pressure_bc)

        tubes_dict = {"0": tube}
        num_cycles = 3
        cyclic_times = np.array([0, 1, 2, 3])
        interface.cycle_tube_pressure_bcs(
            tubes_dict, num_cycles, cyclic_times
        )
        gold = np.array([0,12,12,12])
        self.assertTrue(np.array_equal(gold, tube.pressure_bc.data))

    def test_set_and_downsample_tube_temp_bcs(self):
        times = np.array([0, 1, 2])
        cyclic_times = np.array([0, 1, 2])
        num_cycles = 1
        tube = interface.create_tube(tube_dict_unit, times, [])
        tubes_dict = {"0": tube}
        data = 245 * np.ones([tube.ntime, tube.nr, tube.nt, tube.nz])
        tube.add_results("temperature", data)
        # Test 1: inlet T
        interface.set_and_downsample_tube_temp_bcs(
            tubes_dict,
            True,
            245,
            cyclic_times,
            num_cycles,
            "3d",
            "",
        )
        self.assertEqual(245, tube.T0)
        analysis_type = "2d"
        locs = ["max_T", "max_avg_T"]
        for i_loc, loc in enumerate(locs):
            tube = interface.create_tube(tube_dict_unit, times, [])
            tubes_dict = {"0": tube}
            data = (i_loc + 245) * np.ones([tube.ntime, tube.nr, tube.nt, tube.nz])
            tube.add_results("temperature", data)
            interface.set_and_downsample_tube_temp_bcs(
                tubes_dict,
                False,
                245,
                cyclic_times,
                num_cycles,
                analysis_type,
                loc,
            )
            self.assertTrue(
                np.array_equal(tube.results["temperature"], data[:, :, :, 0])
            )


class InterfaceRegressionTest(unittest.TestCase):
    def test_simple_model(self):
        multiprocess.set_start_method("spawn", force=True)
        # Receiver filename to write to
        rec_filename = "./test/Rec_Test"
        # file extension used by srlife
        hdf5_ext = ".hdf5"

        # GEOMETRY
        # rec geom
        num_panels = 2  # number of panels in receiver
        rec_diam = 10000  # mm
        rec_height = 12000  # mm

        # rec tube geom
        tube_od = 21.3  # mm (Custom, Schedule 5)
        tube_t = 1.24  # mm (Custom, Schedule 5)
        tube_h = 12000  # mm (also rec height)
        tube_spacing = 1  # mm
        tube_eta = 0.01e-3  # tube roughness
        tube_absorbance = 0.98  # absorbance of tube
        z_offset = 0.25  # z offset for flux data

        # manifold tube geom
        manifold_tube_od = 502.15  # mm
        manifold_tube_t = 45.24  # mm
        manifold_bend_radius = 2 * manifold_tube_od  # used in p loss calc
        manifold_tube_eta = tube_eta  # roughness
        manifold_bends_per_panel = 4
        # END GEOMETRY

        # ASSUMPTIONS
        ass_tube_per_panel = 2  # number of tubes analyzed at each panel
        outlet_p = 0.5  # MPa

        # tube assumption work
        act_tube_per_panel = int(
            np.pi * rec_diam / (num_panels * (tube_od + tube_spacing))
        )
        tube_multiplier = act_tube_per_panel / ass_tube_per_panel
        print(f"Actual tubes per panel = {act_tube_per_panel}")
        print(f"Tube Multiplier = {tube_multiplier}")

        # Structure assumptions
        panel_k = "disconnect"  # spring stiffness panel connect
        tube_k = "rigid"  # stiffness of tube connect
        set_init_T_to_inlet_T = True
        loc = "max_avg_T"  # "max_T"
        analysis_type = "2d"  # "3d"
        is_single_panel_analysis = True
        single_panel_analysis_id = "0"
        # END ASSUMPTIONS

        # BC/IC INFO
        # flux data input and shape
        flux_data_dir = "./test"
        flux_data_shape = [25, 25]
        tube_initial_temp = 300.0  # K
        # END BC/IC INFO

        # ANALYSIS TIMES and CYCLE INFO
        num_days = 1
        start_time = 6
        end_time = 7
        time_step = 1  # hour, can do nonInt
        # stimes is index of hours to analyze, starting with first hour of flux BCs
        stimes = np.arange(1, (end_time + 1) - start_time + time_step, time_step)
        period = len(stimes)
        # array of sample times of across number of days
        times = np.tile(stimes, num_days)
        # add time zero for initial conditions
        times = np.append(np.array([0]), times)
        # month and date from flux data
        month = 6
        day = 20
        print("Times analyzed: \n", times)
        num_cycles = 2
        # END ANALYSIS TIMES AND CYCLE INFO

        # TUBE DISC INFO
        nz = int(interface.convert_mm_to_m(rec_height) * 0.5 + 1)
        nt = 8
        nr = 2
        # END TUBE DISC INFO

        # SOLVERS #
        # setup solver parameters
        num_threads = 1
        # NOTE: using large tolerances to assure quick convergence
        rtol = 1.0e-2
        atol = 1.0e-3
        params = interface.sample_parameters(num_threads, False, rtol, atol)
        # thermal
        thermal_solver = thermal.ThermohydraulicsThermalSolver(params["thermal"])
        # Structural solver
        structural_solver = structural.PythonTubeSolver(params["structural"])
        # receiver system solver
        system_solver = system.SpringSystemSolver(params["system"])
        # material damage model
        damage_model = damage.TimeFractionInteractionDamage(params["damage"])
        # END SOLVERS #

        # MATERIALS
        # fluid model
        mat_fluid = library.load_thermal_fluid("32MgCl2-68KCl", "base")
        # material models
        st_mat_model = "elastic_creep"  # "elastic_model" "base"
        mat_thermal, mat_deformation, mat_damage = library.load_material(
            "740H", "base", st_mat_model, "base"
        )
        # END MATERIALS

        # structure output
        save_struct_files_to_vtu = False
        st_filename = "structure_" + st_mat_model
        tube_filename = "tube_" + st_mat_model

        # FLOW SYSTEM SETUP
        T_out_target = 720  # target outlet temperature, unit: C
        # tolerance for mass flow optimization
        # NOTE: making very large for test for single iter convergence
        pct_err_outlet_temp = 100.25
        panel_flow_path = [["1"], ["0"]]
        mass_flow_per_path = np.ones((len(panel_flow_path),len(times)))*650  # kg/s
        T_in_per_path = np.array([500, 500])  # Celcius
        use_cycle_reset_heuristic = False
        save_heat_to_vtu = False
        # END FLOW SYSTEM SETUP

        # START ANALYSIS WORK
        # rec tube data structure
        tube_dict = {
            "od": tube_od,
            "t": tube_t,
            "h": tube_h,
            "nr": nr,
            "nt": nt,
            "nz": nz,
            "spacing": tube_spacing,
            "T0": tube_initial_temp,
            "tube_k": tube_k,
            "eta": tube_eta,
            "tube_mult": tube_multiplier,
            "ass_tube_per_panel": ass_tube_per_panel,
        }
        # manifold tube data structure
        manifold_tube_dict = {
            "od": manifold_tube_od,
            "t": manifold_tube_t,
            "bend_radius": manifold_bend_radius,
            "bends_per_panel": manifold_bends_per_panel,
            "eta": manifold_tube_eta,
        }

        # STEP 1: create a receiver
        rec = interface.create_receiver(
            tube_dict, num_days, times, period, panel_k, num_panels, results=[]
        )

        # STEP 2: read and assign flux BCs to receiver tubes
        # read flux files, make interpolator functions by hour and save
        # NOTE: not doing multiple days right now
        # NOTE: can we assume zData and thetaData are constant in time???
        flux_interpolators_by_hour = interface.get_flux_interpolators_from_data_files(
            times,
            start_time,
            end_time,
            month,
            day,
            flux_data_dir,
            flux_data_shape,
            z_offset,
        )
        # Take flux interpolators and write flux BCs for each tube
        # flux is sampled at tube centerline and applied smoothly
        # across sunSideSurface via cosine
        interface.calc_and_write_tube_flux_bcs(
            rec,
            ass_tube_per_panel,
            num_panels,
            flux_interpolators_by_hour,
            tube_absorbance,
        )

        rec.save(rec_filename + hdf5_ext)

        # STEP 3: add flow paths to receiver
        # add flow paths to receiver
        interface.set_rec_flow_paths(
            rec, panel_flow_path, mass_flow_per_path, T_in_per_path
        )
        rec.save(rec_filename + hdf5_ext)

        # make solver
        solver = managers.SolutionManager(
            rec,
            thermal_solver,
            mat_thermal,
            mat_fluid,
            structural_solver,
            mat_deformation,
            mat_damage,
            system_solver,
            damage_model,
            pset=params,
        )

        # STEP 4: optimize mass flow rate to match target outlet temp
        # now optimize mass flow rate for each path
        # this takes a long time, can skip if it has already been done
        run_mass_flow_opt = True
        # Heuristics
        if use_cycle_reset_heuristic:
            solver.add_heuristic(managers.CycleResetHeuristic())
        if run_mass_flow_opt:
            interface.optimize_mass_flow_rate_per_path(
                rec,
                rec_filename,
                T_out_target,
                pct_err_outlet_temp,
                solver,
                save_heat_to_vtu,
            )
            # Saving optimized results to base filename
            rec.save(rec_filename + hdf5_ext)
        else:
            rec_filename = "./Rec_Test_ht_iter_2"
            rec = receiver.Receiver.load(rec_filename + hdf5_ext)

        mass_flow = np.array([path["mass_flow"] for path in rec.flowpaths.values()])
        print(f"Mass flow: {mass_flow/3600}")

        # STEP 5: calc pressure loss from flow
        # calculate pressure loss from
        # pipe friction
        # head loss
        # manifold loss
        flow_path_p_loss = interface.calc_p_loss_from_flows_temps(
            rec, tube_dict, manifold_tube_dict, mat_fluid, outlet_p
        )
        inlet_p = outlet_p + interface.convert_Pa_to_MPa(flow_path_p_loss)
        interface.update_tube_pressure_bcs(rec, inlet_p, outlet_p)
        rec.save(rec_filename+hdf5_ext)

        # STEP 6: solve structure and life receiver
        struct_output_dict = {
            "save_to_vtu": save_struct_files_to_vtu,
            "st_filename": st_filename,
            "tube_filename": tube_filename,
        }
        interface.run_struct_analysis(
            rec_filename,
            set_init_T_to_inlet_T,
            loc,
            analysis_type,
            is_single_panel_analysis,
            single_panel_analysis_id,
            num_cycles,
            solver,
            struct_output_dict,
        )
        # END ANALYSIS WORK
        # do some checks on hdf5 files
        file_gold = h5py.File("./test/Rec_Gold.hdf5", "r+")
        file_test = h5py.File(rec_filename + hdf5_ext, "r+")
        # flux bcs for a panel/tube
        data_name = "/panels/1/tubes/0/outer_bc/data"
        print("Flux BC")
        self.assertTrue(
            np.allclose(
                file_test[data_name], file_gold[data_name], rtol=rtol, atol=atol
            )
        )
        # fluid temps for two tubes
        print("Fluid Temp")
        data_name = "/panels/1/tubes/1/axial_results/fluid_temperature"
        self.assertTrue(
            np.allclose(
                file_test[data_name], file_gold[data_name], rtol=rtol, atol=atol
            )
        )
        # temps for a tube
        print("Temp")
        data_name = "/panels/1/tubes/0/results/temperature"
        self.assertTrue(
            np.allclose(
                file_test[data_name], file_gold[data_name], rtol=rtol, atol=atol
            )
        )
        # NOTE: in interface.run_struct_analysis we make this a
        # single panel model, and don;t save it in this run
        # so structure results are not available
        # all else matches and run_struct_analysis finished,
        # so call it cleared
