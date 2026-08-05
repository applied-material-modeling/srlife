"""
A small 2 panel receiver test for srlife-MOOSE interface
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from srlife import receiver
from srlife import damage, interface, library, moose_interface, moose_runner

TEST_DIR = Path(__file__).resolve().parent
GOLD_FILE = TEST_DIR / "moose_interface_gold.json"

REGRESSION_RTOL = 5.0e-2
REGRESSION_ATOL = 1.0e-6

MOOSE_THM_FILENAME = "srlife_moose_test"


def _moose_available():
    paths = (
        os.environ.get("MOOSE_THM"),
        os.environ.get("NEMLAPP"),
        os.environ.get("MOOSE_MPI"),
    )
    return all(p and os.path.isfile(p) and os.access(p, os.X_OK) for p in paths)


def build_small_receiver():
    """Two panels, two analysis tubes/panel, two flow paths: the minimum that
    exercises the multi-panel structural + reliability machinery."""
    num_panels = 2
    rec_diam = 10000.0  # mm
    rec_height = 12000.0  # mm

    tube_od = 42.2  # mm
    tube_t = 6.0  # mm
    tube_spacing = 1.0  # mm
    tube_eta = 0.01e-3  # roughness, mm

    ass_tube_per_panel = 2
    act_tube_per_panel = int(np.pi * rec_diam / (num_panels * (tube_od + tube_spacing)))
    tube_multiplier = act_tube_per_panel / ass_tube_per_panel

    flux_data_shape = [25, 25]
    start_time, end_time, time_step = 6, 7, 1
    stimes = np.arange(1, (end_time + 1) - start_time + time_step, time_step)
    period = len(stimes)
    times = np.append([0], stimes)
    month, day = 6, 20

    tube_dict = {
        "od": tube_od,
        "t": tube_t,
        "h": rec_height,
        "nr": 3,
        "nt": 8,
        "nz": 5,
        "spacing": tube_spacing,
        "T0": 500.0,
        "tube_k": "rigid",
        "eta": tube_eta,
        "tube_mult": tube_multiplier,
        "ass_tube_per_panel": ass_tube_per_panel,
    }

    rec = interface.create_receiver(
        tube_dict, 1, times, period, "disconnect", num_panels, results=[]
    )

    flux_interpolators_by_hour = interface.get_flux_interpolators_from_data_files(
        times, start_time, end_time, month, day, str(TEST_DIR), flux_data_shape, 0.25
    )
    interface.calc_and_write_tube_flux_bcs(
        rec, ass_tube_per_panel, num_panels, flux_interpolators_by_hour, 0.98
    )

    panel_flow_path = [["0"], ["1"]]
    mass_flow_per_path = np.ones((len(panel_flow_path), len(times))) * 4.0  # kg/s
    T_in_per_path = np.array([550.0, 550.0])  # C
    interface.set_rec_flow_paths(
        rec, panel_flow_path, mass_flow_per_path, T_in_per_path
    )

    rec_data = {
        "tube_multiplier": tube_multiplier,
        "rec_diam": rec_diam,
        "tube_od": tube_od,
        "tube_spacing": tube_spacing,
        "tube_eta_m": interface.convert_mm_to_m(tube_eta),
        "period": period,
        "time_step": time_step,
    }
    return rec, rec_data


class MooseInterfaceTest(unittest.TestCase):
    """
    Test MOOSE THM + solid mechanics + srlife reliability.
    """

    @classmethod
    def setUpClass(cls):
        if not _moose_available():
            raise unittest.SkipTest(
                "MOOSE executables not available; Provide path to MOOSE_THM, NEMLAPP and MOOSE_MPI."
            )
        os.environ["MOOSE_NPROCS"] = "2"  # mpi procs, this is enough for the test mesh
        cls.addClassCleanup(os.chdir, os.getcwd())
        os.chdir(tempfile.mkdtemp())

    def test_full_moose_pipeline(self):
        """Run the moose solve and compare"""
        rec, rec_data = build_small_receiver()
        manifold_tube = receiver.Tube(0.5 * 502.15, 45.24, -1.0, 1, 1, 4)

        moose_thm_filenames = rec.create_moose_thm_model(
            moose_filename=MOOSE_THM_FILENAME,
            rec_diam=rec_data["rec_diam"],
            tube_od=rec_data["tube_od"],
            tube_spacing=rec_data["tube_spacing"],
            tube_roughness=rec_data["tube_eta_m"],
            manifold_tube=manifold_tube,
            outlet_p=interface.convert_MPa_to_Pa(20.0),
            target_outlet_T=interface.convert_C_to_K(700.0),
            start_time=0,
            end_time=rec_data["period"] * 3600 * rec_data["time_step"],
            dt=1,
            nl_rel_tol=1e-5,
            nl_abs_tol=1e-3,
        )

        for fname in moose_thm_filenames:
            moose_runner.run_moose(fname, "MOOSE_THM")

        moose_sm_input_files, moose_sm_output_files = (
            moose_interface.create_moose_sm_inputs(MOOSE_THM_FILENAME)
        )
        for sm_in in moose_sm_input_files:
            moose_runner.run_moose(str(sm_in), "NEMLAPP")

        params = interface.sample_parameters(1, False, 1e-4, 1e-6)
        damage_model = damage.PIAModel(params["damage"])
        _, _, mat_damage = library.load_material("SiC", "base", "cares", "tested_3P")

        results = moose_interface.compute_moose_reliability(
            rec,
            mat_damage,
            damage_model,
            lifetime=1.0,
            moose_sm_output_files=moose_sm_output_files,
            tube_multiplier=rec_data["tube_multiplier"],
        )
        self.check_against_gold(results)

    def check_against_gold(self, results):
        """Funtion to compare against gold results"""
        with open(GOLD_FILE, encoding="utf-8") as f:
            gold = json.load(f)
        for key in gold:
            got = np.atleast_1d(np.asarray(results[key], dtype=float))
            ref = np.atleast_1d(np.asarray(gold[key], dtype=float))
            self.assertEqual(got.shape, ref.shape, f"gold shape mismatch for {key}")
            self.assertTrue(
                np.allclose(got, ref, rtol=REGRESSION_RTOL, atol=REGRESSION_ATOL),
                f"{key} drifted from gold:\n  got={got}\n  ref={ref}",
            )
