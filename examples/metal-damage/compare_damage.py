#!/usr/bin/env python3

import numpy as np

import matplotlib.pyplot as plt

import sys

sys.path.append("../..")

from srlife import receiver, damage, library, solverparams

if __name__ == "__main__":
    model = receiver.Receiver.load("receiver.hdf5")
    _, _, material = library.load_material("740H", "base", "base", "base")

    params = solverparams.ParameterSet()

    det_dmg = damage.TimeFractionInteractionDamage(params)
    stat_dmg = damage.StatisticalTimeFractionDamage(params)

    for pi, panel in model.panels.items():
        for ti, tube in panel.tubes.items():
            Dc = np.max(det_dmg.creep_damage(tube, material, model))
            N = 1.0 / Dc
            print(f"{pi}-{ti}: {N:.2f} cycles")

            reps, cdf = stat_dmg.cdf_tube_chain(2 * N, tube, material, model)
            reps, cdf2 = stat_dmg.cdf_tube_max(2 * N, tube, material, model)

            plt.plot(reps, cdf, label="Chained")
            plt.plot(reps, cdf2, label="Max")
            plt.axvline(N, color="k", linestyle="--", lw=3)
            plt.xlabel("Cycles")
            plt.ylabel("CDF")
            plt.legend(loc="best")

            plt.show()
