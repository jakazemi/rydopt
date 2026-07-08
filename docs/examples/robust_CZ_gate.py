import pickle
import re
from datetime import datetime

import jax
import numpy as np
from matplotlib import pyplot as plt

import rydopt as ro

if __name__ == "__main__":
    run_optimization = False
    plot_results = True
    # n_phase_params = 6
    softmax_scale = 0
    softmax_scale_list = [0]
    n_phase_params_list = [6, 10]
    # for iteration, n_phase_params, softmax_scale in product(
    #         range(1), n_phase_params_list, softmax_scale_list):
    for iteration in range(3):
        # a symmetric ansatz for the parity phase gate on three atoms with finite interactions
        target_params = np.random.normal(2.8, 0.02, 8)
        # target_params = [2.8] * 8
        sampled_gates = [
            ro.gates.TwoQubitGate(
                phi=None,
                theta=np.pi,
                Vnn=15400 / target_param**6,
                decay=0.0001,
            )
            for target_param in target_params
        ]

        gate_family = ro.gates.GateFamily(
            fixed_parameter_gates=sampled_gates,
            parameter_values=target_params,
            reduction="softmax",
            softmax_scale=softmax_scale,
        )

        if not run_optimization:
            filename = "results/robust_CZ/"
            if iteration == 0:
                filename = "results/CZ_n4_fid_2.62e-04_T_10.35_20260702_111444.pkl"
                # filename += "robust_CZ_n4_fid_4.40e-03_T_10.51_20260703_010207.pkl"
                # filename += "robust_CZ_n4_fid_9.29e-03_T_10.54_20260703_010553.pkl"
            elif iteration == 1:
                filename += "robust_CZ_n6_fid_2.26e-04_T_12.74_20260703_111512.pkl"
            elif iteration == 2:
                # filename += "robust_CZ_n6_fid_1.68e-04_T_16.66_20260703_011400.pkl"
                filename += "robust_CZ_n10_fid_2.15e-04_T_11.97_20260703_112229.pkl"
            elif iteration == 3:
                # filename += "robust_CZ_n8_fid_7.99e-04_T_15.54_20260703_012949.pkl"
                filename += "robust_CZ_n8_fid_2.35e-03_T_13.22_20260703_014241.pkl"
            else:
                # filename += "robust_CZ_n10_fid_8.86e-04_T_13.52_20260703_014920.pkl"
                filename += "robust_CZ_n10_fid_2.15e-03_T_13.71_20260703_020314.pkl"
            match = re.search(r"n(\d+)_", filename)
            if match:
                n_phase_params = int(match.group(1))

        # Pulse ansatz: constant detuning, sweep of the laser phase according to
        # a sine CRAB ansatz, and smooth ramps for the Rabi frequency
        degrees = [0, 0, 0, 0]  # duration, detuning, laser phase, Rabi amplitude
        pulse_map = ro.pulses.PolynomialPulseMap(degrees=degrees)
        pulse_family = ro.pulses.PulseFamilyAnsatz(
            detuning_ansatz=ro.pulses.Const(),
            phase_ansatz=ro.pulses.SinCrab(n_phase_params),
            rabi_ansatz=ro.pulses.SoftBoxSeventhOrderSmoothstep(),
            pulse_map=pulse_map,
        )

        # Bounds for the initial pulse parameter guesses
        min_initial_params = ro.pulses.PulseFamilyParams(
            [9],
            np.zeros(degrees[1] + 1),
            -1 * np.ones((n_phase_params, degrees[2] + 1)),
            [1.0, 0.5],
        )

        max_initial_params = ro.pulses.PulseFamilyParams(
            [
                12,
            ],
            np.zeros(degrees[1] + 1),
            1 * np.ones((n_phase_params, degrees[2] + 1)),
            [1.0, 0.5],
        )

        fixed_initial_params = ro.pulses.PulseFamilyParams(
            [False],
            [True],
            [
                [
                    False,
                ]
                * (degrees[2] + 1)
            ]
            * n_phase_params,
            [True, True],
        )
        if run_optimization:
            # Optimize the pulse parameters
            opt_result = ro.optimization.multi_start_optimize(
                gate_family,
                pulse_family,
                min_initial_params,
                max_initial_params,
                fixed_initial_params,
                tol=5e-4,
                num_initializations=80,
                num_steps=1000,
                num_processes=4,
                return_history=False,
                return_all=True,
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = (
                f"results/robust_CZ/robust_CZ_"
                f"n{n_phase_params}_"
                f"fid_{opt_result.infidelity[0]:.2e}_"
                f"T_{opt_result.duration[0]:.2f}_"
                f"{timestamp}.pkl"
            )

            with open(filename, "wb") as f:
                pickle.dump(opt_result, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(filename, "rb") as f:
            opt_result = pickle.load(f)
        optimized_family_params = opt_result.params
        if isinstance(optimized_family_params, list):
            optimized_family_params = optimized_family_params[0]

        print(optimized_family_params)
        if plot_results:
            target_distance = jax.numpy.linspace(2.6, 3.0, 20)
            target_sigma = 1e-2 * jax.numpy.linspace(-1, 1, 20)
            infidelities_distance = []
            infidelities_sigma = []

            def analyze_interaction(
                target_param: float, family_params: ro.pulses.PulseFamilyParams
            ) -> tuple[float, float]:
                pulse = pulse_family.pulse_ansatz
                params = pulse_family.generate_pulse_params(family_params, target_param)
                gate = ro.gates.TwoQubitGate(
                    phi=None,
                    theta=np.pi,
                    Vnn=15400 / target_param**6,
                    decay=0.0001,
                )
                infidelity = 1 - ro.simulation.process_fidelity(gate, pulse, params)
                return infidelity

            @jax.jit
            def analyze_rabi(
                target_param: float, family_params: ro.pulses.PulseFamilyParams
            ) -> tuple[float, float]:
                pulse = pulse_family.pulse_ansatz
                params = pulse_family.generate_pulse_params(
                    family_params,
                    2.8,
                    key=jax.random.key(0),
                    sigma=target_param / jax.random.normal(jax.random.key(0)),
                )
                gate = ro.gates.TwoQubitGate(
                    phi=None,
                    theta=np.pi,
                    Vnn=15400 / 2.8**6,
                    decay=0.0001,
                )
                infidelity = 1 - ro.simulation.process_fidelity(gate, pulse, params)
                return infidelity

            for target_param in target_distance:
                infidelity = analyze_interaction(target_param, optimized_family_params)
                infidelities_distance.append(float(infidelity))
            for target_param in target_sigma:
                infidelity = analyze_rabi(target_param, optimized_family_params)
                infidelities_sigma.append(float(infidelity))

            fig = plt.figure(figsize=(4, 4), dpi=160)
            ax1 = fig.add_subplot(111)

            # First curve
            ax1.plot(target_sigma, infidelities_sigma, color="k")
            ax1.set_xlabel(r"$\epsilon_\Omega$", labelpad=2)
            ax1.set_ylabel("Infidelity", labelpad=8)
            ax1.set_xlim(target_sigma[0] - 1e-5, target_sigma[-1] + 1e-5)
            ax1.set_ylim(1e-4, 4e-3)
            ax1.set_yscale("log")
            ax1.tick_params(which="both", direction="in")
            ax1.grid(True, which="minor", axis="both", alpha=0.3)

            # Second curve
            fig.subplots_adjust(top=0.85, left=0.25)
            ax2 = fig.add_axes(ax1.get_position(), frameon=False)
            ax2.plot(target_distance, infidelities_distance, color="C0")
            ax2.xaxis.set_label_position("top")
            ax2.xaxis.tick_top()
            ax2.set_xlim(target_distance[0], target_distance[-1])
            ax2.set_xlabel(r"$R\, [\rm{\mu m}]$", color="C0", labelpad=8)
            ax2.tick_params(axis="x", colors="C0", which="both", direction="in")
            ax2.grid(alpha=0.3)

            # Hide duplicate y-axis
            ax2.yaxis.set_visible(False)
            ax2.set_yscale("log")
            ax2.set_ylim(ax1.get_ylim())

            ro.characterization.plot_pulse_family(
                pulse_family,
                optimized_family_params,
                gate_family,
                plot_detuning=False,
                plot_rabi=True,
            )
    if plot_results:
        plt.show()
