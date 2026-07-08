import pickle
import re
from datetime import datetime

import jax
import numpy as np
from matplotlib import pyplot as plt
import jax.numpy as jnp

import rydopt as ro
from rydopt.gates.gate_family import FamilyMember


def empirical_cphase_duration(
    target_phase: jax.Array, duration_params: jax.Array
) -> jax.Array:
    if duration_params.size != 3:
        raise ValueError(
            "empirical_cphase_duration expects three duration parameters: (piduration, prefactor, exponent)"
        )

    piduration, prefactor, exponent = duration_params

    # phase_over_2pi = jnp.clip(target_phase / (2.0 * jnp.pi), 0.0, 1.0)
    # x = 2.0 * jnp.minimum(phase_over_2pi, 1.0 - phase_over_2pi)
    phase_over_2pi = jnp.clip(target_phase / (2.0 * jnp.pi), 0.0, 1.0)
    x = 1.0 - jnp.minimum(phase_over_2pi, 1.0 - phase_over_2pi)

    q = prefactor / (piduration * 2.0**exponent)
    safe_inner = jnp.maximum(1.0 - x**exponent, jnp.finfo(x.dtype).tiny)
    duration = piduration * (-jnp.expm1(q * jnp.log(safe_inner)))
    return jnp.where(x >= 1.0, piduration, duration)


if __name__ == "__main__":
    run_optimization = True
    use_optimal_cz_params = False
    plot_results = True
    n_phase_params = 4
    for _ in range(1):
        # a symmetric ansatz for the parity phase gate on two atoms with finite interactions
        # target_phases = np.logspace(-2, 0, 16) * np.pi
        target_phases = np.linspace(0.01, 1.0, 10) * np.pi
        # target_phases = [np.pi]
        # sampled_gates = [
        #     ro.gates.TwoQubitGate(
        #         phi=None,
        #         theta=np.pi-phase,
        #         Vnn=32,
        #         decay=0.0001,
        #     )
        #     for phase in target_phases
        # ]
        # gate_family = ro.gates.GateFamily(
        #     fixed_parameter_gates=sampled_gates,
        #     parameter_values=target_phases,
        #     reduction="softmax",
        #     softmax_scale=0,
        # )

        prototype_gate = ro.gates.TwoQubitGate(
            phi=None,
            theta=None,
            Vnn=32,
            decay=0.0001,
        )
        family_members = [
            FamilyMember(
                interpolation_parameter=phase, gate_parameters={"theta": np.pi - phase}
            )
            for phase in target_phases
        ]
        gate_family = ro.gates.GateFamily(
            prototype_gate=prototype_gate,
            family_members=family_members,
            reduction="softmax",
            softmax_scale=0,
        )

        if not run_optimization:
            filename = "results/ZZ_family_n4_fid_1.50e-04_20260703_162124.pkl"
            # filename = "results/ZZ_family_n6_fid_1.55e-04_20260703_175932.pkl"
            # filename = "results/ZZ_family_n6_fid_2.81e-04_20260705_135337.pkl"
            match = re.search(r"n(\d+)_", filename)
            if match:
                n_phase_params = int(match.group(1))

        # Pulse ansatz: constant detuning, sweep of the laser phase according to
        # a sine CRAB ansatz, and smooth ramps for the Rabi frequency
        degrees = [2, 0, 3, 0]  # duration, detuning, laser phase, Rabi amplitude
        pulse_map = ro.pulses.PolynomialPulseMapWithCustomDuration(
            degrees=degrees,
        )
        # pulse_map = ro.pulses.PolynomialPulseMap(degrees=degrees)
        pulse_family = ro.pulses.PulseFamilyAnsatz(
            detuning_ansatz=ro.pulses.Const(),
            phase_ansatz=ro.pulses.SinCrab(n_phase_params),
            rabi_ansatz=ro.pulses.SoftBoxSeventhOrderSmoothstep(),
            pulse_map=pulse_map,
        )

        if run_optimization:
            if use_optimal_cz_params:
                filename = "results/CZ_n4_fid_2.62e-04_T_10.35_20260702_111444.pkl"
                with open(filename, "rb") as f:
                    opt_cz_result = pickle.load(f)
                print(opt_cz_result)
                optimized_cz_params = opt_cz_result.params
                if isinstance(optimized_cz_params, list):
                    optimized_cz_params = optimized_cz_params[0]
                optimized_cz_phase_params = optimized_cz_params.phase_params

                # Bounds for the initial pulse parameter guesses
                optimized_cz_phase_params_extended = np.concatenate(
                    (
                        optimized_cz_phase_params,
                        np.zeros(n_phase_params - len(optimized_cz_phase_params)),
                    )
                )
                min_initial_phase_params = np.column_stack(
                    (
                        optimized_cz_phase_params_extended,
                        -1 * np.ones((n_phase_params, degrees[2])),
                    )
                )

                min_initial_params = ro.pulses.PulseFamilyParams(
                    # [10.354485, 12, 0.2],
                    [optimized_cz_params.duration[0]]
                    + [
                        0,
                    ]
                    * degrees[0],
                    np.zeros(degrees[1] + 1),
                    min_initial_phase_params,
                    [1.0, 0.5],
                )

                optimized_cz_phase_params_extended = np.concatenate(
                    (
                        optimized_cz_phase_params,
                        np.zeros(n_phase_params - len(optimized_cz_phase_params)),
                    )
                )
                max_initial_phase_params = np.column_stack(
                    (
                        optimized_cz_phase_params_extended,
                        +1 * np.ones((n_phase_params, degrees[2])),
                    )
                )

                max_initial_params = ro.pulses.PulseFamilyParams(
                    # [10.354485, 17, 0.3],
                    [optimized_cz_params.duration[0]]
                    + [
                        10,
                    ]
                    * degrees[0],
                    np.zeros(degrees[1] + 1),
                    max_initial_phase_params,
                    [1.0, 0.5],
                )

                fixed_initial_params = ro.pulses.PulseFamilyParams(
                    # [True, False, False],
                    [True]
                    + [
                        False,
                    ]
                    * degrees[0],
                    [True],
                    [
                        [True]
                        + [
                            False,
                        ]
                        * degrees[2]
                    ]
                    * n_phase_params,
                    [True, True],
                )
            else:
                # Bounds for the initial pulse parameter guesses
                min_initial_params = ro.pulses.PulseFamilyParams(
                    [10, 12, 0.2],
                    # [10],
                    np.zeros(degrees[1] + 1),
                    -5 * np.ones((n_phase_params, degrees[2] + 1)),
                    [1.0, 0.5],
                )

                max_initial_params = ro.pulses.PulseFamilyParams(
                    [20, 17, 0.3],
                    # [13],
                    np.zeros(degrees[1] + 1),
                    5 * np.ones((n_phase_params, degrees[2] + 1)),
                    [1.0, 0.5],
                )

                fixed_initial_params = ro.pulses.PulseFamilyParams(
                    [False, False, False],
                    # [False],
                    [
                        True,
                    ]
                    * (degrees[1] + 1),
                    [
                        [
                            False,
                        ]
                        * (degrees[2] + 1)
                    ]
                    * n_phase_params,
                    [True, True],
                )
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
                f"results/ZZ_family_"
                f"n{n_phase_params}_"
                f"fid_{opt_result.infidelity[0]:.2e}_"
                # f"T_{opt_result.duration:.2f}_"
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
            phases = np.linspace(0.01, 1.0, 20) * np.pi
            durations = []
            infidelities = []

            @jax.jit
            def analyze_phase(
                phase: float, family_params: ro.pulses.PulseFamilyParams
            ) -> tuple[jax.Array, jax.Array]:
                pulse = pulse_family.pulse_ansatz
                params = pulse_family.generate_pulse_params(family_params, phase)
                gate = ro.gates.TwoQubitGate(
                    phi=None,
                    theta=np.pi - phase,
                    Vnn=32,
                    decay=0.0001,
                )
                infidelity = 1 - ro.simulation.process_fidelity(gate, pulse, params)
                return params.duration[0], infidelity

            for phase in phases:
                duration, infidelity = analyze_phase(phase, optimized_family_params)
                durations.append(float(duration))
                infidelities.append(float(infidelity))

            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(4, 4), dpi=160)
            ax1.plot((np.pi - phases) / np.pi, durations)
            ax1.set_ylabel("Duration")
            ax2.plot((np.pi - phases) / np.pi, infidelities, "k-")
            ax2.set_xlim((np.pi - phases[-1]) / np.pi, (np.pi - phases[0]) / np.pi)
            ax2.set_ylabel("Infidelity")
            ax2.set_xlabel(r"Target phase $\times \pi^{-1}$")
            plt.tight_layout()
            # plt.show()
            plt.savefig(filename[:-4] + "_infidelity.pdf")
            ro.characterization.plot_pulse_family(
                pulse_family,
                optimized_family_params,
                gate_family,
                plot_detuning=False,
                plot_rabi=True,
                subtract_phase_offset=True,
            )
            plt.savefig(filename[:-4] + "_pulses.pdf")
            # plt.show()
