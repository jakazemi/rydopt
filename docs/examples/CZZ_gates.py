import pickle
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

import rydopt as ro

if __name__ == "__main__":
    # a symmetric ansatz for the CZZ gate on three atoms arranged on a line with finite interactions
    Vnn = 32
    phase = np.pi / 4
    gate = ro.gates.ThreeQubitGateIsosceles(
        phi=None,
        theta=4 * phase,
        theta_prime=8 * phase,
        lamb=-8 * phase,
        Vnn=Vnn,
        Vnnn=Vnn / 2**6,
        decay=0.0001,
    )
    # Pulse ansatz: constant detuning, sweep of the laser phase according to
    # a sine CRAB ansatz, and smooth ramps for the Rabi frequency
    n_phase_params = 14
    pulse_ansatz = ro.pulses.PulseAnsatz(
        detuning_ansatz=ro.pulses.Const(),
        phase_ansatz=ro.pulses.SinCrab(n_phase_params),
        rabi_ansatz=ro.pulses.SoftBoxSeventhOrderSmoothstep(),
    )

    # Bounds for the initial pulse parameter guesses
    min_initial_params = ro.pulses.PulseParams(
        10, [0], -1 * np.ones(n_phase_params), [1.0, 0.5]
    )
    max_initial_params = ro.pulses.PulseParams(
        30, [0], 1 * np.ones(n_phase_params), [1.0, 0.5]
    )
    fixed_initial_params = ro.pulses.PulseParams(
        False,
        [True],
        [
            False,
        ]
        * n_phase_params,
        [True, True],
    )
    # Optimize the pulse parameters
    opt_result = ro.optimization.multi_start_optimize(
        gate,
        pulse_ansatz,
        min_initial_params,
        max_initial_params,
        fixed_initial_params,
        tol=1e-7,
        num_initializations=200,
        num_steps=200,
        num_processes=4,
        return_history=False,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = (
        f"results/CZZ_"
        f"n{n_phase_params}_"
        f"fid_{opt_result.infidelity:.2e}_"
        f"T_{opt_result.duration:.2f}_"
        f"{timestamp}.pkl"
    )

    with open(filename, "wb") as f:
        pickle.dump(opt_result, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(filename, "rb") as f:
        opt_result = pickle.load(f)

    optimized_params = opt_result.params

    print(optimized_params)

    # Determine the gate's infidelity, infidelity without decay, and Rydberg time
    infidelity, infidelity_nodecay, ryd_time = ro.characterization.analyze_gate(
        gate, pulse_ansatz, optimized_params
    )

    # Print the gate performance measures
    print("\n=== Performance analysis of the best/fastest optimized gate pulse ===\n")
    print(f"Gate infidelity:             {infidelity:.4e}")
    print(f"Gate infidelity (no decay):  {infidelity_nodecay:.4e}")
    print(f"Rydberg time:                {ryd_time:.4f}")

    ro.characterization.plot_spectrum(pulse_ansatz, optimized_params)
    ro.characterization.plot_pulse(pulse_ansatz, optimized_params)
    plt.show()
