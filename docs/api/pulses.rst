rydopt.pulses
=============

.. currentmodule:: rydopt.pulses

.. autoclass:: PulseAnsatz
   :members:

.. autoclass:: TwoPhotonPulseAnsatz
   :members:

.. autoclass:: PulseParams

.. py:type:: ParamsFloatLike
   :canonical: tuple[float, FloatParamComponent, FloatParamComponent, FloatParamComponent] | FloatParamComponent

   Pulse configuration as either ``PulseParams(duration, detuning_params, phase_params, rabi_params)``
   or a packed parameter array/sequence.

   - **duration** - Gate duration
   - **detuning_params** - Parameters for the detuning sweep
   - **phase_params** - Parameters for the phase sweep
   - **rabi_params** - Parameters for the Rabi frequency amplitude sweep

.. py:type:: ParamsBoolLike
   :canonical: tuple[bool, BoolParamComponent, BoolParamComponent, BoolParamComponent] | BoolParamComponent

   Boolean masks as either ``PulseParams(fixed_duration, fixed_detuning_params, fixed_phase_params, fixed_rabi_params)``
   or a packed boolean mask array/sequence, marking which pulse parameters are held constant during optimization.

   - **fixed_duration** - Whether the gate duration is fixed
   - **fixed_detuning_params** - Boolean mask of fixed detuning parameters
   - **fixed_phase_params** - Boolean mask of fixed phase parameters
   - **fixed_rabi_params** - Boolean mask of fixed Rabi frequency amplitude parameters

General Pulse Ansatz Functions
------------------------------

.. automodule:: rydopt.pulses.general_pulse_ansatz_functions
   :members:

Soft-Box Pulse Ansatz Functions
-------------------------------

.. automodule:: rydopt.pulses.softbox_pulse_ansatz_functions
   :members:
