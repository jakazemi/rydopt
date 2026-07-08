import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from rydopt.protocols import GateSystem
from rydopt.pulses import PulseFamilyAnsatz
from rydopt.types import ParamsFloatLike


def _hashable(value):
    if isinstance(value, (list, tuple)):
        return tuple(value)
    try:
        hash(value)
        return value
    except TypeError:
        return str(value)


@dataclass(frozen=True)
class FamilyMember:
    interpolation_parameter: Any
    gate_parameters: Mapping[str, Any]


class GateFamily:
    """Collection of gates evaluated under a shared pulse ansatz with interpolation.

    The infidelity is evaluated independently for each gate and then combined
    according to the specified reduction operation.

    Example:
        >>> import rydopt as ro
        >>> import numpy as np
        >>> target_phases = np.linspace(0.25, 1.0, 4) * np.pi
        >>> sampled_gates = [
        ...     ro.gates.TwoQubitGate(
        ...         phi=None,
        ...         theta=np.pi,
        ...         Vnn=float("inf"),
        ...         decay=0.0001,
        ...     )
        ...     for phase in target_phases
        ... ]
        >>> parametrized_gate = ro.gates.GateFamily(
        ...     fixed_parameter_gates=sampled_gates,
        ...     parameter_values=target_phases,
        ...     reduction="mean",
        ... )

    Args:
        prototype_gate: A gate instance defining the physical system.
        family_members: Sequence of FamilyMember instances.
        reduction: Reduction operation applied to the per-gate infidelities.
            One of {"mean", "max", "softmax"}.
        softmax_scale: Non-negative scale parameter used only when `reduction="softmax"`.
            A value of zero corresponds to the max, and infinity corresponds to the mean.
            Finite values set the scale of per-gate infidelity differences that the
            optimizer should care about.

    """

    def __init__(
        self,
        prototype_gate: GateSystem,
        family_members: Sequence[FamilyMember],
        reduction: Literal["mean", "max", "softmax"] = "mean",
        softmax_scale: float | None = None,
    ) -> None:
        if len(family_members) == 0:
            raise ValueError("family_members cannot be empty.")

        self.prototype_gate = prototype_gate
        self.family_members = list(family_members)
        self._num_gates = len(family_members)

        if reduction == "mean":
            if softmax_scale is not None:
                raise ValueError("softmax_scale may only be provided when reduction='softmax'.")
            self.reduction = float("inf")
        elif reduction == "max":
            if softmax_scale is not None:
                raise ValueError("softmax_scale may only be provided when reduction='softmax'.")
            self.reduction = 0.0
        elif reduction == "softmax":
            if softmax_scale is None:
                raise ValueError("softmax_scale must be provided when reduction='softmax'.")
            self.reduction = softmax_scale
        else:
            raise ValueError("Invalid reduction, must be 'mean', 'max', or 'softmax'.")

        control_flow_keys = prototype_gate.control_flow_keys()
        none_sensitive_keys = prototype_gate.none_sensitive_keys()

        def cf_signature(member: FamilyMember) -> tuple:
            sig = []
            for key in sorted(control_flow_keys):
                value = member.gate_parameters.get(key, "<prototype>")
                sig.append((key, value if value == "<prototype>" else _hashable(value)))
            for key in sorted(none_sensitive_keys):
                value = member.gate_parameters.get(key, getattr(prototype_gate, f"_{key}"))
                sig.append((key, value is None))
            return tuple(sig)

        groups: dict[tuple, list[int]] = {}
        for i, member in enumerate(self.family_members):
            groups.setdefault(cf_signature(member), []).append(i)

        self._groups = []
        for indices in groups.values():
            rep_member = self.family_members[indices[0]]
            base_kwargs = {k: v for k, v in rep_member.gate_parameters.items() if k in control_flow_keys}
            base_gate = prototype_gate.replace(**base_kwargs)

            dynamic_keys = [k for k in rep_member.gate_parameters.keys() if k not in control_flow_keys]
            stacked_dynamic = {
                key: jnp.stack([jnp.asarray(self.family_members[i].gate_parameters[key]) for i in indices])
                for key in dynamic_keys
            }
            interp_params = jax.tree_util.tree_map(
                lambda *xs: jnp.stack(xs),
                *[self.family_members[i].interpolation_parameter for i in indices],
            )

            self._groups.append(
                dict(
                    indices=jnp.asarray(indices),
                    base_gate=base_gate,
                    stacked_dynamic=stacked_dynamic,
                    interp_params=interp_params,
                )
            )

    def cost(self, pulse: PulseFamilyAnsatz, params: ParamsFloatLike, tol: float) -> jax.Array:
        """Compute reduced infidelity over all fixed-target-parameter gates defined within the
        gate family.

        Args:
            pulse: Pulse family ansatz used for all gates.
            params: Trainable pulse family parameters.
            tol: Numerical tolerance passed to the cost function.

        Returns:
            Reduced infidelity value according to `self.reduction`.

        """
        pulse_ansatz = pulse.pulse_ansatz
        all_costs = jnp.zeros(self._num_gates)

        for group in self._groups:
            base_gate = group["base_gate"]

            def single_cost(interp_param, dynamic_vals):
                gate = base_gate.replace(**dynamic_vals) if dynamic_vals else base_gate
                pulse_params = pulse.generate_pulse_params(params, interp_param)
                return gate.cost(pulse_ansatz, pulse_params, tol)

            group_costs = jax.vmap(single_cost, in_axes=(0, 0))(group["interp_params"], group["stacked_dynamic"])
            all_costs = all_costs.at[group["indices"]].set(group_costs)

        if self.reduction == 0.0:
            return jnp.max(all_costs)
        if math.isinf(self.reduction):
            return jnp.mean(all_costs)
        return self.reduction * (logsumexp(all_costs / self.reduction) - math.log(self._num_gates))
