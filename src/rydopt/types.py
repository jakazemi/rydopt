from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, Unpack, cast, overload

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

FidelityType = Literal["process", "average_gate"]

FloatParamComponent = Sequence[float] | jax.Array | npt.NDArray[np.float64]
ParamsLike = tuple[float, Unpack[tuple[FloatParamComponent, ...]]] | FloatParamComponent

BoolParamComponent = Sequence[bool] | jax.Array | npt.NDArray[np.bool_]
FixedParamsLike = tuple[bool, Unpack[tuple[BoolParamComponent, ...]]] | BoolParamComponent


@overload
def _ravel(
    params: ParamsLike,
    *,
    dtype: Literal["float"],
    backend: Literal["numpy"] = "numpy",
) -> npt.NDArray[np.float64]: ...


@overload
def _ravel(
    params: ParamsLike,
    *,
    dtype: Literal["float"],
    backend: Literal["jax"],
) -> jax.Array: ...


@overload
def _ravel(
    params: FixedParamsLike,
    *,
    dtype: Literal["bool"],
    backend: Literal["numpy"] = "numpy",
) -> npt.NDArray[np.bool_]: ...


@overload
def _ravel(
    params: FixedParamsLike,
    *,
    dtype: Literal["bool"],
    backend: Literal["jax"],
) -> jax.Array: ...


def _ravel(
    params: ParamsLike | FixedParamsLike,
    *,
    dtype: Literal["float", "bool"],
    backend: Literal["numpy", "jax"] = "numpy",
) -> npt.NDArray[np.float64] | npt.NDArray[np.bool_] | jax.Array:
    if backend == "jax":
        jax_dtype = jnp.float64 if dtype == "float" else jnp.bool_
        if isinstance(params, tuple):
            first, *rest = params
            first_arr = jnp.asarray(first, dtype=jax_dtype)
            first_arr = first_arr[jnp.newaxis] if first_arr.ndim == 0 else first_arr[..., jnp.newaxis]
            return jnp.concatenate(
                [first_arr, *(jnp.asarray(component, dtype=jax_dtype) for component in rest)],
                axis=-1,
            )
        return jnp.ravel(jnp.asarray(params, dtype=jax_dtype))

    np_dtype = np.float64 if dtype == "float" else np.bool_
    if isinstance(params, tuple):
        first, *rest = params
        first_arr = np.asarray(first, dtype=np_dtype)
        first_arr = first_arr[np.newaxis] if first_arr.ndim == 0 else first_arr[..., np.newaxis]
        return np.concatenate(
            [first_arr, *(np.asarray(component, dtype=np_dtype) for component in rest)],
            axis=-1,
        )  # pyright: ignore[reportReturnType]
    return np.ravel(np.asarray(params, dtype=np_dtype))  # pyright: ignore[reportReturnType]


@overload
def _unravel(
    flat: npt.NDArray[np.float64],
    split_indices: tuple[int, ...],
    *,
    dtype: Literal["float"],
    backend: Literal["numpy"] = "numpy",
) -> ParamsLike: ...


@overload
def _unravel(
    flat: jax.Array,
    split_indices: tuple[int, ...],
    *,
    dtype: Literal["float"],
    backend: Literal["jax"],
) -> ParamsLike: ...


@overload
def _unravel(
    flat: npt.NDArray[np.bool_],
    split_indices: tuple[int, ...],
    *,
    dtype: Literal["bool"],
    backend: Literal["numpy"] = "numpy",
) -> FixedParamsLike: ...


@overload
def _unravel(
    flat: jax.Array,
    split_indices: tuple[int, ...],
    *,
    dtype: Literal["bool"],
    backend: Literal["jax"],
) -> FixedParamsLike: ...


def _unravel(
    flat: npt.NDArray[np.float64] | npt.NDArray[np.bool_] | jax.Array,
    split_indices: tuple[int, ...],
    *,
    dtype: Literal["float", "bool"],
    backend: Literal["numpy", "jax"] = "numpy",
) -> ParamsLike | FixedParamsLike:
    if not split_indices:
        return flat

    if backend == "jax":
        parts = jnp.split(flat, split_indices, axis=-1)
        return cast(ParamsLike | FixedParamsLike, (parts[0][..., 0], *parts[1:]))

    parts = np.split(flat, split_indices, axis=-1)
    first_scalar = np.asarray(parts[0]).reshape(-1)[0]

    if dtype == "float":
        return cast(ParamsLike, (float(first_scalar), *parts[1:]))
    return cast(FixedParamsLike, (bool(first_scalar), *parts[1:]))


PulseFunction = Callable[[float | jax.Array], jax.Array]

HamiltonianFunction = Callable[[float | jax.Array, float | jax.Array, float | jax.Array, float | jax.Array], jax.Array]
