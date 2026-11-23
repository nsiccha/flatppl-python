"""Reverse-mode gradients through ``odesolve``.

``odesolve`` lowers to an adaptive Dormand-Prince loop whose trip count depends
on the parameters. ``LogDensity.grad`` differentiates it in reverse mode, which
needs an ``enzyme-ad`` wheel that reverses data-dependent ``stablehlo.while``
loops. On exponential decay the exact solution ``y0 * exp(-k t)`` gives the
gradient in closed form, and Enzyme's forward mode gives an independent check
of the same discrete program.
"""

import platform
import sys
from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flatppl import flatppl

# pyproject.toml declares a wheel with this capability only for CPython 3.14 on
# Linux x86_64. The other targets' alpha 23 wheels refuse with "WhileOp does not
# have known iteration count for cache removal".
pytestmark = pytest.mark.skipif(
    not (
        sys.version_info[:2] == (3, 14)
        and sys.platform == "linux"
        and platform.machine() == "x86_64"
    ),
    reason="this target's declared enzyme-ad wheel cannot reverse adaptive loops",
)

TIMES = np.array([0.5, 1.0, 2.0, 4.0])
OBSERVED = np.array([1.55, 1.12, 0.62, 0.21])
SIGMA = 0.1

DECAY = f"""
flatppl_compat = "0.1"
times = {TIMES.tolist()}
observed = {OBSERVED.tolist()}
p = elementof(cartprod(k = posreals, y0 = posreals))
decay(t, z, th) = [-th.k * z[1]]
traj = rowstack(odesolve(decay, [p.y0], 0.0, times, p, rtol = 1e-10, atol = 1e-10))
inputs = p
outputs = logdensityof(Normal.(traj[:, 1], {SIGMA}), observed)
"""


@pytest.fixture(autouse=True)
def x64() -> Iterator[None]:
    with jax.enable_x64(True):
        yield


def exact_gradient(k: float, y0: float) -> np.ndarray:
    decayed = np.exp(-k * TIMES)
    mean = y0 * decayed
    score = (OBSERVED - mean) / SIGMA**2
    return np.array([np.sum(score * -TIMES * mean), np.sum(score * decayed)])


@pytest.mark.parametrize(("k", "y0"), [(0.4, 1.9), (0.7, 2.2), (1.3, 1.1)])
def test_a_reverse_gradient_through_odesolve_matches_the_exact_solution(
    k: float, y0: float
) -> None:
    density = flatppl(DECAY).logdensity(dtype="float64", unconstrained=False)
    position = {"k": jnp.asarray(k), "y0": jnp.asarray(y0)}
    gradient = density.grad(position)
    reverse = np.array([gradient["k"], gradient["y0"]])
    np.testing.assert_allclose(reverse, exact_gradient(k, y0), rtol=1e-7)

    forward = np.array(
        [
            jax.jvp(density.fn, (position,), (tangent,))[1]
            for tangent in (
                {"k": jnp.asarray(1.0), "y0": jnp.asarray(0.0)},
                {"k": jnp.asarray(0.0), "y0": jnp.asarray(1.0)},
            )
        ]
    )
    np.testing.assert_allclose(reverse, forward, rtol=1e-10)
