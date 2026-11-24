"""Reverse-mode gradients through ``odesolve``.

``odesolve`` lowers to an adaptive Dormand-Prince loop whose trip count depends
on the parameters. ``LogDensity.grad`` differentiates it in reverse mode, which
needs an ``enzyme-ad`` wheel that reverses data-dependent ``stablehlo.while``
loops. On exponential decay the exact solution ``y0 * exp(-k t)`` gives the
gradient in closed form, and Enzyme's forward mode gives an independent check
of the same discrete program.

``ode_sensitivity_control=True`` differentiates a second solve whose step
control includes the sensitivities, as Stan's gradient does. At an exact steady
state the value-only steps grow to the output times, so only that option keeps
the gradient accurate there; a Lotka-Volterra equilibrium has a closed-form
gradient that checks it.
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


@pytest.mark.parametrize("sensitivity_control", [False, True])
@pytest.mark.parametrize(("k", "y0"), [(0.4, 1.9), (0.7, 2.2), (1.3, 1.1)])
def test_a_reverse_gradient_through_odesolve_matches_the_exact_solution(
    k: float, y0: float, sensitivity_control: bool
) -> None:
    density = flatppl(DECAY).logdensity(
        dtype="float64",
        unconstrained=False,
        ode_sensitivity_control=sensitivity_control,
    )
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


LV_TIMES = np.array([0.5, 1.0, 2.0, 4.0])
LV_U = np.array([1.3, 0.8, 1.1, 0.7])
LV_V = np.array([0.9, 1.2, 0.75, 1.15])
LV_SIGMA = 0.25
LV_NAMES = ("alpha", "beta", "gamma", "delta", "u0", "v0")

LOTKA_VOLTERRA = f"""
flatppl_compat = "0.1"
times = {LV_TIMES.tolist()}
observed_u = {LV_U.tolist()}
observed_v = {LV_V.tolist()}
p = elementof(cartprod(alpha = posreals, beta = posreals, gamma = posreals, delta = posreals, u0 = posreals, v0 = posreals))
lv(t, z, th) = [th.alpha * z[1] - th.beta * z[1] * z[2], -th.gamma * z[2] + th.delta * z[1] * z[2]]
traj = rowstack(odesolve(lv, [p.u0, p.v0], 0.0, times, p, rtol = 1e-10, atol = 1e-10))
inputs = p
outputs = logdensityof(Normal.(traj[:, 1], {LV_SIGMA}), observed_u) + logdensityof(Normal.(traj[:, 2], {LV_SIGMA}), observed_v)
"""


def equilibrium_gradient() -> np.ndarray:
    """The gradient at all parameters and both initial states equal to one.

    That point is a steady state, so the trajectory is constant at (1, 1). The
    state Jacobian there is J = [[0, -1], [1, 0]], so the sensitivities to the
    initial state are the rotation exp(J t), and those to a parameter with
    partial derivative f of the right-hand side are the integral of exp(J s) f
    over [0, t], that is [[sin t, cos t - 1], [1 - cos t, sin t]] f.
    """
    s, c = np.sin(LV_TIMES), np.cos(LV_TIMES)
    # (du/dq, dv/dq) for q = alpha, beta, gamma, delta, u0, v0.
    sensitivities = [
        (s, 1 - c),
        (-s, c - 1),
        (1 - c, -s),
        (c - 1, s),
        (c, s),
        (-s, c),
    ]
    score_u = (LV_U - 1.0) / LV_SIGMA**2
    score_v = (LV_V - 1.0) / LV_SIGMA**2
    return np.array([np.sum(score_u * du + score_v * dv) for du, dv in sensitivities])


def lv_point(values: list[float]) -> dict[str, jax.Array]:
    return {
        name: jnp.asarray(value) for name, value in zip(LV_NAMES, values, strict=True)
    }


def test_sensitivity_control_gives_the_exact_gradient_at_a_steady_state() -> None:
    density = flatppl(LOTKA_VOLTERRA).logdensity(
        dtype="float64", unconstrained=False, ode_sensitivity_control=True
    )
    position = lv_point([1.0] * 6)
    gradient = density.grad(position)
    reverse = np.array([gradient[name] for name in LV_NAMES])
    np.testing.assert_allclose(reverse, equilibrium_gradient(), rtol=1e-8)

    forward = np.array(
        [
            jax.jvp(density.fn, (position,), (lv_point(list(np.eye(6)[i])),))[1]
            for i in range(6)
        ]
    )
    np.testing.assert_allclose(reverse, forward, rtol=1e-12)


@pytest.mark.parametrize(
    "values",
    [[1.0] * 6, [0.9, 1.1, 1.05, 0.95, 1.2, 0.8]],
)
def test_sensitivity_control_keeps_the_value(values: list[float]) -> None:
    module = flatppl(LOTKA_VOLTERRA)
    position = lv_point(values)
    default, controlled = (
        module.logdensity(
            dtype="float64", unconstrained=False, ode_sensitivity_control=control
        ).fn(position)
        for control in (False, True)
    )
    np.testing.assert_array_equal(np.asarray(controlled), np.asarray(default))


def test_sensitivity_control_keeps_the_draws_of_a_query_that_draws() -> None:
    query = flatppl("""
        decay(t, z, th) = [-th * z[1]]
        state = rnginit(0)
        k = elementof(posreals)
        inputs = (state, k)
        outputs = rand(Normal(odesolve(decay, [1.0], 0.0, [1.0], k)[1][1], 1.0), state)
    """)
    key = jnp.array([1729, 0], dtype=jnp.uint64)
    default, controlled = (
        query.compile(autodiff=False, ode_sensitivity_control=control)(key, 0.5)
        for control in (False, True)
    )
    np.testing.assert_array_equal(np.asarray(controlled[0]), np.asarray(default[0]))
    np.testing.assert_array_equal(np.asarray(controlled[1]), np.asarray(default[1]))
