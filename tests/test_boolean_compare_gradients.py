"""Gradients of densities that compare a Boolean array with an integer.

``z .== 1`` on Boolean ``z`` lowers to a conversion followed by a comparison.
The alpha 23 ``enzyme-ad`` wheels fold that comparison to ``false`` when they
differentiate, so ``value_and_grad`` returns ``(-inf, 0)`` while the primal is
right. The wheel declared for CPython 3.14 on Linux x86_64 keeps it.
"""

import platform
import sys

import jax
import numpy as np
import pytest

from flatppl import flatppl

# pyproject.toml declares a wheel with this fix only for CPython 3.14 on Linux
# x86_64. The other targets' alpha 23 wheels return (-inf, 0).
pytestmark = pytest.mark.skipif(
    not (
        sys.version_info[:2] == (3, 14)
        and sys.platform == "linux"
        and platform.machine() == "x86_64"
    ),
    reason="this target's declared enzyme-ad wheel folds Boolean comparisons",
)

# Two animals: Bernoulli(p) observations of the ones flagged by z.
POSTERIOR = """
p ~ Uniform(unitinterval)
z ~ iid(Bernoulli(0.5), 2)
y ~ Bernoulli.(ifelse.({flag}, p, 0.0))
P = lawof(record(p=p, z=z))
K = kernelof(record(y=y), p=p, z=z)
L = likelihoodof(K, record(y=[true, false]))
posterior = bayesupdate(L, P)
u = elementof(unitinterval)
v = elementof(cartpow(booleans, 2))
inputs = (u, v)
outputs = logdensityof(posterior, record(p=u, z=v))
"""


@pytest.mark.parametrize("flag", ["z .== 1", "boolean.(z)"])
def test_boolean_compare_value_and_gradient(flag: str) -> None:
    with jax.enable_x64():
        function = flatppl(POSTERIOR.format(flag=flag)).compile(dtype="float64")
        p, z = 0.5, np.array([True, False])
        # z = (true, false): log(1/4) from the prior of z, plus log(p).
        expected = np.log(0.125)
        np.testing.assert_allclose(jax.jit(function)(p, z), expected, rtol=1e-15)
        value, gradient = jax.jit(jax.value_and_grad(function))(p, z)
        np.testing.assert_allclose(value, expected, rtol=1e-15)
        np.testing.assert_allclose(gradient, 1 / p, rtol=1e-15)
