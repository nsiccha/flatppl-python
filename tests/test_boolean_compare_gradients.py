"""Gradients of densities that compare a Boolean array with an integer.

``z .== 1`` on Boolean ``z`` lowers to a conversion followed by a comparison.
The alpha 23 ``enzyme-ad`` wheels fold that comparison to ``false`` when they
differentiate, so ``value_and_grad`` returns ``(-inf, 0)`` while the primal is
right. Every wheel pyproject.toml declares keeps it.
"""

import jax
import numpy as np
import pytest

from flatppl import flatppl

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
