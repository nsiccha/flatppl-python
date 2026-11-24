"""Cholesky factors of correlation matrices, the support of LKJCholesky.

The PosteriorDB posteriors ``dogs-dogs_nonhierarchical``, ``sat-hier_2pl`` and
``synthetic_grid_RBF_kernels-kronecker_gp`` declare ``cholesky_factor_corr[n]``
parameters with ``lkj_corr_cholesky(eta)`` priors for n = 2, 2 and 30. The
reference below transcribes Stan Math's ``cholesky_corr_constrain`` (with its
log-Jacobian) and ``lkj_corr_cholesky_lpdf``, so these densities score the
same unconstrained coordinates as Stan.
"""

import math
from collections.abc import Iterator

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flatppl import flatppl


@pytest.fixture(autouse=True)
def x64() -> Iterator[None]:
    with jax.enable_x64(True):
        yield


def stan_cholesky_corr_constrain(y: jax.Array, n: int) -> tuple[jax.Array, jax.Array]:
    """Stan's chart: row-major strictly lower coordinates and the log-Jacobian.

    Stan fills each row left to right. This fills one column of every row at a
    time, with the same arithmetic per entry: a row's running sum of squares
    scales its next entry by ``sqrt(1 - sum_sqs)``, adds ``0.5 * log1m(sum_sqs)``
    before each entry after the first, and sets its diagonal to
    ``sqrt(1 - sum_sqs)``.
    """
    z = jnp.tanh(y)
    lp = jnp.sum(jnp.log1p(-(z**2)))
    lower, columns = np.tril_indices(n, -1)
    strict = jnp.zeros((n, n), y.dtype).at[lower, columns].set(z)
    row = np.arange(n)
    sum_sqs = jnp.zeros(n, y.dtype)
    factor: list[jax.Array] = []
    for j in range(n):
        below = row > j
        if j > 0:
            lp += jnp.sum(jnp.where(below, 0.5 * jnp.log1p(-sum_sqs), 0.0))
        entry = jnp.where(below, strict[:, j] * jnp.sqrt(1.0 - sum_sqs), 0.0)
        diagonal = jnp.where(row == j, jnp.sqrt(1.0 - sum_sqs), 0.0)
        factor.append(entry + diagonal)
        sum_sqs += entry**2
    return jnp.stack(factor, axis=1), lp


def stan_lkj_corr_cholesky_lpdf(factor: jax.Array, eta: float) -> jax.Array:
    """Stan's normalized density for eta != 1 (Lewandowski et al. 2009, thm 5)."""
    n = factor.shape[0]
    constant = (n - 1) * math.lgamma(eta + 0.5 * (n - 1))
    for k in range(1, n):
        constant -= 0.5 * k * math.log(math.pi) + math.lgamma(eta + 0.5 * (n - 1 - k))
    log_diagonal = jnp.log(jnp.diagonal(factor)[1:])
    weights = jnp.arange(n - 2, -1, -1) + 2.0 * eta - 2.0
    return constant + jnp.sum(weights * log_diagonal)


def stan_unconstrained(y: jax.Array, n: int, eta: float) -> jax.Array:
    factor, lp = stan_cholesky_corr_constrain(y, n)
    return stan_lkj_corr_cholesky_lpdf(factor, eta) + lp


@pytest.mark.parametrize("eta", [2.0, 4.0])
def test_a_two_by_two_factor_has_one_tanh_coordinate_and_normalizes(
    eta: float,
) -> None:
    # dogs_nonhierarchical (eta = 2) and hier_2pl (eta = 4).
    density = flatppl(f"""
        L = elementof(corrcholesky(2))
        inputs = L
        outputs = logdensityof(LKJCholesky(2, {eta}), L)
    """).logdensity(dtype="float64")
    assert dict(density.shapes) == {"L": (2, 2)}
    assert density.flat.names == ("L[1]",)
    np.testing.assert_array_equal(density.constrain(density.init), np.eye(2))
    lower, upper = density.bounds
    np.testing.assert_array_equal(lower, [[1.0, 0.0], [-1.0, 0.0]])
    np.testing.assert_array_equal(upper, [[1.0, 0.0], [1.0, 1.0]])

    # For n = 2, L[2, 1] = tanh(y) is the correlation r, and the density of y is
    # eta * log(1 - r^2) - log(2^(2 eta - 1) B(eta, eta)).
    y = jnp.linspace(-2.5, 2.5, 11)
    r = np.tanh(np.asarray(y))
    log_beta = 2 * math.lgamma(eta) - math.lgamma(2 * eta)
    closed = eta * np.log1p(-(r**2)) - (2 * eta - 1) * math.log(2) - log_beta
    values = jax.vmap(density)(y[:, None])
    np.testing.assert_allclose(values, closed, rtol=1e-12)
    np.testing.assert_allclose(
        values, jax.vmap(lambda u: stan_unconstrained(u, 2, eta))(y[:, None])
    )
    np.testing.assert_allclose(
        jax.vmap(density.grad)(y[:, None])[:, 0], -2 * eta * r, rtol=1e-12
    )
    grid = jnp.linspace(-12.0, 12.0, 8001)
    total = np.trapezoid(np.exp(np.asarray(jax.vmap(density)(grid[:, None]))), grid)
    assert total == pytest.approx(1.0, abs=1e-9)

    factors = jax.vmap(density.constrain)(y[:, None])
    np.testing.assert_allclose(factors[:, 1, 0], r)
    np.testing.assert_allclose(jax.vmap(density.unconstrain)(factors)[:, 0], y)


def test_a_thirty_by_thirty_factor_in_a_record_matches_stan() -> None:
    # The kronecker_gp prior: 435 free coordinates beside a positive scale.
    n, eta = 30, 2.0
    query = flatppl(f"""
        p = elementof(cartprod(L = corrcholesky({n}), sigma = posreals))
        inputs = p
        outputs = logdensityof(LKJCholesky({n}, {eta}), p.L) + logdensityof(LogNormal(0.0, 1.0), p.sigma)
    """)
    density = query.logdensity(dtype="float64")
    free = n * (n - 1) // 2
    assert dict(density.shapes) == {"L": (n, n), "sigma": ()}
    assert density.flat.init.shape == (free + 1,)
    assert density.flat.names == (*(f"L[{k}]" for k in range(1, free + 1)), "sigma")
    lower, upper = density.bounds
    np.testing.assert_array_equal(upper["L"], np.tril(np.ones((n, n))))
    expected_lower = np.tril(np.full((n, n), -1.0), -1)
    expected_lower[0, 0] = 1.0
    np.testing.assert_array_equal(lower["L"], expected_lower)

    y = jnp.asarray(np.random.default_rng(0).normal(scale=0.7, size=free))
    log_sigma = jnp.float64(0.3)
    point = {"L": y, "sigma": log_sigma}
    sigma = math.exp(0.3)
    lognormal = -math.log(sigma) - 0.5 * math.log(2 * math.pi) - 0.5 * 0.3**2
    reference = stan_unconstrained(y, n, eta) + lognormal + 0.3

    factor, _ = stan_cholesky_corr_constrain(y, n)
    np.testing.assert_allclose(density.constrain(point)["L"], factor, atol=1e-14)
    np.testing.assert_allclose(density(point), reference, rtol=1e-12)
    gradient = density.grad(point)
    np.testing.assert_allclose(
        gradient["L"], jax.grad(stan_unconstrained)(y, n, eta), rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(gradient["sigma"], -0.3, rtol=1e-12)
    np.testing.assert_allclose(density.unconstrain(density.constrain(point))["L"], y)

    # The log density is the compiled transformed query's.
    transformed = query.compile_transformed(logdensity=True, dtype="float64")
    np.testing.assert_allclose(density(point), transformed.unconstrained(point))
    flat = density.flat
    vector = jnp.concatenate((y, jnp.array([log_sigma])))
    np.testing.assert_allclose(flat.fn(vector), reference, rtol=1e-12)


def test_a_corrcholesky_parameter_is_a_query_input_and_a_measure_support() -> None:
    model = flatppl("""
        L = elementof(corrcholesky(3))
        Omega = L * transpose(L)
        prior = LKJCholesky(3, 2.0)
    """)
    assert model.L.domain is not None
    assert dict(model.L.domain) == {"set": "corrcholesky", "n": 3}
    assert repr(model.parameters["L"]) == "<Parameter L in corrcholesky(3)>"

    query = model.query("Omega")
    assert "point = elementof(cartprod(L = corrcholesky(3)))" in query.source
    factor = np.array([[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [-0.3, 0.4, math.sqrt(0.75)]])
    np.testing.assert_allclose(
        query.compile(dtype="float64")({"L": factor}), factor @ factor.T
    )
    with pytest.raises(ValueError, match="flat layout"):
        model.query("Omega", layout="flat")

    density = model.logdensity("prior", dtype="float64")
    assert list(density.shapes.values()) == [(3, 3)]
    assert density.flat.init.shape == (3,)
    np.testing.assert_array_equal(density.constrain(density.init), np.eye(3))
    y = jnp.array([0.4, -0.2, 0.9])
    np.testing.assert_allclose(density(y), stan_unconstrained(y, 3, 2.0), rtol=1e-12)
    np.testing.assert_allclose(
        density.grad(y), jax.grad(stan_unconstrained)(y, 3, 2.0), rtol=1e-10
    )


def test_a_corrcholesky_extent_must_be_bound_before_a_query() -> None:
    model = flatppl("""
        n = external(posintegers)
        L = elementof(corrcholesky(n))
        Omega = L * transpose(L)
    """)
    assert model.L.domain is not None
    assert dict(model.L.domain) == {"set": "corrcholesky", "n": None}
    with pytest.raises(ValueError, match="extent that is not known yet"):
        model.query("Omega")
    query = model.set(n=2).query("Omega")
    assert "L = corrcholesky(2)" in query.source
