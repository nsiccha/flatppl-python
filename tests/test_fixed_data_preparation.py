"""Summaries of fixed data are prepared once per compile, not on every call.

A Gaussian regression on loaded rows, whether they declare bounded entries or
reals, reduces to data-only summaries: a bounded QR factorization of its design.
The compiler moves their setup, with its loops and its runtime checks, into a
preparation program. A
compiled function runs it once, checks its status, and passes its outputs after
the declared inputs of every call.
"""

import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flatppl import Module, flatppl, load
from flatppl._jax import CompiledFunction, _prepare  # pyright: ignore[reportPrivateUsage]
from flatppl._schema import Export, Schema

ROWS = 40

DATA = f"""
X = load_data("X.json", cartpow(interval(-100.0, 100.0), [{ROWS}, 3]))
y = load_data("y.json", cartpow(interval(-100.0, 100.0), {ROWS}))
"""

# PosteriorDB declares loaded regression data on reals; the compiler proves them
# finite from their loaded values.
REALS = f"""
X = load_data("X.json", cartpow(reals, [{ROWS}, 3]))
y = load_data("y.json", cartpow(reals, {ROWS}))
"""

REGRESSION = """
beta = elementof(cartpow(reals, 3))
sigma = elementof(posreals)
inputs = (beta, sigma)
outputs = logdensityof(Normal.(X * beta, sigma), y)
"""

RECORD = """
p = elementof(cartprod(beta = cartpow(reals, 3), sigma = posreals))
inputs = p
outputs = logdensityof(Normal.(X * p.beta, p.sigma), y)
"""


@pytest.fixture(autouse=True)
def x64() -> Iterator[None]:
    with jax.enable_x64(True):
        yield


def data() -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(ROWS, dtype=np.float64)
    design = np.column_stack(
        (np.ones(ROWS), 3 * np.sin(0.37 * t), 2 * np.cos(0.11 * t) + 0.1 * t)
    )
    response = 0.5 + 1.5 * design[:, 1] - 0.7 * design[:, 2] + 0.3 * np.sin(1.3 * t)
    return design, response


def closed_form(beta: np.ndarray, sigma: float) -> tuple[float, np.ndarray, float]:
    """The log density and its gradients in beta and sigma."""
    design, response = data()
    residual = response - design @ beta
    value = np.sum(
        -0.5 * math.log(2 * math.pi) - math.log(sigma) - residual**2 / (2 * sigma**2)
    )
    return (
        float(value),
        design.T @ residual / sigma**2,
        float(-ROWS / sigma + np.sum(residual**2) / sigma**3),
    )


def loaded(
    directory: Path, query: str, design: np.ndarray | None = None, source: str = DATA
) -> Module:
    """``query`` over the rows of :func:`data`, loaded by ``source``."""
    rows, response = data()
    if design is not None:
        rows = design
    (directory / "X.json").write_text(json.dumps(rows.tolist()))
    (directory / "y.json").write_text(json.dumps(response.tolist()))
    path = directory / "query.flatppl"
    path.write_text(source + query)
    return load(path)


def unprepared(module: Module) -> CompiledFunction:
    """The same reduced query, compiled without a preparation."""
    native: Any = module._native  # pyright: ignore[reportPrivateUsage]

    def export(batch: Any = None) -> Any:
        return json.loads(
            native.export(
                "float64",
                True,
                None,
                batch,
                "cpu",
                sufficient_statistics=True,
                prepare_fixed_data=False,
            )
        )

    return CompiledFunction(
        export(),
        autodiff=True,
        export_batch=lambda shape, axes: export((shape, axes)),
    )


POINTS = [
    (np.array([0.5, 1.5, -0.7]), 0.3),
    (np.array([0.0, 1.0, 0.0]), 1.7),
    (np.array([-1.2, 0.4, 0.9]), 0.05),
]


def test_a_fixed_data_regression_scores_with_its_prepared_summaries(
    tmp_path: Path,
) -> None:
    module = loaded(tmp_path, REGRESSION)
    function = module.compile(dtype="float64")
    reference = unprepared(module)
    ordinary = module.compile(dtype="float64", sufficient_statistics=False)
    # The QR setup and its loops run once; each call scores without a loop.
    prepared = function._prepared  # pyright: ignore[reportPrivateUsage]
    # The preparation's device status is checked and dropped.
    assert prepared
    assert all(jnp.issubdtype(leaf.dtype, jnp.floating) for leaf in prepared)
    assert "stablehlo.while" not in function.stablehlo
    assert "stablehlo.while" in reference.stablehlo
    assert "preparation" not in function.schema
    assert [item["name"] for item in function.schema["inputs"]] == ["beta", "sigma"]

    gradient = jax.grad(function, argnums=(0, 1))
    for beta, sigma in POINTS:
        value, beta_gradient, sigma_gradient = closed_form(beta, sigma)
        assert float(function(beta, sigma)) == pytest.approx(value, rel=1e-12)
        d_beta, d_sigma = gradient(beta, sigma)
        np.testing.assert_allclose(d_beta, beta_gradient, rtol=1e-11)
        assert float(d_sigma) == pytest.approx(sigma_gradient, rel=1e-11)
        for other in (reference, ordinary):
            assert float(function(beta, sigma)) == pytest.approx(
                float(other(beta, sigma)), rel=1e-13
            )
            other_beta, other_sigma = jax.grad(other, argnums=(0, 1))(beta, sigma)
            np.testing.assert_allclose(d_beta, other_beta, rtol=1e-12)
            assert float(d_sigma) == pytest.approx(float(other_sigma), rel=1e-12)

    # Batched calls share the prepared summaries across cells.
    betas = jnp.stack([beta for beta, _ in POINTS])
    sigmas = jnp.array([sigma for _, sigma in POINTS])
    np.testing.assert_allclose(
        jax.vmap(function)(betas, sigmas),
        [closed_form(beta, sigma)[0] for beta, sigma in POINTS],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        jax.vmap(function, in_axes=(0, None))(betas, 0.3),
        [closed_form(beta, 0.3)[0] for beta, _ in POINTS],
        rtol=1e-12,
    )


def test_transformed_log_densities_use_the_prepared_program(tmp_path: Path) -> None:
    query = loaded(tmp_path, RECORD)
    assert "stablehlo.while" in unprepared(query).stablehlo
    transformed = query.compile_transformed(logdensity=True, dtype="float64")
    assert "stablehlo.while" not in transformed.unconstrained.stablehlo
    value_and_grad = jax.jit(jax.value_and_grad(transformed.unconstrained))
    for beta, sigma in POINTS:
        u = {"beta": jnp.asarray(beta), "sigma": jnp.log(sigma)}
        value, gradient = value_and_grad(u)
        expected, beta_gradient, sigma_gradient = closed_form(beta, sigma)
        # sigma = exp(u) adds log sigma and scales its derivative by sigma.
        assert float(value) == pytest.approx(expected + math.log(sigma), rel=1e-12)
        np.testing.assert_allclose(gradient["beta"], beta_gradient, rtol=1e-11)
        assert float(gradient["sigma"]) == pytest.approx(
            sigma * sigma_gradient + 1, rel=1e-11
        )


def test_regression_data_on_reals_are_prepared(tmp_path: Path) -> None:
    # Loaded or bound with Module.set, data on reals are proven finite from
    # their values and reduce like bounded rows.
    design, response = data()
    bound = flatppl(
        f"""
        X = external(cartpow(reals, [{ROWS}, 3]))
        y = external(cartpow(reals, {ROWS}))
        """
        + REGRESSION
    ).set(X=design, y=response)
    for module in (loaded(tmp_path, REGRESSION, source=REALS), bound):
        function = module.compile(dtype="float64")
        reference = unprepared(module)
        assert function._prepared  # pyright: ignore[reportPrivateUsage]
        assert "stablehlo.while" not in function.stablehlo
        assert "stablehlo.while" in reference.stablehlo
        gradient = jax.grad(function, argnums=(0, 1))
        for beta, sigma in POINTS:
            value, beta_gradient, sigma_gradient = closed_form(beta, sigma)
            assert float(function(beta, sigma)) == pytest.approx(value, rel=1e-12)
            assert float(function(beta, sigma)) == pytest.approx(
                float(reference(beta, sigma)), rel=1e-13
            )
            d_beta, d_sigma = gradient(beta, sigma)
            np.testing.assert_allclose(d_beta, beta_gradient, rtol=1e-11)
            assert float(d_sigma) == pytest.approx(sigma_gradient, rel=1e-11)

    transformed = loaded(tmp_path, RECORD, source=REALS).compile_transformed(
        logdensity=True, dtype="float64"
    )
    assert transformed.unconstrained._prepared  # pyright: ignore[reportPrivateUsage]
    assert "stablehlo.while" not in transformed.unconstrained.stablehlo
    value_and_grad = jax.jit(jax.value_and_grad(transformed.unconstrained))
    for beta, sigma in POINTS:
        u = {"beta": jnp.asarray(beta), "sigma": jnp.log(sigma)}
        value, gradient = value_and_grad(u)
        expected, beta_gradient, sigma_gradient = closed_form(beta, sigma)
        assert float(value) == pytest.approx(expected + math.log(sigma), rel=1e-12)
        np.testing.assert_allclose(gradient["beta"], beta_gradient, rtol=1e-11)
        assert float(gradient["sigma"]) == pytest.approx(
            sigma * sigma_gradient + 1, rel=1e-11
        )


def test_queries_without_separable_summaries_have_no_preparation(
    tmp_path: Path,
) -> None:
    normal = flatppl("""
        x = elementof(reals)
        inputs = x
        outputs = logdensityof(Normal(0.0, 1.0), x)
    """).compile(dtype="float64")
    assert normal._prepared == ()  # pyright: ignore[reportPrivateUsage]
    assert float(normal(0.5)) == pytest.approx(-0.125 - 0.5 * math.log(2 * math.pi))
    ordinary = loaded(tmp_path, REGRESSION).compile(
        dtype="float64", sufficient_statistics=False
    )
    assert ordinary._prepared == ()  # pyright: ignore[reportPrivateUsage]


def preparation(status: int) -> Export:
    """A preparation with two summaries and a device status between them."""

    def program() -> tuple[jax.Array, ...]:
        return jnp.arange(3.0), jnp.asarray(status, jnp.int64), jnp.ones((2, 2))

    lowered: Any = jax.jit(program).lower()  # pyright: ignore[reportUnknownMemberType]
    items: list[Schema] = [
        {
            "kind": "tensor",
            "index": index,
            "dtype": "float64",
            "shape": shape,
            "value_type": "reals",
        }
        for index, shape in ((0, [3]), (2, [2, 2]))
    ]
    return {
        "stablehlo": str(lowered.compiler_ir(dialect="stablehlo")),
        "entry_point": "main",
        "inputs": [],
        "output": {"kind": "tuple", "items": items},
        "multiple_outputs": True,
        "output_names": [None, None],
        "runtime_status": {"index": 1, "dtype": "int64", "shape": []},
    }


def test_preparations_return_their_summaries_without_the_status() -> None:
    summaries = _prepare(preparation(0))
    assert len(summaries) == 2
    np.testing.assert_array_equal(summaries[0], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(summaries[1], np.ones((2, 2)))


def test_fixed_data_that_fail_a_preparation_check_raise_when_compiled() -> None:
    # The device status of a preparation sees only fixed data, so a failure
    # would repeat on every call; it raises once, while compiling.
    with pytest.raises(
        ValueError,
        match="fixed data fail a check of the query: FlatPPL linear solve "
        "requires a nonsingular matrix",
    ):
        _prepare(preparation(4))
    with pytest.raises(ValueError, match="Unknown FlatPPL runtime error"):
        _prepare(preparation(99))
