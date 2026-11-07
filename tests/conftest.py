"""Shared test execution settings."""

from collections.abc import Iterator

import pytest
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]


@pytest.fixture(scope="session", autouse=True)
def single_threaded_numeric_libraries() -> Iterator[None]:
    """Keep BLAS/OpenMP pools from multiplying the eight pytest workers.

    Returns
    -------
    Iterator[None]
        A session with one native numerical thread per worker. Libraries imported
        during collection are already loaded; their limits are restored afterward.
    """
    with threadpool_limits(limits=1):
        yield
