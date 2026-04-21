from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
from numpy.typing import NDArray

Method = Literal["classical", "modified"]


def normalize(v: NDArray[np.float64], *, tol: float = 1e-12) -> NDArray[np.float64]:
    """
    Normalize a 1-D vector.

    Args:
        v: Vector to normalize.
        tol: Minimum norm threshold. Defaults to ``1e-12``.

    Returns:
        Unit vector in the same direction as ``v``.

    Raises:
        ValueError: If the vector has near-zero norm.
    """
    norm = float(np.linalg.norm(v))
    if norm <= tol:
        raise ValueError(
            "Cannot normalize a zero (or near-zero) vector; input may be linearly dependent."
        )
    return v / norm


def _prepare_vectors(vectors: Iterable[NDArray[np.float64]]) -> list[NDArray[np.float64]]:
    prepared = [np.asarray(v, dtype=float) for v in vectors]
    if not prepared:
        raise ValueError("Input vectors cannot be empty.")

    dims = set()
    for i, vec in enumerate(prepared):
        if vec.ndim != 1:
            raise ValueError(f"Vector at index {i} must be 1-D, got shape {vec.shape}.")
        dims.add(vec.shape[0])
    if len(dims) != 1:
        raise ValueError("All vectors must have the same dimension.")
    return prepared


def gram_schmidt(
    vectors: Iterable[NDArray[np.float64]],
    *,
    method: Method = "modified",
    tol: float = 1e-12,
) -> NDArray[np.float64]:
    """
    Compute an orthonormal basis using Gram-Schmidt orthogonalization.

    Args:
        vectors: Iterable of 1-D vectors with equal dimension.
        method: Either ``"classical"`` or ``"modified"``. Defaults to ``"modified"``.
        tol: Minimum norm threshold to detect linear dependence. Defaults to ``1e-12``.

    Returns:
        A 2-D NumPy array of shape ``(k, n)`` where each row is an orthonormal vector.

    Raises:
        ValueError: For invalid inputs, unknown method, or linear dependence.
    """
    if method not in ("classical", "modified"):
        raise ValueError("method must be either 'classical' or 'modified'.")

    u = _prepare_vectors(vectors)
    orthonormal: list[NDArray[np.float64]] = []

    for i in range(len(u)):
        s = u[i].copy()
        for q in orthonormal:
            if method == "classical":
                s = s - np.dot(u[i], q) * q
            else:
                s = s - np.dot(s, q) * q
        orthonormal.append(normalize(s, tol=tol))

    return np.vstack(orthonormal)


def gs(
    vectors: Iterable[NDArray[np.float64]],
    *,
    method: Method = "modified",
    tol: float = 1e-12,
) -> NDArray[np.float64]:
    """Backward-compatible alias for :func:`gram_schmidt`."""
    return gram_schmidt(vectors, method=method, tol=tol)


def _demo() -> None:
    coeffs_1 = np.array([1, 1, 0, 0], dtype=float)
    coeffs_2 = np.array([0, 1, 1, 0], dtype=float)
    coeffs_3 = np.array([0, 0, 1, 1], dtype=float)

    basis = gram_schmidt([coeffs_1, coeffs_2, coeffs_3], method="modified")
    for v in basis:
        print(v)


if __name__ == "__main__":
    _demo()
