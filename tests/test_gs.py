import numpy as np
import pytest

from gs import gram_schmidt


def test_returns_expected_shape_and_orthonormal_rows() -> None:
    vectors = [
        np.array([1.0, 1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 1.0]),
    ]
    q = gram_schmidt(vectors, method="modified")

    assert q.shape == (3, 4)
    gram = q @ q.T
    assert np.allclose(gram, np.eye(3), atol=1e-10)


def test_classical_and_modified_are_both_supported() -> None:
    vectors = [
        np.array([1.0, 2.0, 3.0]),
        np.array([0.0, 1.0, 1.0]),
        np.array([2.0, 0.0, 1.0]),
    ]
    q_classical = gram_schmidt(vectors, method="classical")
    q_modified = gram_schmidt(vectors, method="modified")

    assert np.allclose(q_classical @ q_classical.T, np.eye(3), atol=1e-10)
    assert np.allclose(q_modified @ q_modified.T, np.eye(3), atol=1e-10)


def test_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        gram_schmidt([])


def test_rejects_non_1d_vectors() -> None:
    vectors = [np.array([[1.0, 2.0], [3.0, 4.0]])]
    with pytest.raises(ValueError, match="must be 1-D"):
        gram_schmidt(vectors)


def test_rejects_mismatched_dimensions() -> None:
    vectors = [np.array([1.0, 0.0]), np.array([1.0, 0.0, 0.0])]
    with pytest.raises(ValueError, match="same dimension"):
        gram_schmidt(vectors)


def test_rejects_linearly_dependent_vectors() -> None:
    vectors = [np.array([1.0, 0.0]), np.array([2.0, 0.0])]
    with pytest.raises(ValueError, match="linearly dependent|near-zero|zero"):
        gram_schmidt(vectors)


def test_rejects_unknown_method() -> None:
    vectors = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    with pytest.raises(ValueError, match="method must be either"):
        gram_schmidt(vectors, method="invalid")  # type: ignore[arg-type]
