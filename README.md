# Gram-Schmidt Orthogonalization Process

This repository provides a small Python implementation of Gram-Schmidt vector orthogonalization with:

- Classical Gram-Schmidt (`method="classical"`)
- Modified Gram-Schmidt (`method="modified"`, default for better numerical stability)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
import numpy as np
from gs import gram_schmidt

vectors = [
    np.array([1.0, 1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 1.0, 0.0]),
    np.array([0.0, 0.0, 1.0, 1.0]),
]

Q = gram_schmidt(vectors, method="modified")
print(Q.shape)  # (3, 4)
```

`gram_schmidt(...)` returns a NumPy array of shape `(k, n)` where each row is an orthonormal basis vector.

## API

- `normalize(v, tol=1e-12)`: normalizes one vector; raises for zero/near-zero norm.
- `gram_schmidt(vectors, method="modified", tol=1e-12)`: computes orthonormal rows.
- `gs(...)`: backward-compatible alias to `gram_schmidt(...)`.

## Mathematical note

Given input vectors \(u_1, \dots, u_k\), Gram-Schmidt constructs orthonormal vectors \(q_1, \dots, q_k\) by removing projections onto previously computed vectors and then normalizing.

Modified Gram-Schmidt applies projections iteratively to the residual vector and is usually more numerically stable than the classical form.

## Validation and limitations

The implementation validates:

- non-empty input
- all vectors are 1-D
- all vectors have equal dimension
- no zero/near-zero intermediate vectors (linearly dependent or degenerate input)

For linearly dependent vectors, a `ValueError` is raised.
