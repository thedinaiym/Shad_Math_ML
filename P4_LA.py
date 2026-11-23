"""
    P4. The inverse with the Gauss–Jordan method. Implement inverse(A): row reduce the
n × 2n matrix (A | In) with your rref (use np.hstack and np.eye) and return the right
block. If the left block is not In, raise np.linalg.LinAlgError("matrix is singular")
(Theorem 6.49).
• Check your function on the examples of Subsection 6.3: the inverse of example f-s
• For a random 10 × 10 matrix A, check that A @ inverse(A) and inverse(A) @ A are
both close to I10, and that inverse(A) is close to np.linalg.inv(A).
    """
    
import numpy as np
rng = np.random.default_rng(0)  

def rref(A, tol=1e-10):
    """Return (R, pivots): the RREF of A and the list of pivot columns (from P3)."""
    R = np.array(A, dtype=float)
    m, n = R.shape
    pivots = []
    r = 0
    for c in range(n):
        if r == m:
            break
        p = r + np.argmax(np.abs(R[r:, c]))
        if abs(R[p, c]) < tol:
            R[r:, c] = 0.0
            continue
        if p != r:
            R[[r, p]] = R[[p, r]]
        R[r] = R[r] / R[r, c]
        for i in range(m):
            if i != r:
                R[i] = R[i] - R[i, c] * R[r]
        pivots.append(c)
        r += 1
    return R, pivots


def inverse(A, tol=1e-10):
    """Row reduce (A | I_n) and return the right block."""
    A = np.array(A, dtype=float)
    n, n2 = A.shape
    if n != n2:
        raise ValueError("only square matrices can be inverted")
    R, pivots = rref(np.hstack([A, np.eye(n)]), tol)
    # the left block is I_n  <=>  every one of the first n columns has a pivot
    if [p for p in pivots if p < n] != list(range(n)):
        raise np.linalg.LinAlgError("matrix is singular")
    return R[:, n:]


if __name__ == "__main__":
    # Examples of Subsection 6.3
    A = np.array([[2, 3],
                  [4, -1]])
    Ainv = inverse(A)
    print("inverse of [[2, 3], [4, -1]]:\n", Ainv)
    assert np.allclose(Ainv, np.array([[1, 3], [4, -2]]) / 14)

    B = np.array([[1, 1, 0],
                  [0, 1, 1],
                  [1, 1, 1]])
    Binv = inverse(B)
    print("\ninverse of [[1, 1, 0], [0, 1, 1], [1, 1, 1]]:\n", Binv)
    assert np.allclose(Binv, [[0, -1, 1], [1, 1, -1], [-1, 0, 1]])

    try:
        inverse([[1, 2], [2, 4]])
    except np.linalg.LinAlgError as err:
        print("\n[[1, 2], [2, 4]]:", err)

    # random 10x10 matrix
    A = rng.normal(size=(10, 10))
    Ainv = inverse(A)
    assert np.allclose(A @ Ainv, np.eye(10))
    assert np.allclose(Ainv @ A, np.eye(10))
    assert np.allclose(Ainv, np.linalg.inv(A))
    print("\nrandom 10x10: A @ inverse(A) ~ I, inverse(A) @ A ~ I, "
          "inverse(A) ~ np.linalg.inv(A)")
    print("max |A @ inverse(A) - I| =", np.abs(A @ Ainv - np.eye(10)).max())