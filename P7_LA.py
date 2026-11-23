"""  
P7. Special matrices.
97
• Write functions is_symmetric(A, tol=1e-10) and is_orthogonal(Q, tol=1e-10).
• For a random 3 × 5 matrix A and a random 5 × 3 matrix B, check that A.T @ A is
symmetric, that tr(AB) = tr(BA) (even though AB and BA have different sizes!), and
that tr(A⊤A) equals np.linalg.norm(A, "fro")**2.
• Write a function that splits a square matrix into its symmetric and skew-symmetric parts
(Exercise 6.66), and test it.
• The command Q, R = np.linalg.qr(rng.normal(size=(4, 4))) produces a random
orthogonal matrix Q. Check that Q is orthogonal, that det(Q) = ±1 (Corollary 6.93),
and that Q preserves lengths: np.linalg.norm(Q @ x) equals np.linalg.norm(x) for
random vectors x, where the length of x is q x
2
1 + · · · + x
2
n
"""
import numpy as np
rng = np.random.default_rng(0)  

def is_symmetric(A, tol=1e-10):
    """A is square and A^T = A (up to tol)."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False
    return np.abs(A - A.T).max() < tol


def is_orthogonal(Q, tol=1e-10):
    """Q is square and Q^T Q = I (up to tol)."""
    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        return False
    return np.abs(Q.T @ Q - np.eye(Q.shape[0])).max() < tol


def trace(A):
    s = 0.0
    for i in range(A.shape[0]):
        s += A[i, i]
    return s


def sym_skew_parts(A):
    """Split a square matrix as A = S + K with S symmetric, K skew-symmetric
    (Exercise 6.66): S = (A + A^T)/2, K = (A - A^T)/2."""
    A = np.asarray(A, dtype=float)
    return (A + A.T) / 2, (A - A.T) / 2


def length(x):
    """sqrt(x_1^2 + ... + x_n^2)."""
    return np.sqrt(np.sum(x * x))


if __name__ == "__main__":
    A = rng.normal(size=(3, 5))
    B = rng.normal(size=(5, 3))

    print("A.T @ A symmetric:", is_symmetric(A.T @ A))
    print("A itself symmetric:", is_symmetric(A))
    assert is_symmetric(A.T @ A)

    print(f"tr(AB) = {trace(A @ B):.10f}  (AB is {(A @ B).shape})")
    print(f"tr(BA) = {trace(B @ A):.10f}  (BA is {(B @ A).shape})")
    assert np.isclose(trace(A @ B), trace(B @ A))

    print(f"tr(A^T A) = {trace(A.T @ A):.10f},  ||A||_F^2 = "
          f"{np.linalg.norm(A, 'fro') ** 2:.10f}")
    assert np.isclose(trace(A.T @ A), np.linalg.norm(A, "fro") ** 2)

    # symmetric + skew-symmetric parts
    M = rng.normal(size=(4, 4))
    S, K = sym_skew_parts(M)
    assert is_symmetric(S)
    assert np.allclose(K.T, -K)                    # K is skew-symmetric
    assert np.allclose(S + K, M)
    print("\nM = S + K with S symmetric, K skew-symmetric: OK")
    print("diagonal of K:", np.diag(K), "(a skew-symmetric matrix has zero diagonal)")

    # orthogonal matrices
    Q, R = np.linalg.qr(rng.normal(size=(4, 4)))
    print("\nQ orthogonal:", is_orthogonal(Q), "   M orthogonal:", is_orthogonal(M))
    assert is_orthogonal(Q) and not is_orthogonal(M)
    dQ = np.linalg.det(Q)
    print(f"det(Q) = {dQ:.12f}")
    assert np.isclose(abs(dQ), 1)                  # Corollary 6.93

    for _ in range(5):
        x = rng.normal(size=4)
        print(f"|x| = {length(x):.12f}   |Qx| = {length(Q @ x):.12f}   "
              f"np.linalg.norm(Q @ x) = {np.linalg.norm(Q @ x):.12f}")
        assert np.isclose(length(Q @ x), length(x))
    print("Q preserves lengths: OK")

# Почему всё сходится:
# * (A^T A)^T = A^T (A^T)^T = A^T A, значит A^T A симметрична.
# * tr(AB) = sum_i sum_k A_ik B_ki = sum_k sum_i B_ki A_ik = tr(BA), хотя AB
#   размера 3x3, а BA размера 5x5.
# * tr(A^T A) = sum_j (A^T A)_jj = sum_j sum_i A_ij^2 = ||A||_F^2.
# * |Qx|^2 = (Qx)^T (Qx) = x^T Q^T Q x = x^T x = |x|^2.
# * 1 = det(I) = det(Q^T Q) = det(Q)^2, поэтому det Q = ±1.
