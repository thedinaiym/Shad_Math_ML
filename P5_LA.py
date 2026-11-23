"""
P5. Determinants in two ways. Implement det_cofactor(A), which computes the determinant recursively by cofactor expansion along the first row (Subsection 6.5), and
det_rowreduce(A), which brings A into upper triangular form by row operations, keeps track
of the sign changes caused by row swaps, and multiplies the diagonal entries (Theorem 6.84
and Corollary 6.86). Hint: np.delete(A[1:], j, axis=1) is the matrix obtained from A by
deleting its first row and the column with index j.
• Both functions should return −3 for


1 2 3
4 5 6
7 8 10

, and 0 (up to rounding) for the matrix
F of Example 6.31. Compare both functions with np.linalg.det on random matrices.
• Check the product rule det(AB) = det(A) det(B) and the rule det(A⊤) = det(A) on
random 5 × 5 matrices.
• Measure the running time of det_cofactor for random n×n matrices with n = 2, 3, . . . , 8.
By roughly what factor does it grow from n − 1 to n, and why? Estimate how long
det_cofactor would need for n = 20, and compare with the running time of
det_rowreduce for n = 200.
    """

import numpy as np
rng = np.random.default_rng(0)  
import math
import time



def det_cofactor(A):
    """Determinant by cofactor expansion along the first row (Subsection 6.5)."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if n == 1:
        return A[0, 0]
    total = 0.0
    for j in range(n):
        minor = np.delete(A[1:], j, axis=1)   # delete row 0 and column j
        total += (-1) ** j * A[0, j] * det_cofactor(minor)
    return total


def det_rowreduce(A):
    """Determinant via row reduction to upper triangular form
    (Theorem 6.84, Corollary 6.86).

    - adding a multiple of one row to another does not change det;
    - every row swap changes the sign;
    - det of a triangular matrix = product of its diagonal entries.
    """
    U = np.array(A, dtype=float)
    n = U.shape[0]
    sign = 1.0
    for c in range(n):
        p = c + np.argmax(np.abs(U[c:, c]))   # partial pivoting
        if U[p, c] == 0.0:
            # the whole column below the diagonal is zero, so det = 0.
            # (We test exact zero only to avoid dividing by zero: a tiny
            #  pivot simply gives a tiny determinant, which is correct.)
            return 0.0
        if p != c:
            U[[c, p]] = U[[p, c]]
            sign = -sign
        factors = U[c + 1:, c] / U[c, c]
        U[c + 1:, c:] -= factors[:, None] * U[c, c:]
    det = sign
    for i in range(n):
        det *= U[i, i]
    return det


if __name__ == "__main__":
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 10]])
    print("det_cofactor  =", det_cofactor(A))
    print("det_rowreduce =", det_rowreduce(A))
    assert abs(det_cofactor(A) + 3) < 1e-10 and abs(det_rowreduce(A) + 3) < 1e-10

    # matrix F of Example 6.31 (rows 1..9; it is singular)
    F = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    print("\nF: det_cofactor =", det_cofactor(F), " det_rowreduce =", det_rowreduce(F))
    assert abs(det_cofactor(F)) < 1e-10 and abs(det_rowreduce(F)) < 1e-10

    # compare with np.linalg.det on random matrices
    for n in range(1, 7):
        M = rng.normal(size=(n, n))
        d = np.linalg.det(M)
        assert np.isclose(det_cofactor(M), d) and np.isclose(det_rowreduce(M), d)
    print("\nrandom 1x1 .. 6x6: both functions agree with np.linalg.det")

    # det(AB) = det(A) det(B), det(A^T) = det(A)
    A = rng.normal(size=(5, 5))
    B = rng.normal(size=(5, 5))
    for det in (det_cofactor, det_rowreduce):
        assert np.isclose(det(A @ B), det(A) * det(B))
        assert np.isclose(det(A.T), det(A))
    print("det(AB) = det(A) det(B) and det(A^T) = det(A): OK")

    # running time of det_cofactor
    print("\n n   det_cofactor [s]   t(n)/t(n-1)")
    times = {}
    for n in range(2, 9):
        M = rng.normal(size=(n, n))
        t0 = time.perf_counter()
        det_cofactor(M)
        times[n] = time.perf_counter() - t0
        ratio = "" if n == 2 else f"{times[n] / times[n - 1]:8.1f}"
        print(f"{n:2d} {times[n]:16.6f} {ratio}")

    est_20 = times[8] * math.factorial(20) / math.factorial(8)
    print(f"\nestimate for n = 20: {est_20:.2e} s = {est_20 / (3600 * 24 * 365):.2e} years")

    M = rng.normal(size=(200, 200))
    t0 = time.perf_counter()
    d = det_rowreduce(M)
    t_rr = time.perf_counter() - t0
    print(f"det_rowreduce for n = 200: {t_rr:.4f} s "
          f"(relative error vs NumPy {abs(d - np.linalg.det(M)) / abs(np.linalg.det(M)):.1e})")

# Время det_cofactor растёт от n-1 к n примерно в n раз, потому что общая
# сложность ~ n!.
# Разложение по первой строке вызывает det_cofactor n раз для миноров
# размера n-1, поэтому число операций C(n) = n * C(n-1) + O(n^2) ~ n!.
# (На маленьких n коэффициент шумный из-за накладных расходов Python.)
#
# Для n = 20 оценка t(8) * 20!/8! — порядка 10^12–10^13 секунд, то есть
# сотни тысяч лет. Точное число печатается выше.
# det_rowreduce выполняет ~n^3/3 операций (как метод Гаусса), и для n = 200
# это несколько миллисекунд.