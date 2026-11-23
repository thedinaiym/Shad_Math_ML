""" 
P2. Matrix multiplication from scratch. Implement matmul(A, B) with three nested loops,
directly from the definition (AB)ij =
P k AikBkj . Raise a ValueError if the number of
columns of A is not equal to the number of rows of B.
• Compare matmul(A, B) with NumPy’s A @ B for random matrices of several shapes.
95
• Use your function to check numerically that (AB)C = A(BC) and (AB)
⊤ = B⊤A⊤
(Proposition 6.22), and find random 2 × 2 matrices with AB ̸= BA.
• Measure the running times (with time.perf_counter()) of matmul and of @ for random
n × n matrices with n = 50, 100, 200. By roughly what factor does the running time
of matmul grow when n doubP2. Matrix multiplication from scratch. Implement matmul(A, B) with three nested loops,
directly from the definition (AB)ij =
P k AikBkj . Raise a ValueError if the number of
columns of A is not equal to the number of rows of B.
• Compare matmul(A, B) with NumPy’s A @ B for random matrices of several shapes.
95
• Use your function to check numerically that (AB)C = A(BC) and (AB)
⊤ = B⊤A⊤
(Proposition 6.22), and find random 2 × 2 matrices with AB ̸= BA.
• Measure the running times (with time.perf_counter()) of matmul and of @ for random
n × n matrices with n = 50, 100, 200. By roughly what factor does the running time
of matmul grow when n doubles? Explain this by counting the multiplications that are
performed. (NumPy is much faster because it runs highly optimized compiled code.)
P3. Homework. Row reduction. Implement a function that computes the reduced row echelonles? Explain this by counting the multiplications that are
performed. (NumPy is much faster because it runs highly optimized compiled code.)
"""
import numpy as np
rng = np.random.default_rng(0)  
import time

def matmul(A, B):
    """Return the product AB computed with three nested loops."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    m, p = A.shape
    p2, n = B.shape
    if p != p2:
        raise ValueError(f"cannot multiply {A.shape} by {B.shape}: "
                         f"A has {p} columns but B has {p2} rows")
    C = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            s = 0.0
            for k in range(p):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C


if __name__ == "__main__":
    shapes = [(1, 1, 1), (2, 3, 4), (5, 1, 3), (4, 4, 4), (7, 3, 2), (3, 6, 1)]
    for m, p, n in shapes:
        A = rng.normal(size=(m, p))
        B = rng.normal(size=(p, n))
        assert np.allclose(matmul(A, B), A @ B)
        print(f"({m}x{p}) @ ({p}x{n}): matmul agrees with @")

    try:
        matmul(np.ones((2, 3)), np.ones((2, 3)))
    except ValueError as err:
        print("ValueError raised as expected:", err)

    # 2. (AB)C = A(BC) and (AB)^T = B^T A^T (Proposition 6.22)
    A = rng.normal(size=(3, 4))
    B = rng.normal(size=(4, 5))
    C = rng.normal(size=(5, 2))
    assert np.allclose(matmul(matmul(A, B), C), matmul(A, matmul(B, C)))
    assert np.allclose(matmul(A, B).T, matmul(B.T, A.T))
    print("\n(AB)C == A(BC) and (AB)^T == B^T A^T: OK")

    # random 2x2 matrices with AB != BA
    A = rng.normal(size=(2, 2))
    B = rng.normal(size=(2, 2))
    print("\nA =\n", A, "\nB =\n", B)
    print("AB =\n", matmul(A, B), "\nBA =\n", matmul(B, A))
    print("AB == BA ?", np.allclose(matmul(A, B), matmul(B, A)))

    # 3. Running times
    print("\n   n   matmul [s]     @ [s]   factor vs n/2")
    previous = None
    for n in [50, 100, 200]:
        A = rng.normal(size=(n, n))
        B = rng.normal(size=(n, n))
        t0 = time.perf_counter()
        matmul(A, B)
        t_loop = time.perf_counter() - t0
        t0 = time.perf_counter()
        A @ B
        t_np = time.perf_counter() - t0
        factor = "" if previous is None else f"{t_loop / previous:8.1f}"
        print(f"{n:4d} {t_loop:12.4f} {t_np:10.6f} {factor}")
        previous = t_loop

# When n doubles, the running time of matmul grows by a factor of about 8.
#
# Каждый из n^2 элементов произведения — это сумма n произведений A_ik B_kj.
# Значит, всего выполняется n * n * n = n^3 умножений и столько же сложений.
# При n -> 2n это (2n)^3 = 8 n^3, то есть в 8 раз больше работы, и время
# растёт примерно в 8 раз. На малых n измерения чуть шумнее.
# NumPy (@) делает те же n^3 операций, но в оптимизированном скомпилированном
# коде (BLAS), поэтому он быстрее в тысячи раз.
