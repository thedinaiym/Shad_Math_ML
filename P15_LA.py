"""
P15. Homework. Tensors with NumPy.
• Create the tensor
T = np.arange(24).reshape(2, 3, 4)
and check that T[i, j, k] equals T.reshape(-1)[12*i + 4*j + k] for all indices, as
in Example 11.10. Compute the unfoldings T.reshape(2, 12) and T.transpose(1, 0,
2).reshape(3, 8). What are their ranks?
• Compute all the operations of Table 1 with np.einsum for random arrays, and compare
with @, np.trace, np.outer and .T. For a batch of 100 random 4 × 4 matrices, stored
as an array of shape 100 × 4 × 4, compute all their traces and all the products AbBb with
a single call of np.einsum each, and compare with a Python loop.
• For random matrices A (3 × 4), B (2 × 5) and X (4 × 5), check that (A @ X @
B.T).reshape(-1) equals np.kron(A, B) @ X.reshape(-1) (Remark 11.17). How
many numbers does np.kron(A, B) store for A, B ∈ R
100×100, and why is it much
cheaper to compute AXB⊤ directly?
100
• Write a function that returns the outer product u ⊗ v ⊗ w of three vectors:
def rank_one(u, v, w):
return np.einsum("i,j,k->ijk", u, v, w)
For random vectors u, v and w, check that the slices T[:, :, k] of T = rank_one(u,
v, w) are multiples of the matrix uv⊤ (Example 11.20).
"""

import numpy as np
rng = np.random.default_rng(0) 
import time

def rank_one(u, v, w):
    """The outer product u ⊗ v ⊗ w of three vectors."""
    return np.einsum("i,j,k->ijk", u, v, w)


if __name__ == "__main__":
    T = np.arange(24).reshape(2, 3, 4)
    flat = T.reshape(-1)
    for i in range(2):
        for j in range(3):
            for k in range(4):
                assert T[i, j, k] == flat[12 * i + 4 * j + k]
    print("T[i, j, k] == T.reshape(-1)[12*i + 4*j + k] for all indices: OK")

    U1 = T.reshape(2, 12)
    U2 = T.transpose(1, 0, 2).reshape(3, 8)
    print("\nT.reshape(2, 12) =\n", U1)
    print("T.transpose(1, 0, 2).reshape(3, 8) =\n", U2)
    print("ranks:", np.linalg.matrix_rank(U1), np.linalg.matrix_rank(U2))
    assert np.linalg.matrix_rank(U1) == 2 and np.linalg.matrix_rank(U2) == 2

    # 2. Operations with einsum vs @, np.trace, np.outer, .T
    x = rng.normal(size=4)
    y = rng.normal(size=4)
    A = rng.normal(size=(4, 4))
    B = rng.normal(size=(4, 3))
    checks = {
        "inner product  x·y        'i,i->'":       (np.einsum("i,i->", x, y), x @ y),
        "outer product  x y^T      'i,j->ij'":     (np.einsum("i,j->ij", x, y), np.outer(x, y)),
        "matrix-vector  A x        'ij,j->i'":     (np.einsum("ij,j->i", A, x), A @ x),
        "matrix product A B        'ij,jk->ik'":   (np.einsum("ij,jk->ik", A, B), A @ B),
        "trace          tr A       'ii->'":        (np.einsum("ii->", A), np.trace(A)),
        "transpose      A^T        'ij->ji'":      (np.einsum("ij->ji", A), A.T),
        "diagonal       diag A     'ii->i'":       (np.einsum("ii->i", A), np.diag(A)),
        "bilinear form  x^T A y    'i,ij,j->'":    (np.einsum("i,ij,j->", x, A, y), x @ A @ y),
        "Hadamard       A * A      'ij,ij->ij'":   (np.einsum("ij,ij->ij", A, A), A * A),
        "sum of entries            'ij->'":        (np.einsum("ij->", A), A.sum()),
    }
    print()
    for name, (mine, ref) in checks.items():
        assert np.allclose(mine, ref)
        print(f"{name}: OK")

    # batch of 100 random 4x4 matrices
    As = rng.normal(size=(100, 4, 4))
    Bs = rng.normal(size=(100, 4, 4))

    t0 = time.perf_counter()
    traces = np.einsum("bii->b", As)
    t_einsum_tr = time.perf_counter() - t0
    t0 = time.perf_counter()
    traces_loop = np.array([np.trace(As[b]) for b in range(100)])
    t_loop_tr = time.perf_counter() - t0
    assert np.allclose(traces, traces_loop)

    t0 = time.perf_counter()
    prods = np.einsum("bij,bjk->bik", As, Bs)
    t_einsum_pr = time.perf_counter() - t0
    t0 = time.perf_counter()
    prods_loop = np.array([As[b] @ Bs[b] for b in range(100)])
    t_loop_pr = time.perf_counter() - t0
    assert np.allclose(prods, prods_loop)
    print(f"\nbatch traces  : einsum {t_einsum_tr * 1e6:7.1f} us, loop {t_loop_tr * 1e6:7.1f} us")
    print(f"batch products: einsum {t_einsum_pr * 1e6:7.1f} us, loop {t_loop_pr * 1e6:7.1f} us")

    # 3. Kronecker product (Remark 11.17)
    A = rng.normal(size=(3, 4))
    B = rng.normal(size=(2, 5))
    X = rng.normal(size=(4, 5))
    lhs = (A @ X @ B.T).reshape(-1)
    rhs = np.kron(A, B) @ X.reshape(-1)
    print("\n(A X B^T).reshape(-1) == kron(A, B) @ X.reshape(-1):", np.allclose(lhs, rhs))
    assert np.allclose(lhs, rhs)

    n = 100
    print(f"for A, B in R^(100x100): kron(A, B) has {n**2} x {n**2} = {n**4:.0e} entries "
          f"= {n**4 * 8 / 1e9:.1f} GB in float64")
    print(f"multiplications: kron(A, B) @ vec(X) ~ {n**4:.0e},  A @ X @ B.T ~ 2*{n}^3 = {2 * n**3:.0e}")

    # 4. rank-one tensors
    u = rng.normal(size=3)
    v = rng.normal(size=4)
    w = rng.normal(size=5)
    T = rank_one(u, v, w)
    print("\nrank_one(u, v, w).shape =", T.shape)
    uvT = np.outer(u, v)
    for k in range(len(w)):
        assert np.allclose(T[:, :, k], w[k] * uvT)      # slice = w_k * u v^T
    print("every slice T[:, :, k] equals w_k * u v^T: OK")

# Ранги развёрток.
# T.reshape(2, 12) = [[0, ..., 11], [12, ..., 23]]. Строки не пропорциональны,
# поэтому ранг 2 (больше 2 быть не может, строк две).
# В T.transpose(1, 0, 2).reshape(3, 8) строка j равна
# (4j, ..., 4j+3, 12+4j, ..., 12+4j+3) = b + 4j * (1, ..., 1),
# где b = (0, 1, 2, 3, 12, 13, 14, 15). Все строки лежат в span{b, 1},
# поэтому ранг тоже 2, а не 3.
#
# Kronecker. kron(A, B) для A, B размера 100x100 — матрица 10^4 x 10^4,
# 10^8 чисел (~0.8 ГБ). Умножение её на vec(X) стоит ~10^8 умножений.
# Прямое вычисление A X B^T — два произведения 100x100 матриц,
# ~2 * 10^6 умножений, то есть в 50 раз меньше операций, без огромной
# промежуточной матрицы, и с быстрым BLAS.
#
# einsum на батче делает всё одним векторизованным вызовом без цикла Python.
# Цикл тратит время на 100 отдельных вызовов (накладные расходы интерпретатора).
#
# Срезы rank-one тензора: T[i, j, k] = u_i v_j w_k, поэтому при фиксированном k
# T[:, :, k] = w_k * (u_i v_j)_{ij} = w_k * u v^T. Все срезы — кратные одной
# матрицы ранга 1.