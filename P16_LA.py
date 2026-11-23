"""  P16. Homework. The LU decomposition and rank-one updates.
def lu(A):
# return L, U with A = L @ U (no row swaps)
...
def solve_lu(L, U, b):
# solve L U x = b by forward and back substitution
...
• Implement lu by the elimination of Theorem 6.106: for each column, store the multipliers in L and replace the rest of the matrix by its Schur complement. Check it on
Example 6.107, and check that L @ U equals A for random 6 × 6 matrices.
• Implement solve_lu, and compare it with np.linalg.solve for random systems. For
A =
 
10−17 1
1 1! and b = (1, 2)⊤, the solution is very close to (1, 1). What does your
solver return, and why? Swapping the two rows first (partial pivoting) fixes the problem.
• For a random 200 × 50 data matrix X, compute (X⊤X)
−1
. Then add 100 new random
rows one at a time, updating the inverse with the Sherman–Morrison formula of Theorem 6.108, and compare the result with (X⊤X)
−1
computed from scratch for the final
matrix. Compare the running times of the two approaches
""" 

import numpy as np
rng = np.random.default_rng(0) 
import time

def lu(A):
    # return L, U with A = L @ U (no row swaps)
    """Elimination of Theorem 6.106: for each column store the multipliers in L
    and replace the rest of the matrix by its Schur complement."""
    S = np.array(A, dtype=float)          # the current Schur complement
    n = S.shape[0]
    L = np.eye(n)
    U = np.zeros((n, n))
    for k in range(n):
        pivot = S[k, k]
        if pivot == 0.0:
            # exact zero only: we must not divide by zero
            raise ValueError(f"zero pivot in step {k}: A has no LU without row swaps")
        U[k, k:] = S[k, k:]                       # k-th row of U
        L[k + 1:, k] = S[k + 1:, k] / pivot       # multipliers
        # Schur complement: S22 - (S21 / s11) S12
        S[k + 1:, k + 1:] -= L[k + 1:, k][:, None] * U[k, k + 1:][None, :]
    return L, U


def solve_lu(L, U, b):
    # solve L U x = b by forward and back substitution
    n = L.shape[0]
    b = np.asarray(b, dtype=float)
    y = np.zeros(n)
    for i in range(n):                    # forward: L y = b
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):        # back: U x = y
        x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def inverse_lu(A):
    """A^{-1} column by column: A x_j = e_j, using one LU decomposition."""
    n = A.shape[0]
    L, U = lu(A)
    Ainv = np.zeros((n, n))
    I = np.eye(n)
    for j in range(n):
        Ainv[:, j] = solve_lu(L, U, I[:, j])
    return Ainv


def sherman_morrison(Minv, x):
    """Inverse of M + x x^T from Minv = M^{-1} (Theorem 6.108):
    (M + u v^T)^{-1} = M^{-1} - M^{-1} u v^T M^{-1} / (1 + v^T M^{-1} u)."""
    Mx = Minv @ x                         # M^{-1} x  (M symmetric: x^T M^{-1} = Mx^T)
    return Minv - np.outer(Mx, Mx) / (1.0 + x @ Mx)


if __name__ == "__main__":
    A = np.array([[2.0, 1.0, 1.0],
                  [4.0, -6.0, 0.0],
                  [-2.0, 7.0, 2.0]])
    L, U = lu(A)
    print("A =\n", A, "\nL =\n", L, "\nU =\n", U)
    assert np.allclose(L @ U, A)
    assert np.allclose(L, np.tril(L)) and np.allclose(U, np.triu(U))

    for _ in range(10):
        A = rng.normal(size=(6, 6))
        L, U = lu(A)
        assert np.allclose(L @ U, A)
    print("random 6x6: L @ U == A: OK")

    # 2. solve_lu vs np.linalg.solve
    for n in [3, 10, 50]:
        A = rng.normal(size=(n, n))
        b = rng.normal(size=n)
        x = solve_lu(*lu(A), b)
        assert np.allclose(x, np.linalg.solve(A, b))
    print("solve_lu agrees with np.linalg.solve for random systems")

    # tiny pivot
    A = np.array([[1e-17, 1.0],
                  [1.0, 1.0]])
    b = np.array([1.0, 2.0])
    L, U = lu(A)
    x_bad = solve_lu(L, U, b)
    print("\nA = [[1e-17, 1], [1, 1]], b = (1, 2)")
    print("L =\n", L, "\nU =\n", U)
    print("solve_lu without pivoting:", x_bad)
    print("np.linalg.solve          :", np.linalg.solve(A, b))
    # partial pivoting: swap the two rows (of A and of b) first
    L2, U2 = lu(A[[1, 0]])
    x_good = solve_lu(L2, U2, b[[1, 0]])
    print("solve_lu after row swap  :", x_good)
    assert np.allclose(x_good, [1, 1])

    # 3. Sherman-Morrison updates of (X^T X)^{-1}
    X = rng.normal(size=(200, 50))
    new_rows = rng.normal(size=(100, 50))

    # (a) start with (X^T X)^{-1} and add the rows one at a time
    t0 = time.perf_counter()
    Minv = inverse_lu(X.T @ X)
    t_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    for x in new_rows:
        Minv = sherman_morrison(Minv, x)
    t_sm = time.perf_counter() - t0

    # (b) from scratch for the final matrix
    X_final = np.vstack([X, new_rows])
    t0 = time.perf_counter()
    Minv_scratch = inverse_lu(X_final.T @ X_final)
    t_scratch = time.perf_counter() - t0

    rel = np.abs(Minv - Minv_scratch).max() / np.abs(Minv_scratch).max()
    print(f"\nSherman-Morrison after 100 rows vs from scratch: relative difference {rel:.1e}")
    assert np.allclose(Minv, Minv_scratch)
    assert np.allclose(Minv_scratch, np.linalg.inv(X_final.T @ X_final))

    # (c) from scratch after EVERY new row (what SM replaces)
    t0 = time.perf_counter()
    Xc = X.copy()
    for x in new_rows:
        Xc = np.vstack([Xc, x])
        inverse_lu(Xc.T @ Xc)
    t_scratch_all = time.perf_counter() - t0

    print(f"time: initial inverse                     {t_first * 1e3:8.2f} ms")
    print(f"      100 Sherman-Morrison updates         {t_sm * 1e3:8.2f} ms "
          f"({t_sm / 100 * 1e3:.3f} ms per row)")
    print(f"      one inverse from scratch (final X)   {t_scratch * 1e3:8.2f} ms")
    print(f"      from scratch after every new row     {t_scratch_all * 1e3:8.2f} ms")

# Почему solve_lu ошибается для A = [[1e-17, 1], [1, 1]].
# Без перестановок ведущий элемент равен 1e-17, множитель l21 = 1e17, и
#     U22 = 1 - 1e17 * 1 = 1 - 1e17,
# что в double округляется до -1e17: единица "теряется", она меньше
# 16 значащих цифр. Дальше y = (1, 2 - 1e17) ~ (1, -1e17), x2 = 1, и
#     x1 = (1 - x2) / 1e-17 = 0 / 1e-17 = 0.
# Ответ (0, 1) вместо (1, 1). Малое число 1e-17 в знаменателе умножает
# ошибку округления на 1e17. Если сначала переставить строки (частичный выбор
# главного элемента), ведущий элемент равен 1, множитель 1e-17, и
# U22 = 1 - 1e-17 ~ 1. Никакого катастрофического сокращения нет, ответ (1, 1).
#
# Sherman-Morrison. Добавление строки x к X меняет X^T X на X^T X + x x^T
# (обновление ранга 1). Формула пересчитывает обратную за O(n^2) операций
# (одно умножение матрицы на вектор и одно внешнее произведение) вместо
# O(n^3) для обращения с нуля. Для n = 50 одно обновление в разы дешевле
# одного полного обращения. Если обратная нужна после каждой новой строки
# (онлайн-регрессия), выигрыш получается в сотни раз (см. времена выше:
# ~1 мс на 100 обновлений против ~0.7 с на 100 обращений с нуля). Результат после
# 100 обновлений совпадает с вычислением с нуля до ~1e-14.