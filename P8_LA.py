"""  
P8. Why you should not compute the inverse. The Hilbert matrix Hn is the n × n matrix
with entries (Hn)ij = i+j
1
−1
. It is invertible, but very close to being singular, which makes
computations with it very sensitive to rounding errors. For n = 4, 6, 8, 10, 12, let x_true =
np.ones(n) and b = H @ x_true, and solve Hnx = b in three ways:
x1 = np.linalg.solve(H, b)
x2 = np.linalg.inv(H) @ b
x3 = inverse(H) @ b # your function from P4
For each method, print the error maxi
|xi−1| and the residual maxi
|(Hnx−b)i
|. What do you
observe? Which method keeps the residual smallest? What happens with your own inverse
for larger n, and why? (Hint: think about the tolerance in rref.) Relate your observations
to Remark 6.53
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
    """Gauss-Jordan inverse (from P4)."""
    A = np.array(A, dtype=float)
    n = A.shape[0]
    R, pivots = rref(np.hstack([A, np.eye(n)]), tol)
    if [p for p in pivots if p < n] != list(range(n)):
        raise np.linalg.LinAlgError("matrix is singular")
    return R[:, n:]


def hilbert(n):
    """(H_n)_ij = 1 / (i + j - 1) for i, j = 1..n (Python indices start at 0)."""
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = 1.0 / ((i + 1) + (j + 1) - 1)
    return H


def report(name, x, H, b):
    if x is None:
        print(f"   {name:22s} failed")
        return
    err = np.abs(x - 1).max()
    res = np.abs(H @ x - b).max()
    print(f"   {name:22s} error = {err:9.2e}   residual = {res:9.2e}")


if __name__ == "__main__":
    for n in [4, 6, 8, 10, 12]:
        H = hilbert(n)
        x_true = np.ones(n)
        b = H @ x_true
        print(f"n = {n:2d}   cond(H) = {np.linalg.cond(H):.1e}")

        x1 = np.linalg.solve(H, b)
        x2 = np.linalg.inv(H) @ b
        try:
            x3 = inverse(H) @ b                  # your function from P4
        except np.linalg.LinAlgError as err:
            print(f"   inverse(H) raised LinAlgError: {err}")
            x3 = None
        report("np.linalg.solve", x1, H, b)
        report("np.linalg.inv(H) @ b", x2, H, b)
        report("inverse(H) @ b", x3, H, b)
        if x3 is None:
            # With a smaller tolerance our function does return a "result"...
            x4 = inverse(H, tol=1e-20) @ b
            report("inverse(H, tol=1e-20)", x4, H, b)
        print()

# Что видно (точные числа печатаются выше):
# 1. Ошибка max|x_i - 1| быстро растёт с n у ВСЕХ методов, примерно как
#    cond(H) * 1e-16. При n = 12 (cond ~ 1e16) она ~0.3 даже у solve и ~6 у
#    inv(H) @ b, хотя точное решение — вектор из единиц. Это свойство самой задачи (H почти вырождена), а не
#    метода: любое округление в b или H умножается на cond(H).
# 2. Невязка max|(Hx - b)_i| у np.linalg.solve остаётся на уровне 1e-16 при
#    всех n. Это метод Гаусса с выбором главного элемента (LU), и он обратно
#    устойчив: найденный x — точное решение очень близкой системы.
#    У inv(H) @ b невязка заметно больше. Ошибки при вычислении H^{-1}
#    (элементы которой огромны, до ~1e15 при n = 12) затем умножаются на b.
# 3. Наша inverse из P4 при больших n перестаёт работать. В rref по ходу
#    исключения ведущие элементы становятся меньше tol = 1e-10, и rref решает,
#    что в этом столбце нет опорного элемента, то есть что H вырождена
#    ("matrix is singular"), хотя H_n обратима. Если уменьшить tol, обратная
#    "вычисляется", но с огромными ошибками и невязкой.
#    Допуск tol абсолютный, а элементы H_n и её "почти нулевые" ведущие
#    элементы малы, поэтому граница между "маленьким" и "нулём" тут условна.
# Вывод (ср. Remark 6.53): для решения Ax = b не нужно вычислять A^{-1}.
# Решайте систему методом исключения / LU (np.linalg.solve). Это дешевле
# (~n^3/3 против ~n^3 операций) и точнее (меньше невязка).