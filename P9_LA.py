""" 
P9. Mini-project: fitting a line to data. Given data points (x1, y1), . . . ,(xN , yN ), we look for
the line y = w0 + w1x that fits the data best, in the sense that the sum of squared errors
P
i
(yi − w0 − w1xi)
2
is as small as possible. This is the simplest example of linear regression.
Let X be the N × 2 matrix with rows (1, xi), and let y = (y1, . . . , yN )
⊤. One can show (we
do not prove this here) that the best w = (w0, w1)
⊤ is the solution of the normal equations
(X⊤X) w = X⊤y.
• Generate data on a line with some noise:
x = np.linspace(0, 1, 50)
y = 2 + 3 * x + rng.normal(0, 0.1, size=50) # a line plus noise
X = np.column_stack([np.ones_like(x), x])
• Compute w using your own functions from P2 and P4, and check that w is close to (2, 3).
Compare with np.linalg.lstsq(X, y, rcond=None)[0].
• Explain why X⊤X is symmetric. Show that if the columns of X are linearly independent, then X⊤X is invertible. (Hint: if X⊤Xw = 0, compute w⊤X⊤Xw and use
Corollary 7.11.)
• Fit a parabola y = w0 + w1x + w2x
2
to the data y = 1 − 2x + 4x
2 + noise, using the
columns 1, x, x2
. How is this related to the vector space R[x]≤2?
"""

import numpy as np
rng = np.random.default_rng(0)  
def matmul(A, B):
    """Matrix product with three nested loops (from P2)."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    m, p = A.shape
    p2, n = B.shape
    if p != p2:
        raise ValueError(f"cannot multiply {A.shape} by {B.shape}")
    C = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            s = 0.0
            for k in range(p):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C


def rref(A, tol=1e-10):
    """RREF with partial pivoting (from P3)."""
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
# -----------------------------------------------------------------------------


def fit_normal_equations(X, y):
    """Solve (X^T X) w = X^T y with our own matmul and inverse."""
    Xt = X.T
    XtX = matmul(Xt, X)
    Xty = matmul(Xt, y.reshape(-1, 1))
    return matmul(inverse(XtX), Xty).ravel(), XtX


if __name__ == "__main__":
    # data on a line with noise
    x = np.linspace(0, 1, 50)
    y = 2 + 3 * x + rng.normal(0, 0.1, size=50)   # a line plus noise
    X = np.column_stack([np.ones_like(x), x])

    w, XtX = fit_normal_equations(X, y)
    w_np = np.linalg.lstsq(X, y, rcond=None)[0]
    print("line fit:  w (normal equations) =", w)
    print("           np.linalg.lstsq       =", w_np)
    assert np.allclose(w, w_np)
    assert np.allclose(w, [2, 3], atol=0.15)
    print("X^T X =\n", XtX, "\nsymmetric:", np.allclose(XtX, XtX.T))

    # parabola y = 1 - 2x + 4x^2 + noise, columns 1, x, x^2
    y2 = 1 - 2 * x + 4 * x ** 2 + rng.normal(0, 0.1, size=50)
    X2 = np.column_stack([np.ones_like(x), x, x ** 2])
    w2, _ = fit_normal_equations(X2, y2)
    w2_np = np.linalg.lstsq(X2, y2, rcond=None)[0]
    print("\nparabola fit: w =", w2, " (true: [1, -2, 4])")
    print("              np.linalg.lstsq =", w2_np)
    assert np.allclose(w2, w2_np)

# 1. Почему X^T X симметрична: (X^T X)^T = X^T (X^T)^T = X^T X.
#
# 2. Если столбцы X линейно независимы, то X^T X обратима.
#    Пусть X^T X w = 0. Тогда
#        0 = w^T X^T X w = (Xw)^T (Xw) = ||Xw||^2,
#    значит Xw = 0 (вектор нулевой длины нулевой, Corollary 7.11).
#    Xw — линейная комбинация столбцов X с коэффициентами w. Столбцы
#    независимы, поэтому w = 0. Итак, ker(X^T X) = {0}, и квадратная матрица
#    X^T X обратима.
#
# 3. Связь с R[x]_{<=2}. Вектор w = (w0, w1, w2) — координаты многочлена
#    p(x) = w0 + w1 x + w2 x^2 в базисе мономов 1, x, x^2 пространства
#    R[x]_{<=2}. Матрица X со столбцами 1, x, x^2 — матрица линейного
#    отображения "вычисление в точках":
#        R[x]_{<=2} -> R^N,  p -> (p(x_1), ..., p(x_N)).
#    Подгонка параболы ищет многочлен из R[x]_{<=2}, значения которого ближе
#    всего к y. Модель остаётся линейной по w, хотя нелинейна по x, поэтому
#    работают те же нормальные уравнения.