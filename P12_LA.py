"""   
P12. Homework. Dual bases and interpolation. Write a function dual_basis(P) that returns
the rows r1, . . . , rn of the dual basis of the columns of an invertible matrix P (Proposition 8.7).
• For a random invertible 5 × 5 matrix P, check that the matrix of all values ribj is
the identity, and that applying the rows to a vector x gives its coordinates, i.e., that
x =
P i
(rix) bi
.
• Polynomials of degree at most d are stored by their coefficient vectors in R
d+1. For
d + 1 distinct nodes t0, . . . , td, the evaluation functionals p 7→ p(tk) are the rows of the
Vandermonde matrix V with entries Vkj = t
j
k
. Explain why the coefficient vectors of
the Lagrange polynomials L0, . . . , Ld (with Lj (tk) = δjk) are the columns of V
−1
, and
compute them for the nodes −1, 0, 1; compare with Example 8.9.
• Interpolate f(t) = sin(3t) at 9 equally spaced nodes in [−1, 1] with the formula p = P
k
f(tk)Lk, and plot f and p. Then repeat with 21 equally spaced nodes on [−5, 5]
for f(t) = 1+
1
t
2 . What goes wrong near the ends of the interval? (This is Runge’s
phenomenon.)
"""

import numpy as np
rng = np.random.default_rng(0)  
import matplotlib.pyplot as plt


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


def dual_basis(P):
    """Rows r_1, ..., r_n of the dual basis of the columns b_1, ..., b_n of P.

    r_i b_j = delta_ij  <=>  R P = I  <=>  R = P^{-1}   (Proposition 8.7).
    Returned as a matrix whose i-th row is r_i.
    """
    return inverse(P)


def vandermonde(t, d):
    """V_kj = t_k^j for k = 0..len(t)-1, j = 0..d."""
    t = np.asarray(t, dtype=float)
    V = np.zeros((len(t), d + 1))
    for k in range(len(t)):
        for j in range(d + 1):
            V[k, j] = t[k] ** j
    return V


def lagrange_coefficients(nodes):
    """Column j = coefficient vector of the Lagrange polynomial L_j."""
    d = len(nodes) - 1
    return inverse(vandermonde(nodes, d))


def interpolate(f, nodes):
    """Coefficients of p = sum_k f(t_k) L_k."""
    C = lagrange_coefficients(nodes)
    return C @ f(np.asarray(nodes, dtype=float))


def evaluate(coeffs, s):
    """p(s) for p = sum_j coeffs[j] s^j (Horner's scheme)."""
    result = np.zeros_like(s, dtype=float)
    for c in coeffs[::-1]:
        result = result * s + c
    return result


if __name__ == "__main__":
    # 1. dual basis of a random invertible 5x5 matrix
    P = rng.normal(size=(5, 5))
    R = dual_basis(P)
    values = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            values[i, j] = R[i] @ P[:, j]           # r_i b_j
    print("max |(r_i b_j) - delta_ij| =", np.abs(values - np.eye(5)).max())
    assert np.allclose(values, np.eye(5))

    x = rng.normal(size=5)
    x_rebuilt = np.zeros(5)
    for i in range(5):
        x_rebuilt += (R[i] @ x) * P[:, i]           # sum_i (r_i x) b_i
    print("x == sum_i (r_i x) b_i :", np.allclose(x, x_rebuilt))
    assert np.allclose(x, x_rebuilt)

    # 2. Lagrange polynomials for the nodes -1, 0, 1
    nodes = [-1.0, 0.0, 1.0]
    V = vandermonde(nodes, 2)
    C = lagrange_coefficients(nodes)
    print("\nV =\n", V, "\nV^{-1} (columns = L_0, L_1, L_2 in the basis 1, t, t^2) =\n", C)
    # Example 8.9: L0 = t(t-1)/2, L1 = 1 - t^2, L2 = t(t+1)/2
    expected = np.array([[0.0, 1.0, 0.0],
                         [-0.5, 0.0, 0.5],
                         [0.5, -1.0, 0.5]])
    assert np.allclose(C, expected)
    print("L_0(t) = (t^2 - t)/2,  L_1(t) = 1 - t^2,  L_2(t) = (t^2 + t)/2")
    assert np.allclose(V @ C, np.eye(3))            # L_j(t_k) = delta_jk

    # 3. interpolation and Runge's phenomenon
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    f1 = lambda t: np.sin(3 * t)
    nodes1 = np.linspace(-1, 1, 9)
    c1 = interpolate(f1, nodes1)
    s1 = np.linspace(-1, 1, 500)
    err1 = np.abs(evaluate(c1, s1) - f1(s1)).max()
    axes[0].plot(s1, f1(s1), label="f(t) = sin(3t)")
    axes[0].plot(s1, evaluate(c1, s1), "--", label="interpolant p, 9 nodes")
    axes[0].plot(nodes1, f1(nodes1), "ko", ms=4)
    axes[0].set_title(f"sin(3t) on [-1, 1]: max |p - f| = {err1:.1e}")
    axes[0].legend()

    f2 = lambda t: 1 / (1 + t ** 2)
    nodes2 = np.linspace(-5, 5, 21)
    c2 = interpolate(f2, nodes2)
    s2 = np.linspace(-5, 5, 1000)
    err2 = np.abs(evaluate(c2, s2) - f2(s2)).max()
    axes[1].plot(s2, f2(s2), label="f(t) = 1/(1 + t^2)")
    axes[1].plot(s2, evaluate(c2, s2), "--", label="interpolant p, 21 nodes")
    axes[1].plot(nodes2, f2(nodes2), "ko", ms=4)
    axes[1].set_ylim(-1.5, 2)
    axes[1].set_title(f"Runge: 1/(1+t^2) on [-5, 5]: max |p - f| = {err2:.1f}")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig("P12_interpolation.png", dpi=130)
    print(f"\nsin(3t), 9 nodes:       max |p - f| = {err1:.2e}")
    print(f"1/(1+t^2), 21 nodes:    max |p - f| = {err2:.2f}")
    print("saved P12_interpolation.png")
    plt.show()

# Почему столбцы V^{-1} — коэффициенты многочленов Лагранжа.
# Строка k матрицы V — это функционал "вычисление в t_k": если c —
# коэффициенты многочлена p, то (V c)_k = sum_j c_j t_k^j = p(t_k).
# L_j определяется условием L_j(t_k) = delta_jk, то есть V c(L_j) = e_j,
# откуда c(L_j) = V^{-1} e_j — j-й столбец V^{-1}. То же самое на языке двойственных
# базисов: функционалы ev_{t_0}, ..., ev_{t_d} (строки V) образуют базис
# пространства, двойственного к R[x]_{<=d}, а L_0, ..., L_d — двойственный
# к нему базис R[x]_{<=d}. Поэтому p = sum_k p(t_k) L_k.
#
# Феномен Рунге. При равноотстоящих узлах интерполяционный многочлен высокой
# степени (здесь 20) совпадает с f в узлах, но между узлами у концов отрезка
# сильно осциллирует. Ошибка у краёв огромна (~60, см. вывод) и растёт с
# числом узлов. Причина в том, что многочлены Лагранжа L_k для равноотстоящих
# узлов сами принимают огромные значения у концов отрезка (константа Лебега
# растёт экспоненциально). Для sin(3t) на [-1, 1] с 9 узлами всё хорошо,
# ошибка ~1e-3. Лечится узлами Чебышёва, которые сгущаются к концам отрезка.