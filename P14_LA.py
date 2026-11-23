"""P14. Differentiation and integration as matrices. Store a polynomial a0 + a1x + · · · + adx
d
by its coefficient vector np.array([a0, a1, ..., ad]).
• Write functions diff_matrix(d), which returns the d × (d + 1) matrix of D : R[x]≤d →
R[x]≤d−1, Dp = p
′
, and int_matrix(d), which returns the (d + 2) × (d + 1) matrix of
J : R[x]≤d → R[x]≤d+1, (Jp)(x) = R 0
x
p(t) dt, with respect to the bases of monomials.
• Check that diff_matrix(d + 1) @ int_matrix(d) is the identity matrix, and that
int_matrix(d - 1) @ diff_matrix(d) is not. Which coefficient does the second product lose?
• The midpoint sum MN (f) = N
1 P N
k=1 f(xk) with xk =
k−
N
1/2
approximates R 0
1
f(x) dx.
Explain why MN is a linear functional on C[0, 1], and why it agrees with the integral
on all polynomials of degree at most 1. Compute MN (f) for f(x) = sin(πx) and N =
10, 100, 1000, and compare with R 0
1
sin(πx) dx = π
2
. By which factor does the error shrink
when N grows by a factor of 10?
""" 

import numpy as np
rng = np.random.default_rng(0)  

def diff_matrix(d): 
    """d x (d+1) matrix of D: R[x]_{<=d} -> R[x]_{<=d-1}, Dp = p'."""
    D = np.zeros((d, d + 1))
    for k in range(1, d + 1):
        D[k - 1, k] = k            # (x^k)' = k x^{k-1}
    return D


def int_matrix(d):
    """(d+2) x (d+1) matrix of J: R[x]_{<=d} -> R[x]_{<=d+1}, (Jp)(x) = int_0^x p."""
    J = np.zeros((d + 2, d + 1))
    for k in range(d + 1):
        J[k + 1, k] = 1 / (k + 1)  # int_0^x t^k dt = x^{k+1} / (k+1)
    return J


def midpoint_sum(f, N):
    """M_N(f) = (1/N) sum_{k=1}^N f(x_k), x_k = (k - 1/2) / N."""
    s = 0.0
    for k in range(1, N + 1):
        s += f((k - 0.5) / N)
    return s / N


if __name__ == "__main__":
    d = 4
    print("diff_matrix(4) =\n", diff_matrix(d))
    print("int_matrix(4) =\n", np.round(int_matrix(d), 4))

    # check on p(x) = 1 + 2x + 3x^2: p' = 2 + 6x, int_0^x p = x + x^2 + x^3
    p = np.array([1.0, 2.0, 3.0])
    assert np.allclose(diff_matrix(2) @ p, [2, 6])
    assert np.allclose(int_matrix(2) @ p, [0, 1, 1, 1])

    DJ = diff_matrix(d + 1) @ int_matrix(d)
    JD = int_matrix(d - 1) @ diff_matrix(d)
    print("\ndiff_matrix(d+1) @ int_matrix(d) =\n", DJ)
    print("int_matrix(d-1) @ diff_matrix(d) =\n", JD)
    assert np.allclose(DJ, np.eye(d + 1))
    assert not np.allclose(JD, np.eye(d + 1))
    q = rng.normal(size=d + 1)
    print("\nrandom q =", np.round(q, 4))
    print("J D q    =", np.round(JD @ q, 4), " <- the constant term a0 is lost")

    # midpoint sums
    for N in [1, 5, 50]:
        assert np.isclose(midpoint_sum(lambda x: 1.0, N), 1.0)      # int 1 = 1
        assert np.isclose(midpoint_sum(lambda x: x, N), 0.5)        # int x = 1/2
        assert np.isclose(midpoint_sum(lambda x: 7 - 3 * x, N), 5.5)
    print("\nM_N is exact on polynomials of degree <= 1: OK")
    print("but M_N(x^2) for N = 5:", midpoint_sum(lambda x: x ** 2, 5), "vs 1/3")

    exact = 2 / np.pi
    print(f"\n   N        M_N(sin(pi x))     error        error ratio")
    previous = None
    for N in [10, 100, 1000]:
        M = midpoint_sum(lambda x: np.sin(np.pi * x), N)
        err = abs(M - exact)
        ratio = "" if previous is None else f"{previous / err:10.1f}"
        print(f"{N:5d}   {M:.15f}   {err:.3e}   {ratio}")
        previous = err

# Почему D(J p) = p, но J(D p) != p.
# D J = I: сначала берём первообразную, потом производную, и получаем исходное.
# J D = diag(0, 1, ..., 1): int_0^x p'(t) dt = p(x) - p(0), то есть теряется
# свободный член a0. У производной константы нет информации о ней (D имеет ядро
# — константы), а J всегда выдаёт многочлен с нулём в 0.
#
# Почему M_N — линейный функционал на C[0, 1]:
# M_N(f + g) = (1/N) sum (f(x_k) + g(x_k)) = M_N(f) + M_N(g),  M_N(a f) = a M_N(f).
# Каждое вычисление f -> f(x_k) линейно, а M_N — их линейная комбинация.
#
# Почему M_N точна на многочленах степени <= 1. Достаточно проверить на базисе
# 1, x (по линейности и M_N, и интеграл линейны):
#   M_N(1) = (1/N) * N = 1 = int_0^1 1 dx,
#   M_N(x) = (1/N) sum (k - 1/2)/N = (1/N^2)(N(N+1)/2 - N/2) = 1/2 = int_0^1 x dx.
# (На каждом отрезке [(k-1)/N, k/N] линейная функция симметрична относительно
#  середины, поэтому значение в середине = среднему значению.)
#
# Ошибка для sin(pi x) уменьшается примерно в 100 раз, когда N растёт в 10 раз.
# Ошибка метода средних прямоугольников ~ f''/(24 N^2) = O(1/N^2).