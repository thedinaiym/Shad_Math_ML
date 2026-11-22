"""
14 Programming Exercises
In these exercises, you implement the main algorithms of this lecture in Python. You need Python 3
with NumPy, and Matplotlib for Exercises P6 and P12. Implement the algorithms yourself, using
only loops, indexing and arithmetic on NumPy arrays; use NumPy’s built-in linear algebra routines
(np.linalg) only to check your results. Start each file with

import numpy as np
rng = np.random.default_rng(0) # random number generator for test matrices

A word on floating-point numbers. Computers store real numbers with only about 16 significant digits, 
so almost every computation has tiny rounding errors; for example, 
0.1 + 0.2
== 0.3 is False in Python. 
Therefore, never test whether a computed number is exactly zero:

test abs(x) < tol for a small tolerance such as tol = 1e-10 instead, and compare arrays with
np.allclose(X, Y). 
Also remember that Python counts from 0: the entry Aij of these notes is
A[i-1, j-1] in NumPy.
P1. Matrices of linear transformations. 
Write a function that takes a linear transformation
T : R
n → R
m, given as a Python function that maps NumPy vectors of length n to vectors of
length m, and returns its m × n matrix, whose i-th column is T(ei) (Theorem 6.9).
def matrix_of(T, n):

Return the matrix of the linear transformation T: R^n -> R^m.

...
T = lambda v: np.array([2*v[0] + 3*v[1], 4*v[0] - v[1]])

print(matrix_of(T, 2)) # should print [[ 2. 3.] [ 4. -1.]]

• Also test your function on T(x, y, z) = (x + 2y − z, 3x − y + 4z), and compare with the

matrix computed in the examples after Theorem 6.9.

• Write a function looks_linear(T, n, trials=100) that checks additivity and 
homogeneity for random vectors u = rng.normal(size=n), v = rng.normal(size=n) and
random scalars a = rng.normal(). 
Test it on the T above and on the maps (x, y) 7→
(x + 1, y) and (x, y) 7→ (xy, y).

• Why can such a random test prove that a map is not linear, but never prove that a map
is linear?
 
"""

import numpy as np
rng = np.random.default_rng(0)  

def matrix_of(T, n):
    """Return the matrix of the linear transformation T: R^n -> R^m."""
    columns = []
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0                      # i-th standard basis vector e_i
        columns.append(np.asarray(T(e), dtype=float))
    m = len(columns[0])
    M = np.zeros((m, n))
    for i in range(n):
        M[:, i] = columns[i]            # i-th column is T(e_i)
    return M


def looks_linear(T, n, trials=100):
    """Check additivity and homogeneity of T on random vectors and scalars.
    Returns False as soon as a counterexample is found, True otherwise.
    """
    for _ in range(trials):
        u = rng.normal(size=n)
        v = rng.normal(size=n)
        a = rng.normal()
        if not np.allclose(T(u + v), T(u) + T(v)):   # additivity
            return False
        if not np.allclose(T(a * u), a * T(u)):      # homogeneity
            return False
    return True


if __name__ == "__main__":
    T = lambda v: np.array([2*v[0] + 3*v[1], 4*v[0] - v[1]])
    print(matrix_of(T, 2))  # should print [[ 2. 3.] [ 4. -1.]]
    assert np.allclose(matrix_of(T, 2), [[2, 3], [4, -1]])

    # T(x, y, z) = (x + 2y - z, 3x - y + 4z)
    T3 = lambda v: np.array([v[0] + 2*v[1] - v[2], 3*v[0] - v[1] + 4*v[2]])
    M3 = matrix_of(T3, 3)
    print("\nMatrix of T(x, y, z) = (x + 2y - z, 3x - y + 4z):")
    print(M3)
    assert np.allclose(M3, [[1, 2, -1], [3, -1, 4]])
    # the matrix really represents T3: M3 @ x == T3(x)
    for _ in range(10):
        x = rng.normal(size=3)
        assert np.allclose(M3 @ x, T3(x))

    shift = lambda v: np.array([v[0] + 1, v[1]])        # (x, y) -> (x + 1, y)
    product = lambda v: np.array([v[0] * v[1], v[1]])   # (x, y) -> (xy, y)
    print()
    for name, f in [("T", T), ("(x, y) -> (x + 1, y)", shift),
                    ("(x, y) -> (xy, y)", product)]:
        print(f"looks_linear({name}) = {looks_linear(f, 2)}")
    assert looks_linear(T, 2)
    assert not looks_linear(shift, 2)
    assert not looks_linear(product, 2)

# Why can a random test prove that a map is NOT linear, but never prove that it IS?
# Линейность — это утверждение "для ВСЕХ u, v и всех a":
#     T(u + v) = T(u) + T(v),   T(a u) = a T(u).
# * Чтобы его опровергнуть, достаточно одного контрпримера. Если на какой-то
#   тройке (u, v, a) равенство нарушено сильнее ошибок округления,
#   то T точно не линейно. Так, (x, y) -> (x + 1, y) не переводит 0 в 0,
#   а для (x, y) -> (xy, y) T(2u) = (4xy, 2y) != 2 T(u).
# * Чтобы доказать, нужно проверить бесконечно много троек, а тест проверяет
#   только конечное число (здесь 100). Отображение может нарушать линейность
#   лишь на "тонком" множестве, куда случайные точки почти никогда не попадают.
#   Пример: T(x, y) = (x, y) при y != 0 и T(x, 0) = (0, 0). Оно не линейно,
#   потому что T(e1) = 0, но T(e1 + e2) - T(e2) = e1. Но rng.normal() почти
#   наверное не выдаст y = 0, и тест вернёт True.
# * Кроме того, сравнение идёт с допуском (np.allclose), поэтому крошечную
#   нелинейность, например T(x) = x + 1e-12 * x**2, тест тоже не заметит.