""" 
P11. Diagonalization in floating-point arithmetic. The function np.linalg.eig(A) returns
the eigenvalues and a matrix P whose columns are eigenvectors.
• For a random 5 × 5 matrix A, check that A is close to P DP −1
. (The eigenvalues may be
complex.) Compare P D20P
−1 with np.linalg.matrix_power(A, 20).
• Compute the Fibonacci numbers Fn in two ways: exactly, with Python integers and
the recursion, and with Binet’s formula (Example 10.33) in floating-point arithmetic,
rounding to the nearest integer. For which n does Binet’s formula give a wrong answer
for the first time, and why?
• Apply np.linalg.eig to the shear 1 1
0 1! . What do you notice about the two eigenvectors it returns?"""
import numpy as np
rng = np.random.default_rng(0) 
import math

def fib_exact(n_max):
    """F_0, ..., F_{n_max} with exact Python integers."""
    F = [0, 1]
    while len(F) <= n_max:
        F.append(F[-1] + F[-2])
    return F


def fib_binet(n):
    """Binet's formula (Example 10.33) in floating point, rounded."""
    sqrt5 = math.sqrt(5)
    phi = (1 + sqrt5) / 2
    psi = (1 - sqrt5) / 2
    return round((phi ** n - psi ** n) / sqrt5)


if __name__ == "__main__":
    # 1. A = P D P^{-1}
    A = rng.normal(size=(5, 5))
    vals, P = np.linalg.eig(A)
    print("eigenvalues of A:", np.round(vals, 4))
    Pinv = np.linalg.inv(P)
    A_rec = P @ np.diag(vals) @ Pinv
    print("max |A - P D P^-1| =", np.abs(A - A_rec).max(),
          "  (imaginary part:", np.abs(A_rec.imag).max(), ")")
    assert np.allclose(A_rec, A)

    A20 = P @ np.diag(vals ** 20) @ Pinv
    A20_np = np.linalg.matrix_power(A, 20)
    rel = np.abs(A20 - A20_np).max() / np.abs(A20_np).max()
    print(f"P D^20 P^-1 vs matrix_power(A, 20): relative difference {rel:.1e}, "
          f"max |imag| = {np.abs(A20.imag).max():.1e}")
    assert np.allclose(A20.real, A20_np, rtol=1e-8, atol=1e-8 * np.abs(A20_np).max())

    # 2. Fibonacci: exact vs Binet
    F = fib_exact(100)
    first_wrong = None
    for n in range(101):
        if fib_binet(n) != F[n]:
            first_wrong = n
            break
    print(f"\nBinet's formula is first wrong at n = {first_wrong}:")
    for n in range(first_wrong - 2, first_wrong + 3):
        print(f"   n = {n}: exact F_n = {F[n]:>20d}   Binet = {fib_binet(n):>20d}"
              f"   diff = {fib_binet(n) - F[n]}")
    print(f"   F_{first_wrong} ~ {F[first_wrong]:.2e},  2^53 ~ {2.0 ** 53:.2e}")

    # 3. the shear [[1, 1], [0, 1]]
    S = np.array([[1.0, 1.0], [0.0, 1.0]])
    vals, P = np.linalg.eig(S)
    print("\nshear: eigenvalues", vals)
    print("eigenvector matrix P =\n", P)
    print("det(P) =", np.linalg.det(P), " cond(P) =", f"{np.linalg.cond(P):.1e}")

# Почему Binet ошибается начиная с n = 71 (первое неверное n печатается выше).
# phi хранится не точно: float(phi) = phi * (1 + delta), где delta ~ 3.4e-17
# (относительная ошибка округления). Тогда
#     float(phi)^n ~ phi^n * (1 + n * delta),
# то есть относительная ошибка растёт в n раз, а абсолютная ошибка результата
# ~ F_n * n * delta. Пока она меньше 0.5, round() даёт верное целое.
# При n = 71 имеем F_71 * 71 * delta ~ 3.08e14 * 71 * 3.4e-17 ~ 0.73 > 0.5,
# и ответ ошибается на 1.
# Ответ сбивается задолго до того, как F_n перестаёт помещаться в мантиссу
# (2^53 ~ 9e15): мешает накопленная ошибка, а не только конечность мантиссы.
# Целые числа Python точны при любом n.
#
# Сдвиг [[1, 1], [0, 1]]. Собственное значение 1 кратности 2, но собственное
# пространство одномерно (span{e1}), поэтому матрица не диагонализуема.
# np.linalg.eig всё равно возвращает два столбца, но они почти параллельны:
# (1, 0) и (-1, 2.2e-16). Второй отличается от -e1 лишь на ошибку округления.
# Матрица P практически вырождена (det ~ 2e-16, огромное число обусловленности),
# и формулу A = P D P^{-1} численно использовать нельзя.