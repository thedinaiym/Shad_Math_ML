"""   
P10. Power iteration and PageRank. Implement power iteration:
98
def power_iteration(A, num_iter=100):
Approximate an eigenvector for the eigenvalue of largest absolute value.
x = rng.normal(size=A.shape[0])
for _ in range(num_iter):
x = A @ x
x = x / np.abs(x).max() # rescale to avoid overflow
...
and estimate the eigenvalue from the final vector x, for instance as (Ax)i/xi for an index i
with xi ̸= 0.
• Test it on the matrices of Examples 10.7 and 10.8, and compare with np.linalg.eig.
How does the speed of convergence depend on the ratio |λ2/λ1| of the two largest eigenvalues (in absolute value)?
• What happens for the rotation by 90◦
, and for diag(1, −1)? Explain.
• A tiny web has 4 pages, with links 1 → 2, 1 → 3, 2 → 3, 3 → 1, 4 → 3. Let L be the
4 × 4 matrix with Lij = number of links on page
1
j
if page j links to page i, and 0 otherwise.
The PageRank vector is the stationary distribution of G = 0.85L +
0.
4
15 11⊤, where 11⊤
is the matrix of all ones. Compute it with power iteration, check that it is an eigenvector
of G for the eigenvalue 1, and rank the pages. """
import numpy as np
rng = np.random.default_rng(0)  

def power_iteration(A, num_iter=100):
    """Approximate an eigenvector for the eigenvalue of largest absolute value.

    Returns (lam, x): the eigenvalue estimate and the (rescaled) vector.
    """
    A = np.asarray(A, dtype=float)
    x = rng.normal(size=A.shape[0])
    for _ in range(num_iter):
        x = A @ x
        x = x / np.abs(x).max()    # rescale to avoid overflow
    Ax = A @ x
    i = np.argmax(np.abs(x))       # an index with x_i != 0
    lam = Ax[i] / x[i]
    return lam, x


def iterations_to_converge(A, v_true, tol=1e-8, max_iter=10000):
    """Number of iterations until the direction of x is tol-close to v_true."""
    x = rng.normal(size=A.shape[0])
    v_true = v_true / np.sqrt(np.sum(v_true ** 2))
    for k in range(1, max_iter + 1):
        x = A @ x
        x = x / np.sqrt(np.sum(x ** 2))
        if min(np.abs(x - v_true).max(), np.abs(x + v_true).max()) < tol:
            return k
    return None


if __name__ == "__main__":
    examples = {
        "Example 10.7": np.array([[2.0, 1.0], [1.0, 2.0]]),   # eigenvalues 3, 1
        "Example 10.8": np.array([[4.0, 1.0], [2.0, 3.0]]),   # eigenvalues 5, 2
    }
    for name, A in examples.items():
        lam, x = power_iteration(A)
        vals, vecs = np.linalg.eig(A)
        k = np.argmax(np.abs(vals))
        v = vecs[:, k] / np.abs(vecs[:, k]).max()
        print(f"{name}: power iteration lambda = {lam:.12f}, x = {x / np.abs(x).max()}")
        print(f"{'':13s} np.linalg.eig  lambda = {vals[k]:.12f}, v = {v}")
        assert np.isclose(lam, vals[k])

    # speed of convergence vs |lambda2 / lambda1|
    print("\n|l2/l1|   iterations to 1e-8   predicted log(1e-8)/log|l2/l1|")
    Q, _ = np.linalg.qr(rng.normal(size=(2, 2)))           # random rotation
    for ratio in [0.1, 0.5, 0.9, 0.99]:
        A = Q @ np.diag([1.0, ratio]) @ Q.T                # eigenvalues 1, ratio
        k = iterations_to_converge(A, Q[:, 0])
        print(f"{ratio:6.2f}   {k:12d}   {np.log(1e-8) / np.log(ratio):26.0f}")

    # rotation by 90 degrees and diag(1, -1)
    for name, A in [("rotation by 90°", np.array([[0.0, -1.0], [1.0, 0.0]])),
                    ("diag(1, -1)", np.diag([1.0, -1.0]))]:
        print(f"\n{name}: eigenvalues {np.linalg.eig(A)[0]}")
        x = rng.normal(size=2)
        for it in range(1, 7):
            x = A @ x
            x = x / np.abs(x).max()
            print(f"   iteration {it}: x = {np.round(x, 4)}")
        for n_it in (100, 101):
            print(f"   power_iteration(num_iter={n_it}) -> lambda estimate "
                  f"{power_iteration(A, n_it)[0]:.4f}")

    # PageRank: links 1->2, 1->3, 2->3, 3->1, 4->3
    links = [(1, 2), (1, 3), (2, 3), (3, 1), (4, 3)]
    n = 4
    out_degree = np.zeros(n)
    for j, i in links:
        out_degree[j - 1] += 1
    L = np.zeros((n, n))
    for j, i in links:                      # page j links to page i
        L[i - 1, j - 1] = 1 / out_degree[j - 1]
    G = 0.85 * L + 0.15 / 4 * np.ones((n, n))
    print("\nL =\n", L, "\nG =\n", np.round(G, 4))

    _, p = power_iteration(G)
    p = p / p.sum()                          # a probability distribution
    print("PageRank vector:", np.round(p, 4))
    print("G p == p :", np.allclose(G @ p, p))
    assert np.allclose(G @ p, p)
    order = np.argsort(-p) + 1
    print("ranking of pages (best first):", order.tolist())

# Скорость сходимости.
# Разложим x0 = c1 v1 + c2 v2 + ... по собственным векторам. Тогда
#     A^k x0 = lambda1^k (c1 v1 + c2 (lambda2/lambda1)^k v2 + ...),
# и ошибка направления убывает как |lambda2/lambda1|^k. Число итераций до
# точности eps примерно равно log(eps) / log|lambda2/lambda1|. При отношении
# 0.5 нужно ~27 итераций, при 0.99 — ~2000 (см. таблицу выше). Чем ближе
# |lambda2| к |lambda1|, тем медленнее сходимость.
#
# Поворот на 90°: собственные значения ±i, у них одинаковый модуль 1, и они
# комплексные. Вещественного собственного вектора нет, и x просто
# поворачивается на 90° на каждом шаге (период 4). Метод не сходится, а
# оценка lambda зависит от числа итераций.
#
# diag(1, -1): |lambda1| = |lambda2| = 1. Компонента вдоль e2 на каждом шаге
# меняет знак, поэтому x скачет между (a, b) и (a, -b) и не сходится (если
# только b не равно 0). Модули компонент не меняются, и оценка (Ax)_i / x_i
# выдаёт +1 или -1 в зависимости от того, какая компонента СЛУЧАЙНОГО
# начального вектора больше. Метод не "выбирает" собственное значение.
# Предположение |lambda1| > |lambda2| здесь нарушено.
#
# PageRank: страница 3 на первом месте, на неё ссылаются 1, 2 и 4. Затем идёт
# страница 1, на неё ссылается "важная" страница 3. Потом страница 2. На
# странице 4 нет входящих ссылок, и у неё минимальный ранг 0.15/4 = 0.0375.