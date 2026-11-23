""" 
P6. Homework. Visualizing linear transformations. Using Matplotlib, write a function that
draws the unit square, a few grid lines inside it and the letter F, before and after applying a
2 × 2 matrix A, as in Figure 2.
def plot_transformation(A):
Draw the unit square, a grid and the letter F, before and after A.
...
Store a shape as an array P with one point per row; then its image under A is P @ A.T. The
letter F in the figures has the corners
(0.2, 0.1), (0.4, 0.1), (0.4, 0.45), (0.65, 0.45), (0.65, 0.6),
(0.4, 0.6), (0.4, 0.75), (0.8, 0.75), (0.8, 0.9), (0.2, 0.9).
• Reproduce the four panels of Figure 2, and also plot the singular matrix 1 2
2 4! .
• The signed area of a polygon with corners (x1, y1), . . . ,(xk, yk) is given by the shoelace
formula

(xiyi+1 − xi+1yi), where (xk+1, yk+1) = (x1, y1).
It is positive if the corners are listed counterclockwise, and negative if they are listed
clockwise. Implement it, apply it to the image of the unit square with corners listed in
the order (0, 0),(1, 0),(1, 1),(0, 1), and check that the result equals det(A), including the
sign, for all the matrices above and for the matrix 3 1
1 2! of Figure 4 
"""
import numpy as np
rng = np.random.default_rng(0)  
import matplotlib.pyplot as plt

SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
LETTER_F = np.array([(0.2, 0.1), (0.4, 0.1), (0.4, 0.45), (0.65, 0.45),
                     (0.65, 0.6), (0.4, 0.6), (0.4, 0.75), (0.8, 0.75),
                     (0.8, 0.9), (0.2, 0.9)])


def grid_lines(k=4):
    """Lines x = i/k and y = i/k inside the unit square (each as a 2x2 array)."""
    lines = []
    for t in np.linspace(0, 1, k + 1)[1:-1]:
        lines.append(np.array([[t, 0.0], [t, 1.0]]))   # vertical
        lines.append(np.array([[0.0, t], [1.0, t]]))   # horizontal
    return lines


def closed(P):
    """Repeat the first point at the end so the polygon is drawn closed."""
    return np.vstack([P, P[:1]])


def det2(A):
    return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]


def plot_transformation(A, ax=None, title=None):
    """Draw the unit square, a grid and the letter F, before and after A."""
    A = np.asarray(A, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    # before: grey, dashed
    for L in grid_lines():
        ax.plot(L[:, 0], L[:, 1], color="0.75", lw=0.7, ls="--")
    S = closed(SQUARE)
    ax.plot(S[:, 0], S[:, 1], color="0.55", lw=1.0, ls="--")
    ax.fill(LETTER_F[:, 0], LETTER_F[:, 1], color="0.85", zorder=1)

    # after: colored
    for L in grid_lines():
        LA = L @ A.T
        ax.plot(LA[:, 0], LA[:, 1], color="tab:blue", lw=0.7)
    SA = closed(SQUARE) @ A.T
    ax.plot(SA[:, 0], SA[:, 1], color="tab:blue", lw=1.8)
    FA = LETTER_F @ A.T
    ax.fill(FA[:, 0], FA[:, 1], color="tab:orange", alpha=0.85, zorder=2)

    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_aspect("equal")
    ax.set_title(title or f"A = {A.tolist()}, det = {det2(A):.2f}", fontsize=9)
    return ax


def signed_area(P):
    """Shoelace formula: 1/2 * sum (x_i y_{i+1} - x_{i+1} y_i), P = corners (rows)."""
    k = len(P)
    s = 0.0
    for i in range(k):
        x1, y1 = P[i]
        x2, y2 = P[(i + 1) % k]      # (x_{k+1}, y_{k+1}) = (x_1, y_1)
        s += x1 * y2 - x2 * y1
    return s / 2


if __name__ == "__main__":
    c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
    # Four panels of Figure 2. If Figure 2 in your notes uses other matrices,
    # just replace them in this list.
    matrices = {
        "stretch": np.array([[2.0, 0.0], [0.0, 1.0]]),
        "rotation by 30°": np.array([[c, -s], [s, c]]),
        "shear": np.array([[1.0, 1.0], [0.0, 1.0]]),
        "reflection": np.array([[-1.0, 0.0], [0.0, 1.0]]),
        "singular": np.array([[1.0, 2.0], [2.0, 4.0]]),
        "Figure 4": np.array([[3.0, 1.0], [1.0, 2.0]]),
    }

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, (name, A) in zip(axes.ravel(), matrices.items()):
        plot_transformation(A, ax, f"{name}: A = {np.round(A, 2).tolist()}\n"
                                   f"det A = {det2(A):.3f}")
    fig.suptitle("Grey dashed: before, colored: after applying A")
    fig.tight_layout()
    fig.savefig("P6_transformations.png", dpi=130)
    print("saved P6_transformations.png")

    # Shoelace formula on the image of the unit square
    print(f"\n{'matrix':18s} {'signed area':>12s} {'det(A)':>10s}")
    for name, A in matrices.items():
        area = signed_area(SQUARE @ A.T)   # corners (0,0),(1,0),(1,1),(0,1)
        print(f"{name:18s} {area:12.4f} {det2(A):10.4f}")
        assert abs(area - det2(A)) < 1e-10
    print("signed area of the image of the unit square = det(A), including the sign")

    plt.show()

# Почему так получается:
# * Образ единичного квадрата — параллелограмм с вершинами 0, a1, a1 + a2, a2,
#   где a1, a2 — столбцы A. Формула шнурка для него даёт
#   1/2 * 2 * (a11*a22 - a12*a21) = det(A).
# * Знак показывает ориентацию. При det > 0 вершины образа идут против часовой
#   стрелки, как у исходного квадрата. Отражение (det = -1) меняет обход на
#   обратный, поэтому площадь отрицательна, и буква F на картинке зеркальная.
# * Вырожденная матрица [[1, 2], [2, 4]] сплющивает всё на прямую y = 2x,
#   поэтому площадь образа равна 0 = det A.