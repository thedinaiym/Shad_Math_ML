"""
P3. Homework. Row reduction. Implement a function that computes the reduced row echelon
form of a matrix and its pivot columns.
def rref(A, tol=1e-10):
    Return (R, pivots): the RREF of A and the list of pivot columns.
    R = np.array(A, dtype=float) # work on a copy of A
...
Go through the columns from left to right, keeping track of the row r in which the next pivot
will be placed. For each column c, find the entry with the largest absolute value among the
rows r, r + 1, . . . of that column. If it is smaller than tol, the column has no pivot. Otherwise,
swap its row with row r, divide row r by the pivot, and subtract multiples of row r from all the
other rows so that all other entries of column c become zero. (Choosing the largest available
entry as the pivot is called partial pivoting; it reduces rounding errors.)
• Test your function: the matrix 
1 2 -1
3 −1 4 
!
should give 
1 0 1
0 1 −1
!
with pivots 
[0, 1], a
3 −1 4
−1
!
should give 1 0 1
0 1 −1
!
with pivots [0,
1], and the matrix F of Example 6.31 should give

1 0 −1
0 1 2
0 0 0

• Using rref, write the functions rank(A) and null_space_basis(A), which returns one
basis vector for each free variable as in the proof of the rank–nullity theorem, and
col_space_basis(A), which returns the pivot columns of the original matrix (Theorem 7.2).
• For 20 random 5 × 7 matrices A, check that A @ v is close to zero for every vector
v in null_space_basis(A), that rank(A) + len(null_space_basis(A)) == 7 (rank–
nullity), and that rank(A) agrees with np.linalg.matrix_rank(A).
• Random matrices almost always have the largest possible rank. Construct a 6×8 matrix
of rank 4 as the product of a random 6 × 4 matrix and a random 4 × 8 matrix, and check
that your functions find rank 4 and a 4-dimensional null space.
    
    """
import numpy as np
rng = np.random.default_rng(0)  

def rref(A, tol=1e-10):
    """Return (R, pivots): the RREF of A and the list of pivot columns."""
    R = np.array(A, dtype=float)  
    m, n = R.shape
    pivots = []
    r = 0                         
    for c in range(n):
        if r == m:
            break
        p = r + np.argmax(np.abs(R[r:, c]))
        if abs(R[p, c]) < tol:
            R[r:, c] = 0.0         # treat these entries as exact zeros
            continue               # no pivot in this column
        if p != r:
            R[[r, p]] = R[[p, r]]  # swap rows r and p
        R[r] = R[r] / R[r, c]      # make the pivot equal to 1
        for i in range(m):         # clear all other entries of column c
            if i != r:
                R[i] = R[i] - R[i, c] * R[r]
        pivots.append(c)
        r += 1
    return R, pivots


def rank(A, tol=1e-10):
    """Rank = number of pivots."""
    return len(rref(A, tol)[1])


def null_space_basis(A, tol=1e-10):
    """One basis vector of the null space for each free variable."""
    R, pivots = rref(A, tol)
    n = R.shape[1]
    free = [c for c in range(n) if c not in pivots]
    basis = []
    for f in free:
        v = np.zeros(n)
        v[f] = 1.0                 # free variable x_f = 1, other free ones = 0
        for i, p in enumerate(pivots):
            v[p] = -R[i, f]        # solve row i of R x = 0 for the pivot x_p
        basis.append(v)
    return basis


def col_space_basis(A, tol=1e-10):
    """The pivot columns of the ORIGINAL matrix A (Theorem 7.2)."""
    A = np.asarray(A, dtype=float)
    _, pivots = rref(A, tol)
    return A[:, pivots]


if __name__ == "__main__":
    # Test 1
    A = np.array([[1, 2, -1],
                  [3, -1, 4]])
    R, piv = rref(A)
    print("RREF of [[1, 2, -1], [3, -1, 4]]:\n", R, "\npivots:", piv)
    assert np.allclose(R, [[1, 0, 1], [0, 1, -1]]) and piv == [0, 1]

    # Test 2: matrix F of Example 6.31
    # (Its RREF must be [[1,0,-1],[0,1,2],[0,0,0]]. The classic matrix with
    #  rows 1..9 has exactly this RREF; replace F if your notes use another one.)
    F = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    R, piv = rref(F)
    print("\nRREF of F:\n", R, "\npivots:", piv)
    assert np.allclose(R, [[1, 0, -1], [0, 1, 2], [0, 0, 0]])
    print("rank(F) =", rank(F))
    print("null space basis of F:", null_space_basis(F))
    print("column space basis of F:\n", col_space_basis(F))

    # 20 random 5x7 matrices
    for _ in range(20):
        A = rng.normal(size=(5, 7))
        N = null_space_basis(A)
        for v in N:
            assert np.allclose(A @ v, 0)
        assert rank(A) + len(N) == 7                       # rank-nullity
        assert rank(A) == np.linalg.matrix_rank(A)         # check with NumPy
        C = col_space_basis(A)
        assert C.shape[1] == rank(A) == np.linalg.matrix_rank(C)
    print("\n20 random 5x7 matrices: A v = 0 for v in the null space basis, "
          "rank + nullity = 7, rank agrees with np.linalg.matrix_rank")

    # A 6x8 matrix of rank 4
    A = rng.normal(size=(6, 4)) @ rng.normal(size=(4, 8))
    N = null_space_basis(A)
    print("\n6x8 matrix = (6x4)(4x8): rank =", rank(A),
          " dim null space =", len(N),
          " np.linalg.matrix_rank =", np.linalg.matrix_rank(A))
    assert rank(A) == 4 and len(N) == 4
    for v in N:
        assert np.allclose(A @ v, 0)

# Случайные матрицы почти всегда имеют максимально возможный ранг, поэтому
# rank 5 для случайной 5x7 матрицы — ожидаемый результат. Матрица (6x4)(4x8)
# имеет ранг не больше 4, потому что её столбцы лежат в образе 6x4-множителя.
# Наш rref с допуском tol находит ровно 4 опорных столбца: остатки в
# "нулевых" строках имеют порядок 1e-15 < tol, и rref считает их нулями.