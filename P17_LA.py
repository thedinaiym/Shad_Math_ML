""" 
P17. Convolutions and the FFT. Write a function circulant(c) that returns the matrix circ(c)
of Definition 10.36.
• For random c, x ∈ R
8
, check that the product circulant(c) @ x agrees with the inverse
DFT np.fft.ifft(np.fft.fft(c) * np.fft.fft(x)).real (NumPy’s fft computes
xˆ = Fx), and that the eigenvalues of circulant(c) are the entries of np.fft.fft(c),
up to their order.
• Time both ways of computing c ∗x for n = 212 (without counting the construction of the
matrix). Which is faster, and by how much?
• Blur a signal x ∈ R
101 of your choice with the filter c = 4
1
(2, 1, 0, . . . , 0, 1)⊤, add noise
of size 10−3
, and try to recover x by dividing the DFT of the result by cˆ. Plot |cˆm| and
explain why the recovered signal is so noisy. (Lecture 2 explains this with the condition
number of circ(c).)

"""
import numpy as np
rng = np.random.default_rng(0)  
import time
import matplotlib.pyplot as plt


def circulant(c):
    """circ(c) (Definition 10.36): the first column is c, every next column is
    the previous one shifted cyclically down by one: circ(c)_ij = c_{(i-j) mod n}.
    Then circ(c) @ x is the cyclic convolution c * x."""
    c = np.asarray(c)
    n = len(c)
    i = np.arange(n)
    return c[(i[:, None] - i[None, :]) % n]


def conv_fft(c, x):
    """c * x via the DFT: ifft(fft(c) * fft(x))."""
    return np.fft.ifft(np.fft.fft(c) * np.fft.fft(x)).real


if __name__ == "__main__":
    # 1. circ(c) @ x  vs  inverse DFT of fft(c) * fft(x); eigenvalues
    c = rng.normal(size=8)
    x = rng.normal(size=8)
    C = circulant(c)
    print("circulant(c) =\n", np.round(C, 2))
    assert np.allclose(C @ x, conv_fft(c, x))
    print("circulant(c) @ x == ifft(fft(c) * fft(x)).real: OK")

    eig = np.linalg.eigvals(C)
    c_hat = np.fft.fft(c)
    # same multiset (up to order): each value of one list is in the other
    assert all(np.abs(eig - lam).min() < 1e-10 for lam in c_hat)
    assert all(np.abs(c_hat - lam).min() < 1e-10 for lam in eig)
    print("eigenvalues of circulant(c) == entries of fft(c) (up to order): OK")
    # the m-th Fourier vector f_m = (w^{mj})_j, w = e^{2 pi i / n}, is an eigenvector
    n = 8
    j = np.arange(n)
    for m in range(n):
        f = np.exp(2j * np.pi * m * j / n)
        assert np.allclose(C @ f, c_hat[m] * f)
    print("C f_m = c_hat_m f_m for the Fourier vectors f_m: OK")

    # 2. timing for n = 2^12
    n = 2 ** 12
    c = rng.normal(size=n)
    x = rng.normal(size=n)
    C = circulant(c)                              # construction not timed

    reps = 20
    t0 = time.perf_counter()
    for _ in range(reps):
        y1 = C @ x
    t_mat = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        y2 = conv_fft(c, x)
    t_fft = (time.perf_counter() - t0) / reps
    assert np.allclose(y1, y2)
    print(f"\nn = 2^12: circulant(c) @ x  {t_mat * 1e3:8.3f} ms")
    print(f"          FFT                 {t_fft * 1e3:8.3f} ms   -> FFT is ~{t_mat / t_fft:.0f}x faster")

    # 3. deblurring
    n = 101
    t = np.arange(n)
    x = np.where((t >= 30) & (t < 60), 1.0, 0.0) + 0.5 * np.sin(2 * np.pi * t / n)
    c = np.zeros(n)
    c[0], c[1], c[-1] = 2 / 4, 1 / 4, 1 / 4     # c = (1/4)(2, 1, 0, ..., 0, 1)
    blurred = circulant(c) @ x
    noisy = blurred + 1e-3 * rng.normal(size=n)
    c_hat = np.fft.fft(c)
    x_rec = np.fft.ifft(np.fft.fft(noisy) / c_hat).real
    x_rec_clean = np.fft.ifft(np.fft.fft(blurred) / c_hat).real

    print(f"\nblur filter: min |c_hat| = {np.abs(c_hat).min():.2e} (at m = "
          f"{np.argmin(np.abs(c_hat))}),  max |c_hat| = {np.abs(c_hat).max():.2f}")
    print(f"condition number of circ(c) = max|c_hat| / min|c_hat| = "
          f"{np.abs(c_hat).max() / np.abs(c_hat).min():.0f}")
    print(f"recovery error without noise: {np.abs(x_rec_clean - x).max():.1e}")
    print(f"recovery error with noise 1e-3: {np.abs(x_rec - x).max():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    m = np.arange(n)
    axes[0].semilogy(m, np.abs(c_hat), ".-")
    axes[0].set_title("|c_hat_m| = cos^2(pi m / n)")
    axes[0].set_xlabel("m")
    axes[1].plot(t, x, label="original x", lw=2)
    axes[1].plot(t, noisy, label="blurred + noise 1e-3")
    axes[1].plot(t, x_rec, label="recovered: ifft(fft(y) / c_hat)", alpha=0.8)
    axes[1].legend()
    axes[1].set_title("Deconvolution by dividing by c_hat")
    fig.tight_layout()
    fig.savefig("P17_deconvolution.png", dpi=130)
    print("saved P17_deconvolution.png")
    plt.show()

# Почему FFT быстрее. Умножение матрицы n x n на вектор стоит n^2 = 1.7e7
# операций, а три FFT стоят ~3 n log2 n ~ 1.5e5. Даже с учётом того, что
# BLAS очень эффективен, FFT выигрывает в десятки раз (см. вывод выше).
#
# Почему восстановленный сигнал такой шумный.
# Для c = (1/4)(2, 1, 0, ..., 0, 1):
#     c_hat_m = (2 + e^{-2 pi i m/n} + e^{2 pi i m/n}) / 4 = (1 + cos(2 pi m/n)) / 2
#             = cos^2(pi m / n).
# Это фильтр низких частот: высокие частоты (m ~ n/2) он почти уничтожает.
# При n = 101 минимум |c_hat_m| = sin^2(pi/202) ~ 2.4e-4 (n нечётно, поэтому
# точного нуля нет). При делении на c_hat шум 1e-3 на этих частотах
# умножается примерно на 1/2.4e-4 ~ 4000. Ошибка восстановления ~0.4,
# сравнима с самим сигналом, и в нём видна сильная высокочастотная пила.
# Без шума деление восстанавливает x почти точно (ошибка ~1e-13).
# На языке Лекции 2: circ(c) — нормальная матрица с собственными значениями
# c_hat_m, её число обусловленности равно max|c_hat| / min|c_hat| ~ 4000.
# Относительная ошибка данных 1e-3 может вырасти до ~4000 * 1e-3 ~ 4.