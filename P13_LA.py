""" 
P13. Softmax and cosets. Implement the softmax function in two ways:
def softmax_naive(z):
return np.exp(z) / np.exp(z).sum()
def softmax_stable(z):
z = z - z.max() # another representative of the same coset
return np.exp(z) / np.exp(z).sum()
• For random vectors z and constants c, check that softmax_stable(z + c) equals
softmax_stable(z), i.e., that softmax only depends on the coset z + span{1}.
• Compare both functions on z = (1, 2, 3) and on z = (1001, 1002, 1003). What happens
with the naive version, and why does subtracting the maximum fix the problem without
changing the result?
• Another canonical representative of the coset is the centered vector z − z¯1, where z¯ is
the mean of the entries. Check that it gives the same softmax. Why is subtracting the
maximum preferred in practice? (Hint: what is the largest entry after each shift?)
"""
import numpy as np
rng = np.random.default_rng(0) 


def softmax_naive(z):
    return np.exp(z) / np.exp(z).sum()


def softmax_stable(z):
    z = z - z.max()  # another representative of the same coset
    return np.exp(z) / np.exp(z).sum()


def softmax_centered(z):
    z = z - z.mean()  # the centered representative z - mean(z) * 1
    return np.exp(z) / np.exp(z).sum()


if __name__ == "__main__":
    # softmax depends only on the coset z + span{1}
    for _ in range(5):
        z = rng.normal(size=6)
        c = rng.normal() * 10
        assert np.allclose(softmax_stable(z + c), softmax_stable(z))
        assert np.allclose(softmax_centered(z), softmax_stable(z))
    print("softmax_stable(z + c) == softmax_stable(z) and the centered "
          "representative gives the same softmax: OK")

    for z in [np.array([1.0, 2.0, 3.0]), np.array([1001.0, 1002.0, 1003.0])]:
        with np.errstate(over="ignore", invalid="ignore"):
            naive = softmax_naive(z)
            print(f"\nz = {z}:  exp(z) = {np.exp(z)}")
        print(f"   naive    = {naive}")
        print(f"   stable   = {softmax_stable(z)}")
        print(f"   centered = {softmax_centered(z)}")

    # when does the centered representative fail?
    z = np.array([0.0] * 9 + [1000.0])
    with np.errstate(over="ignore", invalid="ignore"):
        print(f"\nz = (0, ..., 0, 1000) (10 entries):")
        print("   max entry after subtracting max  :", (z - z.max()).max())
        print("   max entry after subtracting mean :", (z - z.mean()).max())
        print("   centered softmax =", softmax_centered(z))
    print("   stable softmax   =", softmax_stable(z))

# Почему наивная версия ломается.
# exp(1001) ~ 1e435, а double вмещает числа до ~1.8e308 (exp(709.8)).
# Поэтому exp(z) = inf, inf / inf = nan, и softmax_naive возвращает nan.
#
# Почему вычитание максимума помогает и не меняет ответ.
# softmax(z + c 1)_i = e^{z_i + c} / sum_j e^{z_j + c}
#                    = e^c e^{z_i} / (e^c sum_j e^{z_j}) = softmax(z)_i.
# Значит, softmax — функция на классах смежности z + span{1} (на фактор-
# пространстве R^n / span{1}), и можно выбрать любого представителя класса.
# z - max(z) — такой представитель. У него все элементы <= 0, поэтому
# exp <= 1 и переполнения нет. Максимальный элемент равен ровно 0, поэтому
# в знаменателе есть слагаемое e^0 = 1, знаменатель >= 1, и нет ни деления
# на 0, ни 0/0.
# Для z = (1001, 1002, 1003) получаем z - max = (-2, -1, 0), тот же ответ,
# что и для (1, 2, 3).
#
# Центрированный представитель z - mean(z) 1 тоже из того же класса и даёт
# тот же softmax (если вычисление не переполнится). Но гарантии нет:
# наибольший элемент после центрирования равен max - mean, а это может быть
# очень много (для (0, ..., 0, 1000) это 900, exp(900) = inf). После вычитания
# максимума наибольший элемент всегда ровно 0, поэтому так и делают на практике.