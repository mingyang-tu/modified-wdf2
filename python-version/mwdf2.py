import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import time


def mwdf2(x, t, f, dt, df, window):
    assert len(x) == len(t)
    assert len(window) % 2 == 1
    T = len(t)
    F = len(f)
    N = int(1 / dt / df / 2)

    B = len(window) // 2
    Q = B // 2

    f1 = int(f[0] / df)
    f2 = f1 + F - 1
    Qmax = N // 2 - 1
    assert -Qmax <= f1 <= f2 <= Qmax

    x_pad = np.zeros(N * 2, dtype=np.complex128)
    x_pad[:T] = x
    X = np.fft.fft(x_pad, norm="ortho")
    X = np.concatenate((X[-Qmax:], X[: Qmax + 1]))
    X = np.pad(X, (Q, Q))
    X_conj = np.conj(X)

    n = np.arange(0, T)
    w = window[B - Q * 2 : B + Q * 2 + 1 : 2]

    output = np.zeros((F, T), dtype=np.complex128)
    for m in range(f1, f2 + 1):
        auto_correlation = np.zeros(N, dtype=np.complex128)
        start = m + Qmax
        end = m + Q * 2 + Qmax + 1
        auto_correlation[: end - start] = X[start:end] * np.flip(X_conj[start:end]) * w
        idft = np.fft.ifft(auto_correlation) * N
        output[m - f1, :] = 2 * df * np.exp(-1j * 2 * np.pi * n * Q / N) * idft[n]
    return output


def gaussian_window(sigma, dt):
    Q = int(1.9143 / sqrt(sigma) / dt)
    x = np.arange(-Q, Q + 1) * dt
    return sqrt(sqrt(sigma)) * np.exp(-sigma * np.pi * x**2)


def visualize(wigner, t, f):
    t_min, t_max = t[0], t[-1]
    f_min, f_max = f[0], f[-1]

    plt.figure()
    plt.imshow(
        np.abs(wigner),
        cmap="gray",
        origin="lower",
        aspect="auto",
        extent=(t_min, t_max, f_min, f_max),
    )
    plt.title("Modified Wigner Distribution Function (Form 2)")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.show()


def main():
    dt = 0.0125
    df = 0.0125

    t = np.arange(0, 10, dt)
    f = np.arange(-5, 5, df)
    x = np.cos(2 * np.pi * t)

    sigma = 5
    window = gaussian_window(sigma, df)
    print(f"Length of window: {len(window)}")

    start = time.time()
    wdf = mwdf2(x, t, f, dt, df, window)
    end = time.time()
    print(f"Elapsed time: {end - start} s")

    visualize(wdf, t, f)


if __name__ == "__main__":
    main()
