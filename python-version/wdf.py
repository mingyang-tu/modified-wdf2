import numpy as np
import matplotlib.pyplot as plt


def wdf(x, t, f, dt, df):
    assert len(x) == len(t)
    T = len(t)
    F = len(f)
    N = int(1 / dt / df / 2)

    f1 = int(f[0] / df)
    f2 = f1 + F - 1
    Qmax = N // 2 - 1
    assert -Qmax <= f1 <= f2 <= Qmax

    x_pad = np.zeros(N * 2, dtype=np.complex128)
    x_pad[:T] = x
    X = np.fft.fft(x_pad, norm="ortho")
    X = np.concatenate((X[-Qmax:], X[: Qmax + 1]))
    X_conj = np.conj(X)

    n = np.arange(0, T)

    output = np.zeros((F, T), dtype=np.complex128)
    for m in range(f1, f2 + 1):
        Q = min(Qmax - m, Qmax + m)
        auto_correlation = np.zeros(N, dtype=np.complex128)
        start = m - Q + Qmax
        end = m + Q + Qmax + 1
        auto_correlation[: end - start] = X[start:end] * np.flip(X_conj[start:end])
        idft = np.fft.ifft(auto_correlation) * N
        output[m - f1, :] = 2 * df * np.exp(-1j * 2 * np.pi * n * Q / N) * idft[n]
    return output


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
    plt.title("Wigner Distribution Function")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.show()


def main():
    dt = 0.0125
    df = 0.0125

    t = np.arange(0, 10, dt)
    f = np.arange(-5, 15, df)
    x = np.exp(1j * (t - 5) ** 3)

    result = wdf(x, t, f, dt, df)

    visualize(result, t, f)


if __name__ == "__main__":
    main()
