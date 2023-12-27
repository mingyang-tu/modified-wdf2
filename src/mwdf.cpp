#include "mwdf.h"

inline void fft1d(vcd1d& x, vcd1d& y, int N) {
    fftw_complex* in = reinterpret_cast<fftw_complex*>(x.data());
    fftw_complex* out = reinterpret_cast<fftw_complex*>(y.data());
    fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
    for (int i = 0; i < N; i++)
        y[i] /= sqrt(N);
}

inline void ifft1d(vcd1d& x, vcd1d& y, int N) {
    fftw_complex* in = reinterpret_cast<fftw_complex*>(x.data());
    fftw_complex* out = reinterpret_cast<fftw_complex*>(y.data());
    fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
}

namespace ModifiedWDF {

// x: input signal, t: time vector, f: frequency vector
// dt: time step, df: frequency step
// window: frequency window
// return: 2D vector of complex numbers
vcd2d mwdf2(const vcd1d& x, const vector<double>& t, const vector<double>& f, double dt, double df,
            const vcd1d& window) {
    if (t.size() != x.size())
        throw invalid_argument("t and x must have the same size");
    if (window.size() % 2 != 1)
        throw invalid_argument("window must have odd size");

    const int T = t.size();
    const int F = f.size();
    const int N = int(1 / dt / df / 2);
    const int len_FFT = int(1 / dt / df);

    const int B = window.size() / 2;
    const int Q = B / 2;

    const int f1 = int(f[0] / df), f2 = f1 + F - 1;
    const int Qmax = N / 2 - 1;
    if (f1 < -Qmax || f2 > Qmax)
        throw invalid_argument("frequency out of range");

    vcd1d x_pad(len_FFT, 0);
    for (int i = 0; i < T; i++)
        x_pad[i] = x[i];
    vcd1d X(len_FFT);
    fft1d(x_pad, X, len_FFT);

    const int len_FFT_low = 2 * Qmax + 1 + 2 * Q;
    vcd1d X_low(len_FFT_low, 0);
    int idx = Q;
    for (int i = Qmax; i >= 1; i--) {
        X_low[idx] = X[len_FFT - i];
        idx++;
    }
    for (int i = 0; i <= Qmax; i++) {
        X_low[idx] = X[i];
        idx++;
    }
    vcd1d X_conj_low(len_FFT_low);
    for (int i = 0; i < len_FFT_low; i++)
        X_conj_low[i] = conj(X_low[i]);

    vcd1d w;
    for (int i = B % 2; i < window.size(); i += 2)
        w.push_back(window[i]);

    vcd2d output(F, vcd1d(T));
    complex<double> bias = complex<double>(0, 1) * (-2.0 * M_PI * Q / N);

    for (int m = f1; m <= f2; m++) {
        vcd1d auto_corr(N);
        int start = m + Qmax, end = m + Q * 2 + Qmax + 1;
        for (int i = 0; i < end - start; i++)
            auto_corr[i] = X_low[start + i] * X_conj_low[end - i - 1] * w[i];
        vcd1d idft(N);
        ifft1d(auto_corr, idft, N);
        for (int n = 0; n < T; n++)
            output[m - f1][n] = 2 * df * exp(bias * double(n)) * idft[n];
    }

    return output;
}
}  // namespace ModifiedWDF
