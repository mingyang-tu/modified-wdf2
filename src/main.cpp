#include "matplotlibcpp.h"
#include "mwdf.h"
#include "timer.h"

#define JJ complex<double>(0, 1)

namespace plt = matplotlibcpp;

using namespace std;

vector<int> get_ticks(int start, int end, int num_ticks) {
    vector<int> ticks;
    int step = (end - start) / (num_ticks - 1);
    for (int i = start; i <= end; i += step)
        ticks.push_back(i);
    return ticks;
}

vector<string> get_ticklabels(double start, double end, int num_ticks) {
    vector<string> ticklabels;
    double step = (end - start) / (num_ticks - 1);
    for (double i = start; i <= end; i += step) {
        stringstream ss;
        ss << i;
        ticklabels.push_back(ss.str());
    }
    return ticklabels;
}

// visualize the result
void visualize(vector<vector<complex<double>>> result, double t_min, double t_max, double f_min,
               double f_max) {
    int nrows = result.size(), ncols = result[0].size();
    vector<float> image(ncols * nrows);
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            image[ncols * i + j] = abs(result[i][j]);
        }
    }

    const float* image_ptr = &(image[0]);

    plt::title("Modified Wigner Distribution Function (Form 2)");
    plt::imshow(image_ptr, nrows, ncols, 1, {{"cmap", "gray"}, {"origin", "lower"}, {"aspect", "auto"}});
    plt::xlabel("Time");
    plt::ylabel("Frequency");

    int num_ticks_x = 5, num_ticks_y = 5;
    vector<int> xticks = get_ticks(0, ncols, num_ticks_x), yticks = get_ticks(0, nrows, num_ticks_y);
    vector<string> xticklabels = get_ticklabels(t_min, t_max, num_ticks_x),
                   yticklabels = get_ticklabels(f_min, f_max, num_ticks_y);
    plt::xticks(xticks, xticklabels);
    plt::yticks(yticks, yticklabels);
    plt::show();
}

vector<complex<double>> gaussian_window(double sigma, double dt) {
    int Q = static_cast<int>(1.9143 / std::sqrt(sigma) / dt);
    vector<double> x(2 * Q + 1);
    vector<complex<double>> result(2 * Q + 1);

    for (int i = -Q; i <= Q; i++)
        x[i + Q] = i * dt;

    for (int i = 0; i < x.size(); i++)
        result[i] = sqrt(sqrt(sigma)) * exp(-sigma * M_PI * x[i] * x[i]);

    return result;
}

int main() {
    double dt = 0.0125, df = 0.0125;
    double t_start = 0, t_end = 10;
    double f_start = -5, f_end = 5;
    const int T = static_cast<int>((t_end - t_start) / dt) + 1;
    const int F = static_cast<int>((f_end - f_start) / df) + 1;

    vector<double> t(T);
    for (int i = 0; i < T; i++)
        t[i] = t_start + i * dt;
    vector<complex<double>> x(T);
    for (int i = 0; i < T; i++)
        x[i] = cos(2 * M_PI * t[i]);
    vector<double> f(F);
    for (int i = 0; i < F; i++)
        f[i] = f_start + i * df;
    vector<complex<double>> window = gaussian_window(5, dt);
    cout << "window size: " << window.size() << endl;

    Timer clock;
    clock.tick();

    vector<vector<complex<double>>> result = ModifiedWDF::mwdf2(x, t, f, dt, df, window);

    clock.tock();
    cout << "Elapsed time: " << clock.duration() << " s" << endl;

    visualize(result, t_start, t_end, f_start, f_end);

    return 0;
}
