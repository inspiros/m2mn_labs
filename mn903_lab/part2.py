import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from entropy_estimator import EntropyEstimator
from quantizer import quant_midrise, quant_midtread, distortion


def main():
    x = np.random.normal(0, 1, 1000)

    qx_mr, idx_mr = quant_midrise(x, 1, m=2 ** 3)
    print(f'Dmr={distortion(x, qx_mr)}')
    print(f'Hmr={EntropyEstimator(idx_mr).entropy()}')

    qx_mt, idx_mt = quant_midtread(x, 1, m=2 ** 3)
    print(f'Dmt={distortion(x, qx_mt)}')
    print(f'Hmt={EntropyEstimator(idx_mt).entropy()}')

    # ---------- plot ----------
    deltas = np.logspace(-2, 1, num=1000)
    Hmr = np.empty_like(deltas)
    Dmr = np.empty_like(deltas)
    Hmt = np.empty_like(deltas)
    Dmt = np.empty_like(deltas)
    for i, delta in enumerate(deltas):
        qx_mr, idx_mr = quant_midrise(x, delta)
        Hmr[i] = EntropyEstimator(idx_mr).entropy()
        Dmr[i] = distortion(x, qx_mr)

        qx_mt, idx_mt = quant_midtread(x, delta)
        Hmt[i] = EntropyEstimator(idx_mt).entropy()
        Dmt[i] = distortion(x, qx_mt)

    plt.figure()
    plt.semilogy(Hmr, Dmr)
    plt.semilogy(Hmt, Dmt)
    plt.xlabel('Entropy (bits/symbol)')
    plt.ylabel('Distortion')
    plt.legend(['midrise', 'midtread'])
    plt.show()

    # ---------- wav ----------
    sr, s = wavfile.read('resources/Alarm05.wav')
    deltas = np.logspace(-1, np.log(501), num=100)
    s_flatten = s.reshape(-1)

    ms = [2, 4, 8, 16]
    plt.figure()
    for plot_id, m in enumerate(ms):
        Hm = np.empty_like(deltas)
        Dm = np.empty_like(deltas)
        for i, delta in enumerate(deltas):
            qs_0, idx_0 = quant_midtread(s[:, 0], delta, m=m)
            qs_1, idx_1 = quant_midtread(s[:, 1], delta, m=m)
            qs = np.concatenate([qs_0.reshape(-1, 1), qs_1.reshape(-1, 1)], axis=1)

            qs_flatten = qs.reshape(-1)
            idx_flatten = np.concatenate([idx_0, idx_1])

            Hm[i] = EntropyEstimator(idx_flatten).entropy()
            Dm[i] = distortion(s_flatten, qs_flatten)

        ax = plt.subplot(2, 2, plot_id + 1)
        ax.semilogy(Hm, Dm)
        ax.set_title(f'm={m}')
        ax.set_xlabel('Entropy (bits/symbol)')
        ax.set_ylabel('Distortion')
    plt.show()

    # wavfile.write(f'resources/Alarm05_{m:02d}_{delta:0.5f}.wav', sr, qs.astype(s.dtype))


if __name__ == '__main__':
    main()
