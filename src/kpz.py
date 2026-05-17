"""
the goal of this program is to demonstrate the multi species 
exclusion process falls within the the KPZ universality class.
"""

import numba as nb
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

from msep import MultiSpeciesExclusionProcess, jit_update

@nb.njit
def _jit_fourier_time_series(chain, rates_matrix, max_rate, n_samples=60000, species=0, sample_every=1):
    state = chain.copy()
    L = len(state)

    q = 2.0 * np.pi / L

    cos_q = np.empty(L, dtype=np.float64)
    sin_q = np.empty(L, dtype=np.float64)

    for j in range(L):
        cos_q[j] = np.cos(q * j)
        sin_q[j] = np.sin(q * j)

    X = np.empty(n_samples, dtype=np.complex128)

    for n in range(n_samples):
        re = 0.0
        im = 0.0

        for j in range(L):
            if state[j] == species:
                re += cos_q[j]
                im += sin_q[j]

        X[n] = re + 1j * im

        for _ in range(sample_every):
            for _ in range(L):
                jit_update(state, rates_matrix, max_rate)

    return X

"""
fast autocorrelation with fast fourier transform.
"""
def autocorrelation(x):
    signal = np.asarray(x)
    signal -= np.mean(signal)

    n_samples = len(signal)

    raw_ac = sp.signal.correlate(signal, signal, mode="full", method="fft")
    positive_lags_ac = raw_ac[n_samples - 1 :]

    # unbiased normalization
    unbiased_ac = positive_lags_ac / np.arange(n_samples, 0, -1)
    
    return unbiased_ac / unbiased_ac[0]

if __name__ == "__main__":
    L_values = np.arange(30, 300, 3)
    taus = []

    for L in L_values:
        dimension = 3
        density = [1/3, 1/3, 1/3]
        rates = {(0, 1) : 1.0, (1, 0) : 0.0, (0, 2) : 1.0, (2, 0) : 0.0, (1, 2) : 1.0, (2, 1) : 0.0}

        process = MultiSpeciesExclusionProcess(dimension=dimension, density=density, rates=rates, length=L, checkPairwiseBalence=False)

        X = _jit_fourier_time_series(chain=process.chain, rates_matrix=process.rates_matrix, max_rate=process.max_rate)
        C = autocorrelation(X)

        t = np.arange(len(C))
        envelope = np.abs(C)

        # smoothing to reduce monte carlo noise
        window = 7
        kernel = np.ones(window) / window
        envelope_smooth = np.convolve(envelope, kernel, mode="same")

        # time where envelope has decayed significantly
        crossings = np.where((t > 0) & (envelope_smooth < 0.2))[0]

        if len(crossings) > 0:
            end = crossings[0]
        else:
            end = int(min(len(t), 4 * L ** 1.5))

        # fir where decay is neither too early nor too noisy
        mask = (
            (t > 0)
            & (np.arange(len(t)) < end)
            & (envelope_smooth < 0.85)
            & (envelope_smooth > 0.25)
        )

        if np.sum(mask) < 8:
            mask = (
                (t > 0)
                & (np.arange(len(t)) < end)
                & (envelope_smooth < 0.9)
                & (envelope_smooth > 0.15)
            )

        slope, intercept = np.polyfit(t[mask], np.log(envelope_smooth[mask]), 1)
        taus.append(-1.0 / slope)

    taus = np.array(taus)
    logL = np.log(L_values)
    logtau = np.log(taus)        

    z, intercept = np.polyfit(logL, logtau, 1)
    fit = intercept + z * logL

    print(f"z = {z:.3f}")

    plt.figure(figsize=(6, 4))
    plt.plot(logL, logtau, "o", label="monte carlo data")
    plt.plot(logL, fit, "--", label=fr"$z \approx {z:.3f}$")
    
    plt.xlabel(r"$\log L$")
    plt.ylabel(r"$\log \tau(L)$")
    plt.title(r"dynamic critical exponent monte carlo")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("figures/dynamical_critical_exponent_simulation.png", dpi=300)
    plt.show()