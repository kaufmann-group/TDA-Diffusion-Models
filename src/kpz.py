"""
the goal of this program is to demonstrate the multi species 
exclusion process falls within the the KPZ universality class.
"""

from msep.msep_cpp import MultiSpeciesExclusionProcess


import numba as nb
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

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

        rates_matrix = np.array(
            [
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        process = MultiSpeciesExclusionProcess(
            dimension=dimension,
            density=density,
            rates_matrix=rates_matrix,
            length=L,
            seed=2504,
            shuffle=True,
            check_pairwise_balance=False,
        )

        X = process.fourier_time_series(
            n_samples=60000,
            species=0,
            sample_every=1,
        )
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