from VQEMonteCarlo import *

if __name__ == "__main__":
    results = np.load("HEA2_xxz_results.npy")

    for i in range(1):
        unaware_variance = results[0, :, i, 0, 0, -1]
        aware_variance = results[0, :, i, 0, 1, -1]
        unaware_error = results[0, :, i, 1, :, -1].T
        aware_error = results[0, :, i, 2, :, -1].T

        plt.errorbar(
            [(0.05) * i for i in range(20)],
            aware_variance,
            yerr=(aware_variance - aware_error[0], aware_error[1] - aware_variance),
            marker="o",
            label=6 + 2 * i,
        )

    plt.yscale("log")
    plt.title("aware variance, HEA2, 16 layers, XXZ hamiltonian")
    plt.xlabel("Probability to Place Gate")
    plt.ylabel("aware Variance")
    plt.legend(title="qubits")
    plt.show()

    for i in range(1):
        aware_variance = results[0, 0, i, 0, 1, :]
        unaware_error = results[0, 0, i, 1, :]
        aware_error = results[0, 0, i, 2, :]

        plt.errorbar(
            [i for i in range(16)],
            aware_variance,
            yerr=(aware_variance - aware_error[0], aware_error[1] - aware_variance),
            marker="o",
            linestyle="",
        )

    plt.yscale("log")
    plt.title("#############")
    plt.xlabel("Layer Index")
    plt.ylabel("unaware Variance")

    plt.show()
