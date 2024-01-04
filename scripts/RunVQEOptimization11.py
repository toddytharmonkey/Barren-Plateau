from VQEMonteCarlo import *

"""
Run this file to collect our data for the barren plateaus optimization runs. This file produced all of the files in the folder .01probruns.
"""

if __name__ == "__main__":
    qubit_range = [8]
    n_layers = 16
    n_samples = 1000
    probability_range = [i * 0.05 for i in range(14)]
    ansatz = "HEA2"
    file_name = "HEA2_xxz_results"

    generate_results(
        qubit_range, n_layers, n_samples, probability_range, ansatz, file_name
    )
