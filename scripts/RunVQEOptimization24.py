from VQEMonteCarlo import *

"""
Run this file to collect our data for the barren plateaus optimization runs. 
"""

if __name__ == "__main__":
    qubit_range = [6]  # also change this to whatever you want
    n_layers = 60
    n_samples = 5000
    # change this to whatever you want
    probability_range = [i * 0.05 for i in range(20)]
    ansatz = "HEA2"
    # ham_type = "z0z1"
    # comment out above and use the below cell when you want ot use the XXZ
    ham_type = "xxz_1_1_05"
    file_name = "6qubits"

    # this saves some files to your harddrive, namely the big .npy files
    # note we are currently generating 67% confidence intervals
    generate_results(
        qubit_range,
        n_layers,
        n_samples,
        probability_range,
        ansatz,
        file_name,
        ham_type,
        parallel=False,
    )
