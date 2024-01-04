from VQEMonteCarlo import *

"""
Run this file to collect our data for the barren plateaus optimization runs. This file produced all of the files in the folder .01probruns.
"""

if __name__ == "__main__":
    print("Optimizations running!")

    ansatz = "HEA2"
    n_qubits = 8
    n_layers = 16
    n_shots = 1  # change this line
    post_selected = True  # change this line
    parallel = False
    gradient = "aware"

    if not os.path.exists("thetas1.npy"):
        thetas = [
            random_parameters(num_parameters(n_qubits, n_layers, ansatz))
            for _ in range(10)
        ]
        print("thetas1", thetas)
        np.save("thetas1", thetas)
    else:
        print("'thetas1.npy' already exists. Not overwriting.")
        thetas = np.load("thetas1.npy")

    for probability in tqdm([0, 0.2, 0.5], desc="probability"):
        ham_type = "z0z1"

        measurements = random_measurements_prob(n_layers, n_qubits, probability)

        dir_name = f"prob_{probability}"

        multiple_optimization_runs(
            ansatz,
            n_qubits,
            n_layers,
            measurements,
            n_shots,
            post_selected,
            dir_name,
            parallel,
            ham_type,
            gradient,
            thetas,
        )
