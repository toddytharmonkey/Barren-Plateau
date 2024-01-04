from VQEMonteCarlo import *

"""
Run this file to collect our data for the barren plateaus optimization runs. This file produces the count_1,2,3.... files. 
"""

if __name__ == "__main__":
    print("Optimizations running!")

    ansatz = "HEA2"
    n_qubits = 8
    n_layers = 20
    n_shots = 1  # change this line
    post_selected = True  # change this line
    parallel = False
    gradient = "aware"

    # print(even_bisect_measurements(n_layers, n_qubits, 16))

    if not os.path.exists("thetas.npy"):
        thetas = [
            random_parameters(num_parameters(n_qubits, n_layers, ansatz))
            for _ in range(10)
        ]
        print("thetas", thetas)
        np.save("thetas", thetas)
    else:
        print("'thetas.npy' already exists. Not overwriting.")
        thetas = np.load("thetas.npy")

    for measurement_count in tqdm(range(1, 20), desc="count"):
        ham_type = "xxz_1_1_05"

        measurements = even_bisect_measurements(n_layers, n_qubits, measurement_count)

        dir_name = f"count_{measurement_count}"

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
