from VQEMonteCarlo import *
"""
Run this file to collect our data for the barren plateaus optimization runs.
"""

if __name__ == "__main__":
    print("Optimizations running!")

    ansatz = "HEA2"
    n_qubits = 8
    n_layers = 20
    n_shots = 1 #change this line
    post_selected = True #change this line
    parallel=False
    gradient = "aware"
   
    if not os.path.exists("thetas.npy"):
        thetas = [random_parameters(num_parameters(n_qubits, n_layers, ansatz)) for _ in range(10)]
        print("thetas", thetas)
        np.save("thetas", thetas)
    else:
        print("'thetas.npy' already exists. Not overwriting.")
        thetas = np.load("thetas.npy")

    for probability in tqdm(np.arange(.6, 1, 0.1), desc="probability"):

        ham_type = "z0z1"

        measurements = random_measurements_prob(n_layers, n_qubits, probability)

        dir_name = f"prob_{probability}"

        multiple_optimization_runs(ansatz, n_qubits, n_layers, measurements, n_shots, post_selected, dir_name, parallel, ham_type, gradient, thetas)