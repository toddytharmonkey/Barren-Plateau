from VQEMonteCarlo import *
from matplotlib import cm

"""
Run this file to collect our data for the barren plateaus optimization runs.
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

    thetas = [thetas[0]]

    probabilities = [0.80]

    measurements = [
        random_measurements_prob(n_layers, n_qubits, probability)
        for probability in probabilities
    ]
    print("measurements printing")
    print(measurements)

    for probability, measurements in tqdm(zip(probabilities, measurements)):
        ham_type = "z0z1"

        dir_name = f"new_prob_{probability}"

        results = multiple_optimization_runs(
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

        for run_i, result in enumerate(results):
            parameters = result[0]

            N = 100
            n_param = len(parameters)
            scale = 0.1

            v1 = np.random.rand(n_param)
            v2 = np.random.rand(n_param)
            v1 = (v1 - 0.5) * scale
            v2 = (v2 - 0.5) * scale

            C = np.zeros((N, N))
            X = np.zeros((N, N))
            Y = np.zeros((N, N))

            for i in tqdm(range(N)):
                for j in tqdm(range(N), leave=False):
                    x = parameters + (i - N / 2) * v1 + (j - N / 2) * v2
                    X[i, j] = (i - N / 2) / scale
                    Y[i, j] = (j - N / 2) / scale
                    # Replace the next line with your actual function call
                    result = gradients_by_layer(
                        n_qubits,
                        n_layers,
                        x,
                        gradient_technique="analytic",
                        measurements=measurements,
                        return_analytic_suite=True,
                        post_selected=post_selected,
                        periodic=True,
                        get_layered_results=False,
                        ham_type=ham_type,
                        ansatz=ansatz,
                        rotations=None,
                    )
                    # print(f"\r {result}", end="")
                    C[i, j] = result[0]

            # Plotting
            fig = plt.figure()
            ax = plt.axes(projection="3d")
            ax.plot_surface(X, Y, C, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            plt.savefig(
                f"landscape_z0z1_{probability}_{run_i}.pdf", transparent=True, dpi=500
            )

            np.save(f"XYC_z0z1_{probability}_{run_i}.npy", np.stack((X, Y, C), axis=0))
