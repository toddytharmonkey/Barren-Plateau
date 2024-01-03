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
    n_shots = 1 #change this line
    post_selected = True #change this line
    parallel=False
    gradient = "aware"

   
# Your existing data
no_measurements = None
point_one_measurement = [[4, 0], [4, 1], [5, 2], [9, 4], [10, 2], [11, 0], [11, 1], [13, 3]]
point_two_measurement = [[0, 0], [0, 2], [1, 0], [2, 1], [3, 3], [4, 2], [6, 0], [6, 4], [7, 0], [7, 1], [7, 2], [7, 5], [9, 2], [10, 0], [10, 7], [11, 1], [11, 6], [11, 7], [13, 0], [14, 0], [14, 3], [14, 6]]

for probability, measurements in ([0,.1,.2], [no_measurements, point_one_measurement, point_two_measurement])):
        ham_type = "z0z1"
        dir_name = f"new_prob_{probability}"

        # Load the data from the file
        file_name = f"new_prob_{probability}_HEA2_q8_l16_shots1_postTrue_z0z1_aware_thetas10_version1.2/all_run_info.npy"
        all_run_data = np.load(file_name, allow_pickle=True)
        # print(all_run_data.shape)

        for run_i, output in enumerate(all_run_data[:2]):

            parameters = output[0]

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
                    result = gradients_by_layer(n_qubits, n_layers, x, gradient_technique="analytic", measurements=measurements, return_analytic_suite=True, post_selected=post_selected,periodic=True,get_layered_results=False, ham_type=ham_type,ansatz=ansatz, rotations=None,)
                    # print(f"\r {result}", end="")
                    C[i, j] = result[0] 

            # Plotting
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.plot_surface(X, Y, C, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            plt.savefig(f'landscape_z0z1_{probability}_{run_i}.pdf',transparent=True, dpi=500)  

            np.save(f"XYC_z0z1_{probability}_{run_i}.npy",np.stack((X,Y,C),axis=0))