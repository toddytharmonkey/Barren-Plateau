from VQEMonteCarlo import *
from matplotlib import cm

# First run this for measurement free landscape. --------------------------------------------------
results = np.load(r"all_run_info_no_measurement.npy", allow_pickle=True)
parameters = results[0, 0]

ansatz = "HEA2"
n_qubits = 8
n_shots = 1  # change this line
post_selected = True  # change this line
parallel = False
gradient = "aware"
n_layers = 16
ham_type = "xxz_1_1_05"
measurements = None

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
plt.show()

np.save("XYC_no_measurements.npy", np.stack((X, Y, C), axis=0))

# Now, just basically copy/pasting everything for the measurement landscape. ----------------------

results = np.load(r"all_run_info_measurement.npy", allow_pickle=True)

parameters = results[0, 0]

ansatz = "HEA2"
n_qubits = 8
n_shots = 1  # change this line
post_selected = True  # change this line
parallel = False
gradient = "aware"
n_layers = 16
ham_type = "xxz_1_1_05"
measurements = [
    [0, 0],
    [0, 2],
    [1, 0],
    [2, 1],
    [3, 3],
    [4, 2],
    [6, 0],
    [6, 4],
    [7, 0],
    [7, 1],
    [7, 2],
    [7, 5],
    [9, 2],
    [10, 0],
    [10, 7],
    [11, 1],
    [11, 6],
    [11, 7],
    [13, 0],
    [14, 0],
    [14, 3],
    [14, 6],
]

# Assuming 'parameters' is predefined and is a numpy array
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
plt.show()

np.save("XYC_with_measurements.npy", np.stack((X, Y, C), axis=0))
