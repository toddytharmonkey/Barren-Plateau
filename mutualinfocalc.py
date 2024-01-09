from MILT_mutual_information import *

"""
This version of the code calculates mutual info, first averaged over n_a thetas, then averaged over n_a different measurement configurations. 
"""

if __name__ == "__main__":
    client = Client()

    print("Dask Dashboard URL:", client.dashboard_link)

    n_a = 100
    n_p = 1000
    qubit_list = [12,14,16]
    n_layers = 60
    probs = [.05,.1,.2]

    for n_qubits in qubit_list:
        for p in probs:
            if p == 0:
                layered_results = mutual_information_only_parameters(
                    n_qubits, n_layers, n_a, measurements=None
                )
                np.save(f"{n_qubits}_{p}_layeredresults_1000_samples", layered_results)

            else:
                layered_results = mutual_info_different_measurements(
                    n_qubits, n_layers, n_a, n_p, p
                )
                np.save(f"{n_qubits}_{p}_layeredresults_1000_samples", layered_results)

    client.close()
