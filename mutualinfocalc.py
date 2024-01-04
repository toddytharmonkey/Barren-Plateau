from MILT_mutual_information import *

"""
This version of the code calculates mutual info, first averaged over n_a thetas, then averaged over n_a different measurement configurations. 
"""

if __name__ == "__main__":
    client = Client()

    n_a = 100
    n_p = 100
    n_qubits = 6
    n_layers = 60
    probs = [0]

    for p in probs:
        if p == 0:
            layered_results = mutual_information_only_parameters(
                n_qubits, n_layers, n_a, measurements=None
            )
            np.save(f"{n_qubits}_{p}_layeredresults", layered_results)

        else:
            layered_results = mutual_info_different_measurements(
                n_qubits, n_layers, n_a, n_p, p
            )
            np.save(f"{n_qubits}_{p}_layeredresults", layered_results)

    client.close()

    # #np.save("layered_resultstest2", np.array(layered_results))

    # #aware
    # print("final results")
    # print(layered_results)
    # plt.plot(layered_results[0])
    # plt.xlabel('layers')
    # plt.ylabel('mutual information')
    # plt.show()

    # #unaware
    # plt.plot(layered_results[1])
    # plt.xlabel('layers')
    # plt.ylabel('mutual information')
    # plt.show()
