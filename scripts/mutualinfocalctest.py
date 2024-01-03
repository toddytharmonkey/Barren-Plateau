
from VQEMonteCarlo import *
"""
Run this file to collect our data for the barren plateaus optimization runs. 
"""

if __name__ == "__main__":

    client = Client()

    n_p = 1
    n_a = 100
    n_c = 100
    n_qubits = 4 
    n_layers = 100
    p = .1

    measurements = random_measurements_prob(n_layers,n_qubits,p)
    layered_results = mutual_information_change_all_parameters(n_qubits,n_layers,measurements, n_a,n_c)

    plt.plot(layered_results)
    plt.show()

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