
from VQEMonteCarlo import *
"""
Run this file to collect our data for the barren plateaus optimization runs. 
"""

if __name__ == "__main__":

    client = Client()

    n_p = 100
    n_a = 100
    n_c = 100
    qubit_range = [4,6,8,10,12,14,16] 
    n_layers = 40
    probabilities = [0,.05,.1,.2,.3,.4,.5]

    for n_qubits in qubit_range:
        for p in probabilities: 

            measurements = random_measurements_prob(n_layers,n_qubits,p)
            layered_results = overall_mutual_information(n_qubits, n_layers, measurements, n_p,n_a ,n_c , dask=True)

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