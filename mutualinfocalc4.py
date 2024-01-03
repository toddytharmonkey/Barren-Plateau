
from VQEMonteCarloZeroVariant import *
"""
which represent our old mutual information code that calculates the mutual information over random theta, for one configuration of measurement gates, ran with the following script: 
"""

if __name__ == "__main__":

    client = Client()

    n_a = 400
    n_qubits = 6 
    n_layers = 60
    probs = [.05] 

    overall_results = []

    for p in probs: 

        print(measurements)

        measurements = random_measurements_prob(n_layers,n_qubits,p)
        layered_results = mutual_information_change_all_parameters(n_qubits,n_layers,n_a, measurements)
        overall_results.append(layered_results)
        np.save(f"{n_qubits}_{p}_layeredresults", layered_results)

    plt.plot(overall_results)
    plt.show()

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