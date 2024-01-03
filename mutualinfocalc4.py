from MILT_mutual_information import * 
"""
This version of the code calculates mutual info, first averaged over n_a thetas, then averaged over n_a different measurement configurations. 
"""

if __name__ == "__main__":

    client = Client()

    n_a = 100
    n_qubits = 6 
    n_layers = 60
    probs = [.1,.2,.3] 

    overall_results = []

    for p in probs: 

        measurements = random_measurements_prob(n_layers, n_qubits, 0)

        layered_results = mutual_information_change_all_parameters(n_qubits,n_layers,n_a, measurements)

        overall_results.append(layered_results)
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