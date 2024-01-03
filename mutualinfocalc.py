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
    probs = [.05] 

    overall_results = []

    for p in probs: 

        layered_results = mutual_info_different_measurements(n_qubits,n_layers,n_a, n_p, p)
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