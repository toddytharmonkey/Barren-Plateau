from VQEMonteCarlo import *

"""
Run this file to collect our data for the barren plateaus optimization runs. 
"""

if __name__ == "__main__":
    client = Client()

    n_qubits = 4
    n_layers = 50

    measurements = random_measurements_prob(n_layers, n_qubits, 0)

    results = []
    for _ in tqdm(range(40)):
        results.append(
            mutual_information_change_all_parameters(n_qubits, n_layers, n_a=1000)
        )

    np.save("layered_results_testtest", results)

    averaged_results = np.mean(results, axis=0)
    print(averaged_results)
    plt.plot(averaged_results)
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
