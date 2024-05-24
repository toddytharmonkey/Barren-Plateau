import numpy as np 
import matplotlib.pyplot as plt
import sys
import distinctipy
sys.path.insert(0, '../..')

"""
This version of the code calculates and plots mutual info from samples. 
"""

if __name__ == "__main__":
    qubit_range = [4]  # also change this to whatever you want
    n_layers = 60
    n_samples = 500
    # change this to whatever you want
    probability_range = [i * 0.05 for i in range(20)]
    ansatz = "HEA2"
    # ham_type = "z0z1"
    # comment out above and use the below cell when you want ot use the XXZ
    ham_type = "xxz_1_1_05"
    file_name = "newresult"

    # results are in shape (num_qubits, num_prob,n_layers,2)
    results = np.load("newresult.npy")

    print(results.shape)

    for i, n_qubits in enumerate(qubit_range):

        results_for_each_p = []
        er_each_p = []

        examined_layer = 59

        for j, p in enumerate(probability_range):

            if j == 0: 
                pass

            mean = results[0,j, i, 0, 1, examined_layer]
            print("mean", mean)
            print("results", results[0,j,i,2,:,examined_layer])
            low = results[0,j,i,2,0,examined_layer]
            high = results[0,j,i,2,1,examined_layer]

            # print("mean, low, high")
            # print(mean,low,high)
            # print("error bar sizes")
            # print(mean-low,high-mean)

            # Load your data based on n_qubits and p
            results_for_each_p.append(mean)
            er_each_p.append((mean-low,high-mean))

        yerr = np.array([[abs(err[0]) for err in er_each_p],
                          [err[1] for err in er_each_p]])

        print("probs",probability_range)
        print("results for each p", results_for_each_p)
        print("yerr", er_each_p)
        plt.errorbar(x=probability_range[1:], y=results_for_each_p[1:], yerr = yerr[:,1:], label = f"{n_qubits}", marker='.')

    plt.xlabel("Probability")
    plt.ylabel("Variance ")
    plt.legend(title="number of qubits")
    plt.yscale('log') 
    plt.title(f'Variance vs probability when putting down random measurements')
    plt.savefig(f"newvariancecalc")
    plt.clf()
