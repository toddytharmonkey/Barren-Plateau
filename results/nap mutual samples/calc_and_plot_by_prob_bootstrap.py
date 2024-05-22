import numpy as np 
import matplotlib.pyplot as plt
import sys
import distinctipy
sys.path.insert(0, '../..')
from MILT_mutual_information import *


"""
This version of the code calculates and plots mutual info from samples. 
"""

if __name__ == "__main__":
    n_ap = 1000
    qubits = [4,6,8,10,12,14,16,18]
    n_layers = 60
    probs = [0,.05,.1,.2,.3,.5,.7,.9]

    # results are in shape (num_qubits, num_prob,n_layers,2)
    results = np.load("aggregated_data_bootstrap.npy")

    for i, n_qubits in enumerate(qubits):

        results_for_each_p = []
        er_each_p = []

        examined_layer = 2*n_qubits

        for j, p in enumerate(probs):

            mean = results[i,j,0]
            low = results[i,j,1]
            high = results[i,j,2]

            print("mean, low, high")
            print(mean,low,high)
            print("error bar sizes")
            print(mean-low,high-mean)
            
            # Load your data based on n_qubits and p
            results_for_each_p.append(mean)
            er_each_p.append((mean-low,high-mean))

        yerr = np.array([[abs(err[0]) for err in er_each_p],
                          [err[1] for err in er_each_p]])

        print("probs",probs)
        print("results for each p", results_for_each_p)
        print("yerr", er_each_p)
        plt.errorbar(x=probs, y=results_for_each_p, yerr = yerr, label = f"{n_qubits}", marker='.')

    plt.xlabel("Probability")
    plt.ylabel("Mutual information")
    plt.legend(title="number of qubits")
    plt.yscale('log') 
    plt.title(f'Mutual info vs probability at 2n layers')
    plt.savefig(f"probability_at_layer_2n_bootstrap.png")
    plt.clf()
