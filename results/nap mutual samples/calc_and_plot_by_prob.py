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
    probs = [0,.05,.1,.2,.24,.26,.28,.3,.5,.7,.9]

    # results are in shape (num_qubits, num_prob,n_layers,2)
    results = np.load("aggregated_data.npy")

    for examined_layer in range(n_layers):
        for i, n_qubits in enumerate(qubits):

            results_for_each_p = []
            er_each_p = []

            for j, p in enumerate(probs):

                mean = results[i,j,:,0]
                error = results[i,j,:,1]
                
                # Load your data based on n_qubits and p
                results_for_each_p.append(mean[examined_layer])
                er_each_p.append(error[examined_layer])

            plt.errorbar(x=probs, y=results_for_each_p, yerr = er_each_p, label = f"{n_qubits}", marker='.')

        plt.xlabel("Probability")
        plt.ylabel("Mutual information")
        plt.legend(title="number of qubits")
        plt.yscale('log') 
        plt.title(f'Mutual info vs probability at {examined_layer} layers')
        plt.savefig(f"examined_layer_images/probability_at_layer_{examined_layer}.png")
        plt.clf()
