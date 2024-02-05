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
    qubits = [4,6,8,10,12,14]
    n_layers = 60
    probs = [.05,.1,.2,.3,.5]

    for i, n_qubits in enumerate(qubits):

        results_for_each_p = []
        er_each_p = []

        for j, p in enumerate(probs):

            if n_qubits == 12 or n_qubits == 14:
                p_i_m_given_thetas = np.load(f"{n_qubits}_{p}_layeredresults_samples_changeboth_1000.npy")
            else:
                p_i_m_given_thetas = np.load(f"{n_qubits}_{p}_layeredresults_samples_changeboth.npy")


            mean, error = mutual_info_standard_error(p_i_m_given_thetas)

            results_for_each_p.append(mean[-1])
            er_each_p.append(error[-1])

        plt.errorbar(x=probs, y=results_for_each_p, yerr = er_each_p, label = f"{n_qubits}", marker='.')

    plt.xlabel("Probability")
    plt.ylabel("Mutual information")
    plt.legend()
    plt.yscale('log') 
    plt.show()