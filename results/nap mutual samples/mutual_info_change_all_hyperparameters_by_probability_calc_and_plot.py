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
    qubits = [4,6,8,10,12,14,16]
    n_layers = 60
    probs = [0,.05,.1,.2,.3,.5,.7,.9]

    for examined_layer in [4,-1]:

        for i, n_qubits in enumerate(qubits):

            results_for_each_p = []
            er_each_p = []

            for j, p in enumerate(probs):
                # Load your data based on n_qubits and p
                if p in [0, .05, .1] and n_qubits in [12,14,16]:
                    p_i_m_given_thetas = np.load(f"{n_qubits}_{p}_layeredresults_samples_nap_10000.npy")
                    mean, error = mutual_info_standard_error(p_i_m_given_thetas)
                elif p in [.2,.3,.5] and n_qubits == 16:
                    p_i_m_given_thetas = np.load(f"{n_qubits}_{p}_layeredresults_samples_nap_10000.npy")
                    mean, error = mutual_info_standard_error(p_i_m_given_thetas)
                elif n_qubits == 10:
                    p_i_m_given_thetas = np.load(f"{n_qubits}_{p}_layeredresults_samples_nap_10000.npy")
                    mean, error = mutual_info_standard_error(p_i_m_given_thetas)
                elif p == 0:
                    mean, error = np.load(f"{n_qubits}_{p}_layeredresults.npy")
                elif n_qubits == 12 or n_qubits == 14:
                    p_i_m_given_thetas = np.load(f"{n_qubits}_{p}_layeredresults_samples_changeboth_1000.npy")
                    mean, error = mutual_info_standard_error(p_i_m_given_thetas)
                else:
                    p_i_m_given_thetas = np.load(f"{n_qubits}_{p}_layeredresults_samples_changeboth.npy")
                    mean, error = mutual_info_standard_error(p_i_m_given_thetas)

                results_for_each_p.append(mean[examined_layer])
                er_each_p.append(error[examined_layer])

            plt.errorbar(x=probs, y=results_for_each_p, yerr = er_each_p, label = f"{n_qubits}", marker='.')

        plt.xlabel("Probability")
        plt.ylabel("Mutual information")
        plt.legend(title="number of qubits")
        plt.yscale('log') 
        plt.title(f'Mutual info vs probability at {examined_layer} layers')
        plt.savefig(f"probability_at_layer_{examined_layer}.png")
        plt.clf()