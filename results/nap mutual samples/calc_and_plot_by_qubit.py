import numpy as np 
import matplotlib.pyplot as plt
import sys
import distinctipy
sys.path.insert(0, '../..')
from MILT_mutual_information import *


"""
This version of the code calculates and plots mutual info from samples and displays them on a plot of mutual information VS qubits.  
"""

if __name__ == "__main__":

    n_ap = 1000
    qubits = [4,6,8,10,12,14,16]
    n_layers = 60
    probs = [0,.05,.1,.2,.3,.5,.7,.9]

    # Create a figure for the plot
    plt.figure(figsize=(10, 7))

    num_colors = len(probs)
    colors = distinctipy.get_colors(num_colors)

    for j, p in enumerate(probs):
        mutual_infos = []
        errors = []

        for i, n_qubits in enumerate(qubits):
            examined_layer = n_qubits
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

            # Extract the fifth element for the fifth layer
            if len(mean) >= 5:  # Check if the fifth element exists
                mutual_infos.append(mean[examined_layer])
                errors.append(error[examined_layer])
            else:
                mutual_infos.append(np.nan)  # Append NaN if the data is not available
                errors.append(np.nan)

        # Plot mutual information vs. n_qubits for the current probability
        plt.errorbar(qubits, mutual_infos, yerr=errors, label=f"p = {p}", color=colors[j], marker='o')

        plt.xlabel("Number of Qubits")
        plt.ylabel("Mutual Information")
        plt.title(f"Mutual Information vs Number of Qubits at Layer index n")
        plt.yscale('log')
        plt.legend()
        plt.savefig(f"qubits_at_layer_{examined_layer}.png")
        plt.clf()
