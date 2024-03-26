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
    probs = [0]

    # version #1 of the code: this does layers vs mutual info for different n_qubits 

    # Generate distinct colors for each n_qubits
    num_colors = len(qubits)
    colors = distinctipy.get_colors(num_colors)

    # Define markers for each probability
    markers = ['.', 'x']  # You can add more markers if you have more probabilities

    # Create subplots: one row for each probability
    fig, axs = plt.subplots(len(probs), 1, figsize=(10, len(probs) * 5))
    if len(probs) == 1:  # If there's only one probability, axs will not be a list
        axs = [axs]

    for j, p in enumerate(probs):
        for i, n_qubits in enumerate(qubits):

            if p == 0:
                mean, error = np.load(f"{n_qubits}_{p}_layeredresults.npy")
            elif n_qubits == 12 or n_qubits == 14:
                p_i_m_given_thetas = np.load(f"{n_qubits}_{p}_layeredresults_samples_changeboth_1000.npy")
                mean, error = mutual_info_standard_error(p_i_m_given_thetas)
            else:
                p_i_m_given_thetas = np.load(f"{n_qubits}_{p}_layeredresults_samples_changeboth.npy")
                mean, error = mutual_info_standard_error(p_i_m_given_thetas)


            mean = mean[~np.isnan(error)]
            x = np.array(range(n_layers))[~np.isnan(error)]
            error = error[~np.isnan(error)]

            print(mean)
            print(error)

            # Plot on the subplot for the current probability
            axs[j].errorbar(x=x, y=mean, yerr=error, label=f"{n_qubits} qubits", color=colors[i], marker='x')

        axs[j].set_yscale('log')
        # Set the y-axis to show powers of 2
        yticks = [2**(-i) for i in range(17)]
        yticklabels = [f"{ytick:g}" for ytick in yticks]
        axs[j].set_yticks(yticks)
        axs[j].set_yticklabels(yticklabels)

        axs[j].set_xlabel("Layers")
        axs[j].set_ylabel("Log mutual information")
        axs[j].set_title(f"Mutual information for p = {p}")
        axs[j].legend(fontsize=12, title="Number of qubits")

    plt.tight_layout()
    plt.show()
