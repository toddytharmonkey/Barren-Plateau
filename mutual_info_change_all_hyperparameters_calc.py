import numpy as np 
import matplotlib.pyplot as plt


"""
This version of the code calculates mutual info, first averaged over n_a thetas, then averaged over n_a different measurement configurations. 
"""

def mutual_info_changeall(p_i_m_given_thetas):
    """
    Given n_{ap} p_i_m_given_thetas samples, calculate the average mutual entropy
    """

    # samples are in shape (n_ap, 2, n_layers)

    # average over n_ap
    p_bi = np.mean(p_i_m_given_thetas, axis=(0))
    
    mutual_info = -np.sum(p_i_m_given_thetas * np.log(p_bi / p_i_m_given_thetas), axis=(1))

    return np.mean(mutual_info, axis=0), np.std(mutual_info, axis=0) / np.sqrt(len(p_i_m_given_thetas))

if __name__ == "__main__":
    n_ap = 1000
    qubits = [12,14,16]
    n_layers = 60
    probs = [.05,.1,.2]


    for i, n_qubits in enumerate(qubits):

        results_for_each_p = []
        er_each_p = []

        for j, p in enumerate(probs):
            p_i_m_given_thetas = np.load(f"{n_qubits}_{p}_layeredresults_samples_changeboth.npy")

            mean, std = mutual_info_changeall(p_i_m_given_thetas)

            #condense to just 1 number for each
             #avg_after_20_layer = np.mean(mean[20:])

            results_for_each_p.append(mean[-1])
            er_each_p.append(std[-1])

        plt.errorbar(x=probs, y=results_for_each_p, yerr = er_each_p, label = f"{n_qubits}", marker='.')

    plt.xlabel("Probability")
    plt.ylabel("Mutual information")
    plt.legend()
    plt.yscale('log') 
    plt.show()