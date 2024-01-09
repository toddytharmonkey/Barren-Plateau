import numpy as np 
import matplotlib.pyplot as plt
from MILT_mutual_information import *


"""
This version of the code calculates mutual info, first averaged over n_a thetas, then averaged over n_a different measurement configurations. 
"""

if __name__ == "__main__":
    n_ap = 1000
    qubits = [4,6,8,10,12,14,16]
    n_layers = 60
    probs = [.05,.1,.2,.3,.5,.7,.9]


    for i, n_qubits in enumerate(qubits):

        results_for_each_p = []
        er_each_p = []

        for j, p in enumerate(probs):
            p_i_m_given_thetas = np.load(f"{n_qubits}_{p}_layeredresults_samples_changeboth.npy")

            bootstrap = mutual_info_bootstrap(p_i_m_given_thetas)
            c_i = bootstrap.confidence_interval

            p_i_m_given_thetas = p_i_m_given_thetas[:,:,-1]

            mutual_info = np.mean(mutual_info_changeall(p_i_m_given_thetas))
            #condense to just 1 number for each
             #avg_after_20_layer = np.mean(mean[20:])

            # results_for_each_p.append(mean[-1])
            # er_each_p.append(std[-1])

        plt.errorbar(x=probs, y=results_for_each_p, yerr = er_each_p, label = f"{n_qubits}", marker='.')

    plt.xlabel("Probability")
    plt.ylabel("Mutual information")
    plt.legend()
    plt.yscale('log') 
    plt.show()