
from MILT_mutual_information import *
"""
Run this file to collect our data for the barren plateaus optimization runs. 
"""

if __name__ == "__main__":

    n_a = 400
    n_qubits = 6 
    n_layers = 60
    probs = [0, .1, .2, .3] 

    overall_results = []

    for p in probs: 
        if p == 0: 
            plt.plot(np.load(f"{n_qubits}_{p}_layeredresults.npy"), label=str(p), linewidth=3)
        else: 
            mean, std = np.load(f"{n_qubits}_{p}_layeredresults.npy")
            plt.errorbar(x= range(n_layers), y=mean, yerr=std, label=str(p), linewidth=3)

    plt.legend(fontsize=12)
    plt.show()