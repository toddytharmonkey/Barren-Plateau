
from VQEMonteCarloWorkingDask import *
"""
Run this file to collect our data for the barren plateaus optimization runs. 
"""

if __name__ == "__main__":

    for n_qubits in [12,14,16]:

        n_layers = 100 

        layered_results = mutual_information_change_all_parameters(n_qubits, n_layers, n_a= 1000)

        np.save(f"layered_results{n_qubits}2", np.array(layered_results))
