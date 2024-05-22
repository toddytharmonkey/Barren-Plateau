import sys
sys.path.insert(0, '../..')
from MILT_mutual_information import *

"""
This version of the code calculates mutual info over n_ap samples, going through different thetas and measurement gate placements at the same time.  
"""

if __name__ == "__main__":
    client = Client()

    print("Dask Dashboard URL:", client.dashboard_link)

    n_ap = 10000
    qubit_list = [16] # remember to go back for 16 qubits, p=.28
    n_layers = 60
    probs = [.28]

    for n_qubits in qubit_list:
        for p in probs:
              p_i_m_given_thetas = generate_mutual_info_change_p_and_m_at_same_time(n_qubits, n_layers, n_ap, p)

              np.save(f"{n_qubits}_{p}_layeredresults_samples_nap_{n_ap}", p_i_m_given_thetas)
