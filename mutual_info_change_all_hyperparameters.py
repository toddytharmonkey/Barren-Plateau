from MILT_mutual_information import *

"""
This version of the code calculates mutual info, first averaged over n_a thetas, then averaged over n_a different measurement configurations. 
"""

if __name__ == "__main__":
    client = Client()

    print("Dask Dashboard URL:", client.dashboard_link)

    n_ap = 1000
    qubit_list = [4,6,8,10,12,14,16]
    n_layers = 60
    probs = [.05,.1,.2,.3,.5,.7,.9]

    for n_qubits in qubit_list:
        for p in probs:
              p_i_m_given_thetas = generate_mutual_info_change_p_and_m_at_same_time(n_qubits, n_layers, n_ap, p)

              np.save(f"{n_qubits}_{p}_layeredresults_samples_changeboth", p_i_m_given_thetas)

    client.close()
