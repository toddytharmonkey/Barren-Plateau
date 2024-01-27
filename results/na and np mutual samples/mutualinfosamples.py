from MILT_mutual_information import *

"""
This version of the code calculates mutual info, first averaged over n_a thetas, then averaged over n_p different measurement configurations. 
This version of the code was originally run on 1/5/2023. 

Sonny Rappaport 
"""

if __name__ == "__main__":
    client = Client()

    print("Dask Dashboard URL:", client.dashboard_link)

    n_a = 100
    n_p = 100
    qubit_list = [12,14,16]
    n_layers = 60
    probs = [.05,.1,.2]

    for n_qubits in qubit_list:
        for p in probs:
              p_i_m_given_thetas = generate_mutual_info_samples_dask_change_all_parameters_and_measurements(n_qubits, n_layers, n_a, n_p, p)

              np.save(f"{n_qubits}_{p}_layeredresults_samples", p_i_m_given_thetas)

    n_a = 100
    n_p = 100
    qubit_list = [4,6,8,10]
    n_layers = 60
    probs = [.05,.1,.2]

    for n_qubits in qubit_list:
        for p in probs:
              p_i_m_given_thetas = generate_mutual_info_samples_dask_change_all_parameters_and_measurements(n_qubits, n_layers, n_a, n_p, p)

              np.save(f"{n_qubits}_{p}_layeredresults_samples", p_i_m_given_thetas)

    n_a = 100
    n_p = 100
    qubit_list = [4,6,8,10,12,14,16]
    n_layers = 60
    probs = [.3,.5,.7,.9]

    for n_qubits in qubit_list:
        for p in probs:
              p_i_m_given_thetas = generate_mutual_info_samples_dask_change_all_parameters_and_measurements(n_qubits, n_layers, n_a, n_p, p)

              np.save(f"{n_qubits}_{p}_layeredresults_samples", p_i_m_given_thetas)

    n_a = 100
    n_p = 100
    qubit_list = [12,14,16]
    n_layers = 60
    probs = [.3,.5,.7,.9]

    for n_qubits in qubit_list:
        for p in probs:
              p_i_m_given_thetas = generate_mutual_info_samples_dask_change_all_parameters_and_measurements(n_qubits, n_layers, n_a, n_p, p)

              np.save(f"{n_qubits}_{p}_layeredresults_samples", p_i_m_given_thetas)


    client.close()
