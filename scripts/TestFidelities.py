from VQEMonteCarlo import *

if __name__ == "__main__":
    ansatz = "HEA2_uber_parameters"
    n_qubits = 8
    n_layers = 1
    probability = 0
    n_shots = 1  # change this line
    post_selected = True  # change this line
    parallel = False
    gradient = "aware"

    import openfermion as of

    # Initialize an empty QubitOperator to store the full Hamiltonian
    full_hamiltonian = of.QubitOperator()

    # Loop through each adjacent pair of qubits in the 12-qubit chain
    for i in range(n_qubits - 1):  # from 0 to 10
        xx_term = of.QubitOperator(f"X{i} X{i+1}", 1)
        yy_term = of.QubitOperator(f"Y{i} Y{i+1}", 1)
        zz_term = of.QubitOperator(f"Z{i} Z{i+1}", 0.5)

        # Add these terms to the full Hamiltonian
        full_hamiltonian += xx_term
        full_hamiltonian += yy_term
        full_hamiltonian += zz_term

    xx_term = of.QubitOperator(f"X{n_qubits-1} X{0}", 1)
    yy_term = of.QubitOperator(f"Y{n_qubits-1} Y{0}", 1)
    zz_term = of.QubitOperator(f"Z{n_qubits-1} Z{0}", 0.5)

    full_hamiltonian += xx_term
    full_hamiltonian += yy_term
    full_hamiltonian += zz_term
    # Convert to a sparse array
    full_ham_array = of.get_sparse_operator(full_hamiltonian)

    from openfermion.linalg import get_ground_state

    ground_energy, a = get_ground_state(full_ham_array)

    print(f"The ground state energy is {ground_energy}")

    exact_groundstate = torch.from_numpy(
        np.reshape(a, ((2,) * n_qubits)).astype(np.complex64)
    )

    # check that the groundstate is encoded directly in our arrays

    print(
        "Groundstate value,",
        Inner(exact_groundstate, ApplyHam(exact_groundstate, "xxz_1_1_05", 8, True)),
    )

    loaded_data = np.load(
        "prob_0_HEA2_uber_parameters_q8_l7_shots1_postTrue_xxz_1_1_05_aware_thetas10_version1.2/all_run_info.npy",
        allow_pickle=True,
    )
    print("loaded data shape", loaded_data.shape)
    print("loaded data, [1,0]", loaded_data[3, 0])
    print("loaded data, [1,1]", loaded_data[3, 1])

    parameters = loaded_data[3, 0]

    variational_state = HEA_uber_gradient_by_layer(
        n_qubits,
        n_layers,
        parameters,
        gradient_technique="analytical",
        gradient_index=0,
        measurements=None,
        dtheta=0.00001,
        return_analytic_suite=False,
        post_selected=False,
        entropy_regions=[[]],
        periodic=False,
        get_layered_results=False,
        ham_type="xxz_1_1_05",
        return_psi_list=True,
    )[0]
    print("variational state shape", variational_state.shape)
    print(
        f"The fidelity between the variational state and the groundstate is {np.abs(Inner(exact_groundstate, variational_state))**2}"
    )
