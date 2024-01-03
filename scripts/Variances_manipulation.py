from VQEMonteCarlo import * 

if __name__ == "__main__":
    results = np.load("HEA2_xxz_results_redone.npy")
    print(results.shape)
    results_8_qubits = np.load("HEA2_xxz_results_8_qubits.npy")
    print(results_8_qubits.shape)

    # Assume 'original_array' is your existing 6D numpy array
    # and 'new_data' is the data you want to append with the same shape,
    # except for the third dimension (axis=2).

    # Step 1: Split the array along the third axis after the first element
    split1 = results[:, :, :1, ...]  # everything before the position
    split2 = results[:, :, 1:, ...]  # everything after the position

    # Step 2: Create or obtain the new data array (new_data)
    # Make sure it has the right shape:
    # new_data.shape == (original_array.shape[0], original_array.shape[1], NEW_SIZE, original_array.shape[3], ...)
    # where NEW_SIZE is the size you want to add along the third axis.

    # Step 3: Concatenate the first part with new_data
    split1_with_8_qubits = np.concatenate((split1, results_8_qubits), axis=2)

    # Step 4: Concatenate the result with the second part
    results = np.concatenate((split1_with_8_qubits, split2), axis=2)

    np.save("HEA_xxz_variances_complete.npy", results)

    # Now 'result_array' is the original array with 'new_data' appended after the first slice along the third axis.

    for i in range(5):

        unaware_variance = results[0,:,i,0,0,-1]
        aware_variance = results[0,:,i,0,1,-1]
        unaware_error = results[0,:,i,1,:,-1].T
        aware_error = results[0,:,i,2,:,-1].T    

        plt.errorbar([(.05)*i for i in range(14)], aware_variance, yerr = (aware_variance-aware_error[0],aware_error[1]-aware_variance), marker='o',label=6+2*i)



    plt.yscale('log')
    plt.title("aware variance, HEA2, 16 layers, XXZ hamiltonian")
    plt.xlabel("Probability to Place Gate")
    plt.ylabel("aware Variance")
    plt.legend(title="qubits")
    plt.yscale('log')
    plt.show()

    for i in range(5):

        aware_variance = results[0,0,i,0,1,:]
        unaware_error = results[0,0,i,1,:]
        aware_error = results[0,0,i,2,:]

        plt.errorbar([i for i in range(16)], aware_variance, yerr = (aware_variance-aware_error[0],aware_error[1]-aware_variance), marker='o', linestyle='')

    plt.yscale('log')
    plt.title("#############")
    plt.xlabel("Layer Index")
    plt.ylabel("unaware Variance")

    plt.show()
