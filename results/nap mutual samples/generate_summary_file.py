import os
import numpy as np
import sys

sys.path.insert(0, "../..")
from MILT_mutual_information import *


def load_and_aggregate_data(directory):
    # Parameters
    num_qubits_values = list(range(4, 20, 2))
    p_values = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    n_layers = 60
    nap_values = [10000, 1000]  # Order matters for preference

    # Initialize the aggregated data array
    aggregated_data = np.empty(
        (len(num_qubits_values), len(p_values), n_layers, 2)
    )  # 2 for mean and error
    aggregated_data.fill(np.nan)  # Fill with NaNs to indicate missing data

        # Iterate over all combinations of num_qubits, p_values, and nap_values
    for i, num_qubits in enumerate(num_qubits_values):
        for j, p_value in enumerate(p_values):
            file_found = False
            for nap in nap_values:
                if file_found:
                    break
                if p_value == 0 and num_qubits < 18:
                    mean, error = np.load(f"{num_qubits}_0_layeredresults.npy")
                else:
                    filename = f"{num_qubits}_{p_value}_{n_layers}layers_nap_{nap}.npy"

                    path = os.path.join(directory, filename)
                    if os.path.isfile(path):
                        file_data = np.load(path)
                        file_found = True 
                        mean, error = mutual_info_standard_error(file_data)
                    else:
                        continue
                        # Calculate mean and error across the axis corresponding to different samples (assuming axis 0)
                        #
                print(error[2])
                print(mean[2])
                # Store the results in the array
                aggregated_data[i, j, :, 0] = mean  # Storing mean
                aggregated_data[i, j, :, 1] = error  # Storing error

    # Check for any NaN values in the aggregated data
    if np.isnan(aggregated_data).any():
        # Create a detailed error report
        empty_cells = np.where(np.isnan(aggregated_data))
        detailed_errors = []
        for idx in zip(*empty_cells):
            detailed_errors.append(
                f"Empty cell found at num_qubits={num_qubits_values[idx[0]]}, p_value={p_values[idx[1]]}, layer={idx[2]}"
            )
        error_message = "\n".join(detailed_errors)
        raise ValueError(
            f"Some entries in the aggregated data array are empty. Please check the source files. Details:\n{error_message}"
        )
    # Save the aggregated data
    np.save(os.path.join(directory, "aggregated_data.npy"), aggregated_data)
    print("Data aggregation complete and saved.")


# Usage
directory = "."
load_and_aggregate_data(directory)

