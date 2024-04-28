import os
import numpy as np

# Directory containing the files
directory = "."


# Function to parse filename and determine new filename based on shape
def rename_file(filename):
    # Skip the file if it matches the format [n]_0_layeredresults.npy
    if "_0_layeredresults.npy" in filename:
        return

    path = os.path.join(directory, filename)
    data = np.load(path)
    num_qubits, p_value = filename.split("_")[:2]

    # Check shape of the file
    shape = data.shape
    if len(shape) == 3:
        nap, two, n_layers = shape
        if two == 2:
            # Generate new filename
            new_filename = f"{num_qubits}_{p_value}_{n_layers}layers_nap_{nap}.npy"
            new_path = os.path.join(directory, new_filename)
            # Rename the file
            os.rename(path, new_path)
            print(f"Renamed '{filename}' to '{new_filename}'")
    else:
        raise ValueError("Uh oh. Shape was not three.")


# Loop through files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".npy"):
        rename_file(filename)
