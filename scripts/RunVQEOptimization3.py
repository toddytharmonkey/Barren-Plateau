from VQEMonteCarlo import *

"""
Testing how many measurement gates (on average) are generated for a given number of layers, qubits, and probability. 
"""

if __name__ == "__main__":
    print("Optimizations running!")

    ansatz = "HEA2"
    n_qubits = 8
    n_layers = 20
    n_shots = 1  # change this line
    post_selected = True  # change this line
    parallel = False
    gradient = "aware"

    for probability in np.arange(0, 1, 0.1):
        print("Probability: ", probability)
        print("Average number of gates: ", probability * (n_layers - 1) * n_qubits)

    for probability in [0.1, 0.01, 0.001, 0.0001]:
        print("Probability: ", probability)
        print("Average number of gates: ", probability * (n_layers - 1) * n_qubits)
