from VQEMonteCarlo import *

a = np.load("layered_results_test.npy")
b = np.load("layered_results_test6qubits.npy")
c = np.load("layered_results_test8qubits.npy")
d = np.load("layered_results_test10qubits.npy")
e

plt.plot(a, label = '4 qubits')
plt.plot(b, label = '6 qubits')
plt.plot(c, label = '8 qubits')
plt.plot(d, label= '10 qubits')
plt.yscale('log')
plt.title("Mutual information vs number of layers")
plt.ylabel("Mutual information")
plt.xlabel("Layer")
plt.legend()
plt.show()