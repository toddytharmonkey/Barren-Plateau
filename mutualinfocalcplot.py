from MILT_mutual_information import *
from distinctipy import distinctipy

"""
This file plots mutual information .npy files generated by MILT_mutual_information.
"""

if __name__ == "__main__":
    qubits = [4,6,8,10,12]
    n_layers = 60
    probs = [0, .05, .1, .2, .3,.5,.7,.9]

    # version #1 of the code: this does layers vs mutual info for different n_qubits 

    # # Generate distinct colors for each n_qubits
    # num_colors = len(qubits)
    # colors = distinctipy.get_colors(num_colors)

    # # Define markers for each probability
    # markers = ['.', 'x']  # You can add more markers if you have more probabilities

    # overall_results = []

    # for i, n_qubits in enumerate(qubits):
    #     for j, p in enumerate(probs):
    #         mean, std = np.load(f"{n_qubits}_{p}_layeredresults.npy")
    #         plt.errorbar(x=range(n_layers), y=mean, yerr=std, label=f"{p}, {n_qubits}", color=colors[i], marker=markers[j])

    # plt.yscale('log')
    # plt.xlabel("Layers")
    # plt.ylabel("Log mutual information")
    # plt.title("Mutual information for several qubits VS number of layers")
    # legend = plt.legend(fontsize=12)
    # legend.set_title("probability, number of qubits")
    # plt.show()

    # version 2 of the code: this does probability vs mutual info for different n_qubits, just like our variance results plots

    for i, n_qubits in enumerate(qubits):

        results_for_each_p = []
        er_each_p = []

        for j, p in enumerate(probs):
             
             # mean, std by layer of mutual information, in shape (n_layers) for mean + std)
            mean, std = np.load(f"results/{n_qubits}_{p}_layeredresults.npy")

            #condense to just 1 number for each
             #avg_after_20_layer = np.mean(mean[20:])

            results_for_each_p.append(mean[-1])
            er_each_p.append(std[-1])

        plt.errorbar(x=probs, y=results_for_each_p, yerr = er_each_p, label = f"{n_qubits}", marker='.')

    plt.xlabel("Probability")
    plt.ylabel("Mutual information")
    plt.legend()
    plt.yscale('log') 
    plt.show()
