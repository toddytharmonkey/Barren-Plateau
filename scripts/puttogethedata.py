from VQEMonteCarlo import *

"""
Run this file to collect our data for the barren plateaus optimization runs. 
"""

if __name__ == "__main__":
    #     qubit_12 = np.load("results1214qubitshi.npy")
    #     last_few = np.load("restof16qubits.npy")

    # # (1, 20, 3, 3, 2, 60) (1, 2, 1, 3, 2, 60)

    #     print(qubit_12[0,:,2,0,0,0])

    #     print(np.shape(qubit_12))
    #     print(np.shape(last_few))

    #     for index, data in enumerate(last_few[0,:,0,:]):
    #         print(index)
    #         qubit_12[0,index+18,2,:] = data

    #     print(qubit_12[0,:,2,0,0,0])

    #     np.save("resultsfor12to16qubits",qubit_12)

    #     qubit_8 = np.load("8thrutenqubits.npy")
    #     qubit_6 = np.load("6thrutenqubits.npy")

    #     print(qubit_8.shape)
    #     print(qubit_6.shape)

    #    #(1, 20, 2, 3, 2, 60)
    #    # (1, 20, 3, 3, 2, 60)

    #     print(qubit_6[0,:,1,0,0,-1])

    #     six_eight_results = np.zeros((1, 20, 2, 3, 2, 60))

    #     six_eight_results[0,:,0,:] = qubit_6[0,:,0,:]
    #     six_eight_results[0,:,1,:] = qubit_8[0,:,0,:]
    #     print(six_eight_results)

    # #    np.save("resultsfor6to8qubits",six_eight_results)

    # twelvetosixteen = np.load("resultsfor12to16qubits.npy")

    # print(twelvetosixteen.shape)

    # sixtoeight = np.load("resultsfor6to8qubits.npy")
    # print(sixtoeight.shape)

    # ten = np.load("tenqubits.npy")
    # print(ten.shape)

    # total_results = np.append(sixtoeight, ten, axis=2)
    # total_results = np.append(total_results, twelvetosixteen, axis=2)
    # print(total_results.shape)

    # np.save("results12112023",total_results)

    a = np.load("results12112023.npy")

    print(a.shape)
