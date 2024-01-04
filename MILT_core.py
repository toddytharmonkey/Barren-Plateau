from scipy.optimize import minimize
import ast
import openfermion as of
from dask.diagnostics import ProgressBar
from dask import delayed, compute
import matplotlib.pyplot as plt
from scipy.stats import bootstrap
from tqdm import tqdm  # assuming my code is going to be run in notebook
from collections import namedtuple
import numpy as np
import random
import os
import numpy as np
from dask.distributed import Client, as_completed
import quspin

code_version = "1.2"

ham_1_1_4 = of.get_sparse_operator(
    of.QubitOperator("X0 X1", 1)
    + of.QubitOperator("Y0 Y1", 1)
    + of.QubitOperator("Z0 Z1", 4)
).toarray()
ham_1_1_4 = np.reshape(
    np.array(ham_1_1_4, dtype=np.complex64),
    tuple([2 for i in range(int(np.log2(ham_1_1_4.size)))]),
)

ham_1_1_05 = of.get_sparse_operator(
    of.QubitOperator("X0 X1", 1)
    + of.QubitOperator("Y0 Y1", 1)
    + of.QubitOperator("Z0 Z1", 0.5)
).toarray()
ham_1_1_05 = np.reshape(
    np.array(ham_1_1_05, dtype=np.complex64),
    tuple([2 for i in range(int(np.log2(ham_1_1_05.size)))]),
)

"""
Generate gradient variance shots using 3 possible methods for different ansatz.

Code can run on the GPU if a device is available.

Sonny Rappaport, Gaurav Gyawali, Michael Lawler, March 2023
"""
# Basic Wavefunction Manipulation/Creations-----------------------------------


def ApplyGate(U, qubits, psi):
    """'Multiplies a state psi by gate U acting on qubits"""

    indices = "".join([chr(97 + q) for q in qubits])
    indices += "".join([chr(65 + q) for q in qubits])
    indices += ","
    indices += "".join(
        [chr(97 + i - 32 * qubits.count(i)) for i in range(len(psi.shape))]
    )

    #     print("U", U)
    #     print("qubits", qubits)
    #     print("indices",indices)

    return np.einsum(indices, U, psi)


def Inner(psi_1, psi_2):
    """<psi_1|psi_2>"""
    indices = "".join([chr(97 + q) for q in range(len(psi_1.shape))])
    indices += ","
    indices += "".join([chr(97 + q) for q in range(len(psi_2.shape))])
    return np.einsum(indices, psi_1.conj(), psi_2)


def Basis(n):
    """Returns the computational basis states for n qubits"""

    def i2binarray(i):
        return [int(c) for c in bin(i)[2:].zfill(n)]

    return [i2binarray(i) for i in range(2**n)]


def initial_state(n_qubits):
    """Initializes the qubits on the computational basis"""
    zero = np.array([1, 0], dtype=np.complex64)
    psi = np.array([1, 0], dtype=np.complex64)
    for i in range(n_qubits - 1):
        psi = np.kron(psi, zero)
    return psi.reshape((2,) * n_qubits)


# Gate Definitons--------------------------------------------------------------


def pauli(i):
    """Pauli matrix. i = 0 for I, 1 for X, 2 for Y, 3 for Z"""
    if i == 0:
        return np.eye(2)
    elif i == 1:
        return np.array([[0, 1], [1, 0]], dtype=np.complex64)
    elif i == 2:
        return np.array([[0, -1j], [1j, 0]], dtype=np.complex64)
    elif i == 3:
        return np.array([[1, 0], [0, -1]], dtype=np.complex64)
    else:
        return ValueError("i=0,1,2,3 only")


CNOT = np.reshape(
    np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=np.complex64,
    ),
    (2, 2, 2, 2),
)

H = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=np.complex64)

zz = np.kron(pauli(3), pauli(3))
xx = np.kron(pauli(1), pauli(1))
yy = np.kron(pauli(2), pauli(2))


def paulipauli(i):
    """Pauli matrix. i = 0 for I, 1 for XX, 2 for YY, 3 for ZZ"""

    if i == 0:
        return np.kron(np.eye(2), np.eye(2)).reshape(2, 2, 2, 2)
    elif i == 1:
        return xx
    elif i == 2:
        return yy
    elif i == 3:
        return zz
    else:
        return ValueError("i=0,1,2,3 only")


def rot(theta, i, grad=0):
    """Rotation gate. i = 1 for x, 2 for y, 3 for z"""
    if not grad:
        return np.cos(theta / 2) * np.eye(
            2,
            dtype=np.complex64,
        ) - 1j * np.sin(
            theta / 2
        ) * pauli(i)
    else:
        return -0.5 * np.sin(theta / 2) * np.eye(
            2,
            dtype=np.complex64,
        ) - 0.5j * np.cos(theta / 2) * pauli(i)


def rotrot(theta, i, grad=0):
    """Rotation gate. i = 1 for x, 2 for y, 3 for z"""
    if not grad:
        return (
            np.cos(theta / 2) * np.eye(4, dtype=np.complex64)
            - 1j * np.sin(theta / 2) * paulipauli(i)
        ).reshape(2, 2, 2, 2)
    else:
        return (
            -0.5 * np.sin(theta / 2) * np.eye(4, dtype=np.complex64)
            - 0.5j * np.cos(theta / 2) * paulipauli(i)
        ).reshape(2, 2, 2, 2)


def measure(i):
    """measurement operator on 0 or 1"""
    if i == 0:
        return np.array(
            [[1.0, 0], [0, 0]],
            dtype=np.complex64,
        )
    elif i == 1:
        return np.array(
            [[0, 0], [0, 1.0]],
            dtype=np.complex64,
        )
    else:
        return ValueError("can measure 0 or 1 only")


# Non-VQE-specific ansatz code--------------------------------------------------


def ApplyHam(psi, ham_type, n_qubits, periodic=True):
    if ham_type == "z0z1":
        return apply_Z0Z1(psi)
    elif ham_type == "xxz_1_1_4":
        return apply_XXZ_1_1_4(psi, periodic, n_qubits)
    elif ham_type == "xxz_1_1_05":
        return apply_XXZ_1_1_05(psi, periodic, n_qubits)
    else:
        raise ValueError("hamiltonian type is invalid")


def apply_XXZ_1_1_4(psi, periodic, n_qubits):
    "applies the XXZ hamiltonian to psi"
    psi_original = psi
    psi_cost = ApplyGate(ham_1_1_4, [0, 1], psi_original)

    #     print("n_qubits", n_qubits)

    for i in range(1, n_qubits - 1):
        psi_cost = psi_cost + ApplyGate(ham_1_1_4, [i, i + 1], psi_original)

    if periodic:
        psi_cost = psi_cost + ApplyGate(ham_1_1_4, [n_qubits - 1, 0], psi_original)

    return psi_cost


def apply_XXZ_1_1_05(psi, periodic, n_qubits):
    "applies the XXZ hamiltonian to psi"
    psi_original = psi
    psi_cost = ApplyGate(ham_1_1_05, [0, 1], psi_original)

    #     print("n_qubits", n_qubits)

    for i in range(1, n_qubits - 1):
        psi_cost = psi_cost + ApplyGate(ham_1_1_05, [i, i + 1], psi_original)

    if periodic:
        psi_cost = psi_cost + ApplyGate(ham_1_1_05, [n_qubits - 1, 0], psi_original)

    return psi_cost


def apply_Z0Z1(psi):
    "applies z0z1 to the first two qubits"
    return ApplyGate(zz.reshape(2, 2, 2, 2), [0, 1], psi)


def prob_rounder(p0, i):
    """
    adjusts rounding errors, particularly when the result is very close to 0
    or 1. the code in this python file is vulnrible to this error.
    """

    if np.isclose(p0, 1) or p0 > 1:
        p0 = 1
        np0 = 0

    elif np.isclose(p0, 0):
        p0 = 0
        np0 = 1
    else:
        np0 = 1 - p0

    return p0, np0


def num_parameters(n_qubits, n_layers, ansatz):
    if ansatz == "HEA2":
        return n_layers * 2
    elif ansatz == "HEA1":
        return n_qubits * n_layers * 2
    elif ansatz == "HEA2_uber_parameters":
        return n_qubits * n_layers * 2
    elif ansatz == "HVA":
        return n_layers * 4  # HVA
    else:
        raise ValueError(f"Input ansatz not identified, you entered {ansatz}")


def random_parameters(num):
    """
    returns returns every individual parameter for the HEA ansatz. This means
    every individual parameter in the returned list is repeated over a whole
    row of qubits.
    """

    return np.random.uniform(low=-4 * np.pi, high=4 * np.pi, size=num)


def random_rotations(num):
    return np.random.randint(low=1, high=4, size=num)


def random_measurements_prob(n_layers, n_qubits, chance):
    """generate a random list of measurements, with 'chance' that one is
    placed at a given location"""

    measure_list = []

    for depth in range(n_layers - 1):
        for qubit_num in range(n_qubits):
            if chance > np.random.rand():
                measure_list.append([depth, qubit_num])

    if len(measure_list) == 0:
        return None
    else:
        return measure_list


def random_measurements_num(n_layers, n_qubits, size):
    """generate a random list of 'size' measurements."""

    measure_list = []

    for depth in range(n_layers - 1):
        for qubit_num in range(n_qubits):
            measure_list.append([depth, qubit_num])

    measure_list = random.sample(measure_list, k=size)

    if len(measure_list) == 0:
        return None
    else:
        return measure_list


def layer_measure(psi_list, p, gradient_technique, post_selected, measurements, layer):
    if measurements is not None:
        measurements = np.array(measurements)
        relevant_qubits = measurements[np.where(measurements[:, 0] == layer)][:, 1:]
        if len(relevant_qubits):
            apply_measure(
                psi_list, relevant_qubits.tolist(), p, gradient_technique, post_selected
            )


def even_bisect_measurements(n_layers, n_qubits, n):
    """generate a list of 'n' measurements, bisecting the circuit into n evenly divided pieces
    and ensuring nearby measurements are spaced apart on qubits."""

    total_positions = n_layers * n_qubits
    partition_size = total_positions / n

    measure_list = []
    last_qubit_num = None
    occupied_qubits = set()

    for i in range(n):
        # Calculate the middle of the i-th partition for layer placement
        middle_position = (partition_size / 2) + (i * partition_size)
        depth = int(middle_position // n_qubits)

        # Decide qubit number based on the last measurement's qubit number
        if last_qubit_num is None:  # If this is the first measurement
            qubit_num = n_qubits // 2 if n_qubits % 2 == 0 else (n_qubits - 1) // 2
        else:
            middle_qubit = n_qubits // 2
            if last_qubit_num < middle_qubit:
                qubit_num = min(n_qubits - 2, last_qubit_num + 2)
            else:
                qubit_num = max(1, last_qubit_num - 2)

        # If qubit_num is already occupied for this depth, adjust its placement
        while (depth, qubit_num) in occupied_qubits:
            qubit_num = (qubit_num + 3) % n_qubits

        measure_list.append([depth, qubit_num])
        occupied_qubits.add((depth, qubit_num))
        last_qubit_num = qubit_num

    return measure_list


def apply_measure(psi_list, measurements, pM, gradient_technique, post_selected=True):
    """project the wavefunctions onto some randomly sampled basis
    returns the unnormalized (conditional) probabiliy
    This is the most subtle part of the entire code so be careful
    We do not normalize psi at each step so the output wavefunctions are
    \tilde{\psi}"""

    for q in measurements:
        outcome = 0  # default outcome

        original_p = 1

        for i in range(len(psi_list)):
            p0 = (
                np.abs(Inner(ApplyGate(measure(0), q, psi_list[i]), psi_list[i]))
                .to("cpu")
                .numpy()
            )

            p0, p1 = prob_rounder(p0, i)

            # if not post-selected, make the outcome from the first (normal) circuit
            if i == 0 and post_selected == False:
                outcome = np.random.choice([0, 1], 1, p=[p0, p1])[0]
            elif i == 0 and post_selected == True:
                outcome = 0

            if i == 0:
                original_p = [p0, p1][outcome]

            p_i = [p0, p1][outcome]

            pM[i] = pM[i] * p_i

            if gradient_technique == "analytic":
                psi_list[i] = ApplyGate(measure(outcome), q, psi_list[i]) / np.sqrt(
                    original_p
                )
            else:
                psi_list[i] = ApplyGate(measure(outcome), q, psi_list[i]) / np.sqrt(p_i)


def gradients_by_layer(
    n_qubits,
    n_layers,
    parameters,
    gradient_technique="numeric",
    gradient_index=0,
    measurements=None,
    dtheta=0.00001,
    return_analytic_suite=False,
    post_selected=False,
    entropy_regions=[[]],
    periodic=False,
    get_layered_results=False,
    ham_type="z0z1",
    ansatz="GG",
    rotations=None,
):
    """
    Given chosen circuit parameters and a method to generate the gradient, constructs
    appropriate psi's and uses them to return unaware, aware gradients or more
    for the HEA ansatz.

    If periodic is true, then an additional CNOT will be placed to wrap around
    from the last qubit to the first one.

    If return_analytic suite is true more information is returned with the
    analytic variance, where entropy_regions is used to return possibly
    muiltiple different regions of entropy

    If post_selected is true, all measurements are post_selected.

    dtheta controls the numeric calculation's precision.
    """

    if (
        gradient_technique is not None
        and gradient_technique != "shift"
        and gradient_technique != "analytic"
        and gradient_technique != "numeric"
    ):
        raise ValueError("gradient_technique is not a valid type")

    if ansatz == "HEA2":
        return HEA_gradient_by_layer(
            n_qubits,
            n_layers,
            parameters,
            gradient_technique,
            gradient_index,
            measurements,
            dtheta,
            return_analytic_suite,
            post_selected,
            entropy_regions,
            periodic,
            get_layered_results,
            ham_type,
        )

    if ansatz == "HEA2_uber_parameters":
        return HEA_uber_gradient_by_layer(
            n_qubits,
            n_layers,
            parameters,
            gradient_technique,
            gradient_index,
            measurements,
            dtheta,
            return_analytic_suite,
            post_selected,
            entropy_regions,
            periodic,
            get_layered_results,
            ham_type,
        )
    elif ansatz == "HEA1":
        return GG_gradient_by_layer(
            n_qubits,
            n_layers,
            parameters,
            rotations,
            gradient_technique,
            gradient_index,
            measurements,
            dtheta,
            return_analytic_suite,
            post_selected,
            entropy_regions,
            periodic,
            get_layered_results,
            ham_type,
        )

    elif ansatz == "HVA":
        return HVA_gradient_by_layer(
            n_qubits,
            n_layers,
            parameters,
            gradient_technique,
            gradient_index,
            measurements,
            dtheta,
            return_analytic_suite,
            post_selected,
            entropy_regions,
            periodic,
            get_layered_results,
            ham_type,
        )


# HEA2 uber parameters specific code ----------------------------------------------------------


def HEA_uber_gradient_by_layer(
    n_qubits,
    n_layers,
    parameters,
    gradient_technique="numeric",
    gradient_index=0,
    measurements=None,
    dtheta=0.00001,
    return_analytic_suite=False,
    post_selected=False,
    entropy_regions=[[]],
    periodic=False,
    get_layered_results=False,
    ham_type="z0z1",
    return_psi_list=False,
):
    layer_results = []

    if gradient_technique == "shift":
        # create nonshifted, -, + versions of psi , for each individually shifted psi
        psi_list = [initial_state(n_qubits for _ in range(3))]
    elif gradient_technique == "analytic":
        # normal and derivative circuits
        psi_list = [initial_state(n_qubits) for _ in range(2)]
    else:
        # normal, dtheta plus and dtheta minus circuit
        psi_list = [initial_state(n_qubits) for _ in range(3)]

    # probability for each psi
    psi_list_probabilities = [1.0 for _ in psi_list]

    current_param = 0  # start with parameter 0

    layer_H(n_qubits, psi_list, [q for q in range(n_qubits)])

    for l in range(n_layers):
        # print(l)
        # 1: Rotation gates on all qubits
        layer_rot_GG(
            n_qubits,
            psi_list,
            parameters,
            2,
            current_param,
            gradient_index,
            one_parameter=True,
        )
        current_param += n_qubits

        # 2: CNOT between qth and (q+1)th qubits where q is even
        layer_CNOT_ladder(n_qubits, psi_list, periodic)

        # 3: Rotation gates on all qubits
        # with analytical gradient
        layer_rot_GG(
            n_qubits,
            psi_list,
            parameters,
            1,
            current_param,
            gradient_index,
            one_parameter=True,
        )
        current_param += n_qubits

        layer_measure(
            psi_list,
            psi_list_probabilities,
            gradient_technique,
            post_selected,
            measurements,
            l,
        )

        if get_layered_results:
            layer_results.append(
                gradients_GG(
                    n_qubits,
                    psi_list,
                    gradient_technique,
                    psi_list_probabilities,
                    dtheta,
                    return_analytic_suite,
                    entropy_regions,
                    ham_type,
                    periodic,
                )
            )
    if return_psi_list:
        return psi_list
    if get_layered_results:
        return layer_results
    else:
        #         print(psi_list)
        return gradients_GG(
            n_qubits,
            psi_list,
            gradient_technique,
            psi_list_probabilities,
            dtheta,
            return_analytic_suite,
            entropy_regions,
            ham_type,
            periodic,
        )


# HEA VQE specific code------------------------------------------------------------------------


def GG_gradient_by_layer(
    n_qubits,
    n_layers,
    parameters,
    rotations,
    gradient_technique="numeric",
    gradient_index=0,
    measurements=None,
    dtheta=0.00001,
    return_analytic_suite=False,
    post_selected=False,
    entropy_regions=[[]],
    periodic=False,
    get_layered_results=False,
    ham_type="z0z1",
):
    if np.any(rotations) == None:
        raise TypeError("GG ansatz needs rotation gate types")

    layer_results = []

    if gradient_technique == "shift":
        # create nonshifted, -, + versions of psi , for each individually shifted psi
        psi_list = [initial_state(n_qubits for _ in range(3))]
    elif gradient_technique == "analytic":
        # normal and derivative circuits
        psi_list = [initial_state(n_qubits) for _ in range(2)]
    else:
        # normal, dtheta plus and dtheta minus circuit
        psi_list = [initial_state(n_qubits) for _ in range(3)]

    # probability for each psi
    psi_list_probabilities = [1.0 for _ in psi_list]

    current_param = 0  # start with parameter 0

    layer_y_rot_GG(n_qubits, psi_list, np.pi)

    for l in range(n_layers):
        # print(l)
        # 1: Rotation gates on all qubits
        layer_rot_GG(
            n_qubits, psi_list, parameters, rotations, current_param, gradient_index
        )
        current_param += n_qubits

        # 2: CNOT between qth and (q+1)th qubits where q is even
        layer_CNOT_even(n_qubits, psi_list)

        # 3: Rotation gates on all qubits
        # TODO- align this with the other layer_rots, also only currently works
        # with analytical gradient
        layer_rot_GG(
            n_qubits, psi_list, parameters, rotations, current_param, gradient_index
        )
        current_param += n_qubits

        # 4: CNOT between qth and (q+1)th qubits where q is even
        layer_CNOT_odd(n_qubits, psi_list)

        layer_measure(
            psi_list,
            psi_list_probabilities,
            gradient_technique,
            post_selected,
            measurements,
            l,
        )

        if get_layered_results:
            layer_results.append(
                gradients_GG(
                    n_qubits,
                    psi_list,
                    gradient_technique,
                    psi_list_probabilities,
                    dtheta,
                    return_analytic_suite,
                    entropy_regions,
                    ham_type,
                    periodic,
                )
            )

    if get_layered_results:
        return layer_results
    else:
        #         print(psi_list)
        return gradients_GG(
            n_qubits,
            psi_list,
            gradient_technique,
            psi_list_probabilities,
            dtheta,
            return_analytic_suite,
            entropy_regions,
            ham_type,
            periodic,
        )

    return layer_results


def gradients_GG(
    n_qubits,
    psi_list,
    gradient_technique,
    p,
    dtheta,
    return_analytic_suite,
    entropy_regions,
    ham_type,
    periodic,
):
    """
    Given a psi_list and a gradient_technique, returns various useful inner
    products such as the gradients or cost function.

    If return_analytic_suite is true, if the gradient_technique is analytic then
    the renyi entropy and cost function are returned as well as the gradients.
    """

    psi = psi_list[0]
    cost_psi = ApplyHam(psi, ham_type, n_qubits, periodic)

    if gradient_technique == "numeric":
        psi_plus = psi_list[1]
        cost_psi_plus = ApplyHam(psi, ham_type, n_qubits, periodic)

        psi_minus = psi_list[2]
        cost_psi_minus = ApplyHam(psi, ham_type, n_qubits, periodic)

        return (
            Inner(psi_plus, cost_psi_plus).real - Inner(psi_minus, cost_psi_minus).real
        ) / (2 * dtheta)

    elif gradient_technique == "shift":
        raise NotImplementedError

        # aware_gradient = 0
        # unaware_gradient = 0

        # grouped_psi_list = np.reshape(psi_list, tuple(
        #     [n_qubits, 3, *[2 for _ in range(n_qubits)]]))

        # grouped_p = np.reshape(p, (n_qubits, 3))

        # for k, psi_list in enumerate(grouped_psi_list):
        #     O = []
        #     for psi in psi_list:
        #         cost_psi = ApplyHam(psi, ham_type, periodic)
        #         O.append(Inner(psi, cost_psi).real)
        #     cost_p, cost_p_plus, cost_p_minus = O
        #     prob, prob_plus, prob_minus = grouped_p[k]

        #     aware_gradient += 0.5 * \
        #         ((cost_p_plus*prob_plus - cost_p_minus*prob_minus) /
        #          prob - (prob_plus-prob_minus)*cost_p/prob)
        #     unaware_gradient += 0.5 * \
        #         (cost_p_plus*prob_plus - cost_p_minus*prob_minus)/prob

        # return unaware_gradient, aware_gradient

    else:  # analytical gradient
        C = Inner(psi, cost_psi).real
        # Calculate the gradients using eqn 22 in the notes. Note we are not multiplying by p because of MC sampling
        term1 = Inner(psi_list[1], cost_psi)  # first term
        # first term - second term
        term2 = term1 - C * Inner(psi_list[1], psi_list[0])

        # Note that the second term is being divided by only prob instead of prob^2 because the cost function is normalized here
        unaware_gradient = 2 * term1.real
        aware_gradient = 2 * term2.real

        if return_analytic_suite:
            return (
                C.cpu().numpy(),
                unaware_gradient.cpu().numpy(),
                aware_gradient.cpu().numpy(),
            )
        else:
            return unaware_gradient.cpu().numpy(), aware_gradient.cpu().numpy()


def layer_rot_GG(
    n_qubits,
    psi_list,
    parameters,
    rotations,
    current_param,
    gradient,
    one_parameter=False,
):
    for q in range(n_qubits):
        for i in range(len(psi_list)):
            param = parameters[current_param]
            take_grad = 1 if (current_param == gradient and i == 1) else 0
            if not one_parameter:
                psi_list[i] = ApplyGate(
                    rot(param, rotations[current_param], take_grad), [q], psi_list[i]
                )
            else:
                psi_list[i] = ApplyGate(
                    rot(param, rotations, take_grad), [q], psi_list[i]
                )
        current_param += 1


def layer_y_rot_GG(n_qubits, psi_list, param):
    for q in range(n_qubits):
        for i in range(len(psi_list)):
            psi_list[i] = ApplyGate(rot(param, 2), [q], psi_list[i])


def layer_CNOT_even(n_qubits, psi_list, periodic_boundary=False):
    """apply a CNOT between an qth and (q+1)th qubits where q is even"""
    for i in range(len(psi_list)):
        for q in range(n_qubits - 1):
            if q % 2 == 0:
                psi_list[i] = ApplyGate(CNOT, [q, q + 1], psi_list[i])
        if periodic_boundary == True and n_qubits % 2 == 1:
            psi_list[i] = ApplyGate(CNOT, [n_qubits - 1, 0], psi_list[i])


def layer_CNOT_odd(n_qubits, psi_list, periodic_boundary=False):
    """apply a CNOT between an qth and (q+1)th qubits where q is odd"""
    for i in range(len(psi_list)):
        for q in range(n_qubits - 1):
            if q % 2 != 0:
                psi_list[i] = ApplyGate(CNOT, [q, q + 1], psi_list[i])
        if periodic_boundary == True and n_qubits % 2 == 0:
            psi_list[i] = ApplyGate(CNOT, [n_qubits - 1, 0], psi_list[i])


# HEA2 VQE specific code---------------------------------------------------------


def HEA_gradient_by_layer(
    n_qubits,
    n_layers,
    parameters,
    gradient_technique="numeric",
    gradient_index=0,
    measurements=None,
    dtheta=0.00001,
    return_analytic_suite=False,
    post_selected=False,
    entropy_regions=[[]],
    periodic=False,
    get_layered_results=False,
    ham_type="z0z1",
):
    layer_results = []

    if gradient_technique == None:
        psi_list = [initial_state(n_qubits)]
    elif gradient_technique == "shift":
        # create nonshifted, -, + versions of psi , for each individually shifted psi
        psi_list = np.stack([initial_state(n_qubits) for _ in range(3 * n_qubits)])
    elif gradient_technique == "analytic":
        # normal and derivative circuits
        psi_list = [initial_state(n_qubits) for _ in range(2)]
    else:
        # normal, dtheta plus and dtheta minus circuit
        psi_list = [initial_state(n_qubits) for _ in range(3)]

    # 0: H rotation gates on all qubits (initial rotation)

    layer_H(n_qubits, psi_list, [i for i in range(n_qubits)])

    p = [1.0 for _ in psi_list]  # probability for each psi
    current_param = 0  # start with parameter 0

    for l in range(n_layers):
        # 1: paramaterized Ry rotation gates on all qubits

        layer_rot(
            n_qubits,
            psi_list,
            parameters,
            2,
            current_param,
            gradient_technique,
            gradient_index,
            dtheta,
        )
        current_param += 1

        # # 2: CNOTs connecting from the top to bottom qubit
        layer_CNOT_ladder(n_qubits, psi_list, periodic)

        # 3: Rotation gates on all qubits
        layer_rot(
            n_qubits,
            psi_list,
            parameters,
            1,
            current_param,
            gradient_technique,
            gradient_index,
            dtheta,
        )
        current_param += 1

        if get_layered_results:
            layer_results.append(
                gradients_HEA(
                    n_qubits,
                    psi_list,
                    gradient_technique,
                    p,
                    dtheta,
                    return_analytic_suite,
                    entropy_regions,
                    ham_type,
                    periodic,
                )
            )

        # If measurements are provided apply them
        layer_measure(psi_list, p, gradient_technique, post_selected, measurements, l)

    if get_layered_results:
        return layer_results
    else:
        return gradients_HEA(
            n_qubits,
            psi_list,
            gradient_technique,
            p,
            dtheta,
            return_analytic_suite,
            entropy_regions,
            ham_type,
            periodic,
        )


def layer_H(n_qubits, psi_list, qubit_range):
    """apply a rotation layer of Hadamard gates to all qubits"""
    for i in range(len(psi_list)):
        for q in range(n_qubits):
            psi_list[i] = ApplyGate(H, [q], psi_list[i])


def gradients_HEA(
    n_qubits,
    psi_list,
    gradient_technique,
    p,
    dtheta,
    return_analytic_suite,
    entropy_regions,
    ham_type,
    periodic,
):
    """
    Given a psi_list and a gradient_technique, returns various useful inner
    products such as the gradients or cost function.

    If return_analytic_suite is true, if the gradient_technique is analytic then
    the renyi entropy and cost function are returned as well as the gradients.
    """

    if entropy_regions == True:
        return generate_entropies(
            psi_list[0],
            n_qubits,
            [[q for q in range(n_qubits // 2)], [0], [n_qubits // 2]],
        )

    psi = psi_list[0]
    cost_psi = ApplyHam(psi, ham_type, n_qubits, periodic)

    if gradient_technique == "numeric":
        psi_plus = psi_list[1]
        cost_psi_plus = ApplyHam(psi, ham_type, n_qubits, periodic)

        psi_minus = psi_list[2]
        cost_psi_minus = ApplyHam(psi, ham_type, n_qubits, periodic)

        return (
            Inner(psi_plus, cost_psi_plus).real - Inner(psi_minus, cost_psi_minus).real
        ) / (2 * dtheta)

    elif gradient_technique == "shift":
        aware_gradient = 0
        unaware_gradient = 0

        grouped_psi_list = np.reshape(
            psi_list, tuple([n_qubits, 3, *[2 for _ in range(n_qubits)]])
        )

        grouped_p = np.reshape(p, (n_qubits, 3))

        for k, psi_list in enumerate(grouped_psi_list):
            O = []
            for psi in psi_list:
                cost_psi = ApplyHam(psi, ham_type, n_qubits, periodic)
                O.append(Inner(psi, cost_psi).real)
            cost_p, cost_p_plus, cost_p_minus = O
            prob, prob_plus, prob_minus = grouped_p[k]

            aware_gradient += 0.5 * (
                (cost_p_plus * prob_plus - cost_p_minus * prob_minus) / prob
                - (prob_plus - prob_minus) * cost_p / prob
            )
            unaware_gradient += (
                0.5 * (cost_p_plus * prob_plus - cost_p_minus * prob_minus) / prob
            )

        return unaware_gradient, aware_gradient

    else:  # analytical gradient
        # NOTE: right now code is in a mode just to calculate cost functions without caring about the gradients.
        # this part definetly needs to be fixed if you actually want to uise gradients

        C = Inner(psi, cost_psi).real
        return C

        # Calculate the gradients using eqn 22 in the notes. Note we are not multiplying by p because of MC sampling
        term1 = Inner(psi_list[1], cost_psi)  # first term
        # first term - second term
        term2 = term1 - C * Inner(psi_list[1], psi_list[0])

        # Note that the second term is being divided by only prob instead of prob^2 because the cost function is normalized here
        unaware_gradient = 2 * term1.real
        aware_gradient = 2 * term2.real

        if return_analytic_suite:
            return (
                C.cpu().numpy(),
                unaware_gradient.cpu().numpy(),
                aware_gradient.cpu().numpy(),
            )
        else:
            return unaware_gradient.cpu().numpy(), aware_gradient.cpu().numpy()


def layer_rot(
    n_qubits,
    psi_list,
    parameters,
    rotation,
    current_param,
    gradient_technique,
    gradient_index,
    dtheta,
):
    if gradient_technique is None:
        layer_rot_none(n_qubits, psi_list, parameters, rotation, current_param)
    elif gradient_technique == "shift":
        layer_rot_shift(
            n_qubits, psi_list, parameters, rotation, current_param, gradient_index
        )
    elif gradient_technique == "analytic":
        layer_rot_analytical(
            n_qubits, psi_list, parameters, rotation, current_param, gradient_index
        )
    else:
        layer_rot_numeric(
            n_qubits,
            psi_list,
            parameters,
            rotation,
            current_param,
            gradient_index,
            dtheta,
        )


def layer_rot_none(n_qubits, psi_list, parameters, rotation, current_param):
    for i in range(len(psi_list)):
        psi_list[i] = rotation_layer(
            n_qubits, psi_list[i], parameters[current_param], rotation
        )


def layer_rot_shift(
    n_qubits, psi_list, parameters, rotation, current_param, gradient_index
):
    p = (0, np.pi / 2, -np.pi / 2)

    # group the psis into n_qubit groups of threes to account for the product rule
    grouped_psi_list = np.reshape(
        psi_list, tuple([n_qubits, 3, *[2 for _ in range(n_qubits)]])
    )

    # for every kth psi whose derivative is taken in the kth row:
    for k in range(len(grouped_psi_list)):
        for q in range(n_qubits):  # over every qubit:
            for i in range(3):  # over each nonshifted, -, + psi
                # if the gradient index is the current parameter and this is the correct psi
                if gradient_index == current_param and k == q:
                    grouped_psi_list[k, i] = ApplyGate(
                        rot(parameters[current_param] + p[i], rotation),
                        [q],
                        grouped_psi_list[k, i],
                    )
                # otherwise, just apply the regular (identical) parameter over the whole column
                else:
                    grouped_psi_list[k, i] = ApplyGate(
                        rot(parameters[current_param], rotation),
                        [q],
                        grouped_psi_list[k, i],
                    )


def row_with_one_gradient(n_qubits, psi, parameter, rotation, gradient_qubit):
    for q in range(n_qubits):
        psi = ApplyGate(rot(parameter, rotation, q == gradient_qubit), [q], psi)
    return psi


def product_rule_psi(n_qubits, psi, parameter, rotation):
    original_psi = psi.clone()
    psi_result = np.zeros(psi.shape)
    for q in range(n_qubits):
        gradient_psi_term = row_with_one_gradient(
            n_qubits, original_psi, parameter, rotation, q
        )

        psi_result = psi_result + gradient_psi_term
    return psi_result


def layer_rot_analytical(
    n_qubits, psi_list, parameters, rotation, current_param, gradient_index
):
    for i in range(len(psi_list)):
        # if this is del_psi and it's time to take a derivative:
        if current_param == gradient_index and i == 1:
            psi_list[i] = product_rule_psi(
                n_qubits, psi_list[i], parameters[current_param], rotation
            )

        else:
            psi_list[i] = rotation_layer(
                n_qubits, psi_list[i], parameters[current_param], rotation
            )


def layer_CNOT_ladder(n_qubits, psi_list, periodic_boundary=False):
    """apply a CNOT between an qth and (q+1)th qubits where q is even"""
    for i in range(len(psi_list)):
        for q in range(n_qubits - 1):
            psi_list[i] = ApplyGate(CNOT, [q, q + 1], psi_list[i])
        if periodic_boundary == True:
            psi_list[i] = ApplyGate(CNOT, [n_qubits - 1, 0], psi_list[i])


def layer_rot_numeric(
    n_qubits, psi_list, parameters, rotation, current_param, gradient_index, dtheta
):
    if gradient_index != current_param:
        for i in range(len(psi_list)):
            psi_list[i] = rotation_layer(
                n_qubits, psi_list[i], parameters[current_param], rotation
            )
    else:
        p = (
            parameters[current_param],
            parameters[current_param] + dtheta,
            parameters[current_param] - dtheta,
        )
        for q in range(n_qubits):
            for i in range(len(psi_list)):
                psi_list[i] = ApplyGate(rot(p[i], rotation), [q], psi_list[i])


def rotation_layer(n_qubits, psi, parameter, rotation):
    """
    applies an identical rotation gate over all the qubits in psi.
    """
    for q in range(n_qubits):
        psi = ApplyGate(rot(parameter, rotation), [q], psi)

    return psi


# HVA VQE specific code----------------------------------------------------------


def HVA_gradient_by_layer(
    n_qubits,
    n_layers,
    parameters,
    gradient_technique="numeric",
    gradient_index=0,
    measurements=None,
    dtheta=0.00001,
    return_analytic_suite=False,
    post_selected=False,
    entropy_regions=[[]],
    periodic=False,
    get_layered_results=False,
    ham_type="z0z1",
):
    layer_results = []

    if gradient_technique == "shift":
        # create nonshifted, -, + versions of psi , for each individually shifted psi
        psi_list = np.stack([initial_state(n_qubits) for _ in range(3 * n_qubits)])
    elif gradient_technique == "analytic":
        # normal and derivative circuits
        psi_list = [initial_state(n_qubits) for _ in range(2)]
    else:
        # normal, dtheta plus and dtheta minus circuit
        psi_list = [initial_state(n_qubits) for _ in range(3)]

    # 0: H rotation gates on all qubits (initial rotation)

    # layer_x_rot_HVA(n_qubits, psi_list, np.pi)

    layer_H(n_qubits, psi_list, [i for i in range(n_qubits) if i % 2 == 0])

    layer_CNOT_even(n_qubits, psi_list)

    p = [1.0 for _ in psi_list]  # probability for each psi
    current_param = 0  # start with parameter 0

    for l in range(n_layers):
        # 1: paramaterized Ry rotation gates on all qubits

        layer_paulixpauli(
            n_qubits,
            psi_list,
            parameters,
            0,
            current_param,
            gradient_technique,
            gradient_index,
            dtheta,
        )
        current_param += 1

        layer_paulixpauli(
            n_qubits,
            psi_list,
            parameters,
            1,
            current_param,
            gradient_technique,
            gradient_index,
            dtheta,
        )
        current_param += 1

        layer_paulixpauli(
            n_qubits,
            psi_list,
            parameters,
            2,
            current_param,
            gradient_technique,
            gradient_index,
            dtheta,
        )
        current_param += 1

        layer_paulixpauli(
            n_qubits,
            psi_list,
            parameters,
            3,
            current_param,
            gradient_technique,
            gradient_index,
            dtheta,
        )
        current_param += 1

        # If measurements are provided apply them
        layer_measure(psi_list, p, gradient_technique, post_selected, measurements, l)

        if get_layered_results:
            layer_results.append(
                gradients_HVA(
                    n_qubits,
                    psi_list,
                    gradient_technique,
                    p,
                    dtheta,
                    return_analytic_suite,
                    entropy_regions,
                    ham_type,
                    periodic,
                )
            )

    if get_layered_results:
        return layer_results
    else:
        return gradients_HVA(
            n_qubits,
            psi_list,
            gradient_technique,
            p,
            dtheta,
            return_analytic_suite,
            entropy_regions,
            ham_type,
            periodic,
        )


def gradients_HVA(
    n_qubits,
    psi_list,
    gradient_technique,
    p,
    dtheta,
    return_analytic_suite,
    entropy_regions,
    ham_type,
    periodic,
):
    psi = psi_list[0]
    cost_psi = ApplyHam(psi, ham_type, n_qubits, periodic)

    C = Inner(psi, cost_psi).real

    # Calculate the gradients using eqn 22 in the notes. Note we are not multiplying by p because of MC sampling
    term1 = Inner(psi_list[1], cost_psi)  # first term
    # first term - second term
    term2 = term1 - C * Inner(psi_list[1], psi_list[0])

    # Note that the second term is being divided by only prob instead of prob^2 because the cost function is normalized here
    unaware_gradient = 2 * term1.real
    aware_gradient = 2 * term2.real

    if return_analytic_suite:
        return (
            C.cpu().numpy(),
            unaware_gradient.cpu().numpy(),
            aware_gradient.cpu().numpy(),
        )
    else:
        return unaware_gradient.cpu().numpy(), aware_gradient.cpu().numpy()


def layer_paulixpauli(
    n_qubits,
    psi_list,
    parameters,
    type_gates,
    current_param,
    gradient_technique,
    gradient_index,
    dtheta,
):
    """

    Puts down a layer of double pauli gates in the fashion of the Roeland paper.
    Type 1 is odd double ZZ gates, Type 2 is odd double YY and XX gates,
    Type 3 is even ZZ gates, Type 3 is even YY and XX

    """

    if gradient_technique == "shift":
        layer_paulixpauli_shift(
            n_qubits, psi_list, parameters, type_gates, current_param, gradient_index
        )
    elif gradient_technique == "analytic":
        layer_paulixpauli_analytical(
            n_qubits, psi_list, parameters, type_gates, current_param, gradient_index
        )
    else:
        layer_paulixpaul_it_numeric(
            n_qubits,
            psi_list,
            parameters,
            type_gates,
            current_param,
            gradient_index,
            dtheta,
        )


def layer_paulixpauli_shift(
    n_qubits, psi_list, parameters, rotation, current_param, gradient_index
):
    raise NotImplementedError


def rotation_layer_HVA_gradient(n_qubits, psi, parameter, type_gate):
    if type_gate == 0:
        psi = layer_paulipauli_derivative_zz(n_qubits, psi, parameter, True, pbc=True)
    elif type_gate == 1:
        psi = layer_paulipauli_derivative_yyxx(n_qubits, psi, parameter, True, pbc=True)
    elif type_gate == 2:
        psi = layer_paulipauli_derivative_zz(n_qubits, psi, parameter, False, pbc=True)
    elif type_gate == 3:
        psi = layer_paulipauli_derivative_yyxx(
            n_qubits, psi, parameter, False, pbc=True
        )
    else:
        raise ValueError("type_gate is not a number from 0-3 inclusive")

    return psi


def rotation_layer_HVA(n_qubits, psi, parameter, type_gate, gradient=False):
    if type_gate == 0:
        psi = layer_paulipauli(n_qubits, psi, parameter, True, "zz", pbc=True)
    elif type_gate == 1:
        psi = layer_paulipauli(n_qubits, psi, parameter, True, "yy", pbc=True)
        psi = layer_paulipauli(n_qubits, psi, parameter, True, "xx", pbc=True)
    elif type_gate == 2:
        psi = layer_paulipauli(n_qubits, psi, parameter, False, "zz", pbc=True)
    elif type_gate == 3:
        psi = layer_paulipauli(n_qubits, psi, parameter, False, "yy", pbc=True)
        psi = layer_paulipauli(n_qubits, psi, parameter, False, "xx", pbc=True)
    else:
        raise ValueError("type_gate is not a number from 0-3 inclusive")

    return psi


def layer_paulixpauli_analytical(
    n_qubits, psi_list, parameters, type_gates, current_param, gradient_index
):
    psi, del_psi = psi_list

    psi_list[0] = rotation_layer_HVA(
        n_qubits, psi, parameters[current_param], type_gates
    )

    if gradient_index == current_param:
        psi_list[1] = rotation_layer_HVA_gradient(
            n_qubits, del_psi, parameters[current_param], type_gates
        )

    else:
        psi_list[1] = rotation_layer_HVA(
            n_qubits, del_psi, parameters[current_param], type_gates
        )


def layer_paulipauli_derivative_zz(n_qubits, psi, theta, odd, pbc=True):
    M = np.mul(-1j * theta / 2, zz)
    U = np.matrix_exp(M).reshape(2, 2, 2, 2)
    d_U = np.mul(np.mul(-1j / 2, zz), np.matrix_exp(M)).reshape(2, 2, 2, 2)

    psi_result = np.zeros(psi.shape)

    for del_q in range(odd, n_qubits - 1, 2):
        individual_product_psi = psi.clone()

        # product rule over the individual paulixpauli gates
        for q in range(odd, n_qubits - 1, 2):
            if del_q == q:
                individual_product_psi = ApplyGate(
                    d_U, [q, q + 1], individual_product_psi
                )
            else:
                individual_product_psi = ApplyGate(
                    U, [q, q + 1], individual_product_psi
                )

        if pbc and n_qubits % 2 != odd:
            individual_product_psi = ApplyGate(
                U, [n_qubits - 1, 0], individual_product_psi
            )

        psi_result = psi_result + individual_product_psi

    # add the product rule term over the last gate, if it exists
    if pbc and n_qubits % 2 != odd:
        individual_product_psi = psi.clone()

        for q in range(odd, n_qubits - 1, 2):
            individual_product_psi = ApplyGate(U, [q, q + 1], individual_product_psi)

        individual_product_psi = ApplyGate(
            d_U, [n_qubits - 1, 0], individual_product_psi
        )

        psi_result = psi_result + individual_product_psi

    return psi_result


def layer_paulipauli_derivative_yyxx(n_qubits, psi, theta, odd, pbc=False):
    UYY = rotrot(theta, 2, False)
    d_UYY = rotrot(theta, 2, True)

    #     print("UYY", UYY)
    #     print(d_UYY)

    UXX = rotrot(theta, 1, False)
    d_UXX = rotrot(theta, 1, True)

    psi_result = np.zeros(psi.shape)

    # index over all the gates with the same parameter
    for derivative_UXX_gate in range(2):
        for derivative_index in range(odd, (n_qubits - 1), 2):
            individual_product_psi = psi.clone()

            # apply first all the UYY gates over the middle qubits
            for q in range(odd, n_qubits - 1, 2):
                # apply the special product rule UYY gate if applicable
                if derivative_index == q and derivative_UXX_gate == 0:
                    individual_product_psi = ApplyGate(
                        d_UYY, [q, q + 1], individual_product_psi
                    )
                else:
                    individual_product_psi = ApplyGate(
                        UYY, [q, q + 1], individual_product_psi
                    )

            if pbc and n_qubits % 2 != odd:
                individual_product_psi = ApplyGate(
                    UYY, [n_qubits - 1, 0], individual_product_psi
                )

            # now apply all the UXX gates over the middle qubits
            for q in range(odd, n_qubits - 1, 2):
                # apply the special product rule UXX gate if applicable
                if derivative_index == q and derivative_UXX_gate == 1:
                    individual_product_psi = ApplyGate(
                        d_UXX, [q, q + 1], individual_product_psi
                    )

                else:
                    individual_product_psi = ApplyGate(
                        UXX, [q, q + 1], individual_product_psi
                    )

            if pbc and n_qubits % 2 != odd:
                individual_product_psi = ApplyGate(
                    UXX, [n_qubits - 1, 0], individual_product_psi
                )

            psi_result = psi_result + individual_product_psi

    # add the product rule term over the last gate, if it exists
    if pbc and n_qubits % 2 != odd:
        # first, find the product rule term for where the derivative of the end
        # UYY gate is taken
        individual_product_psi = psi.clone()
        # apply the regular UYY gates
        for q in range(odd, n_qubits - 1, 2):
            individual_product_psi = ApplyGate(UYY, [q, q + 1], individual_product_psi)

        # apply the product rule UYY gate
        individual_product_psi = ApplyGate(
            d_UYY, [n_qubits - 1, 0], individual_product_psi
        )

        # apply the regular UXX gates
        for q in range(odd, n_qubits - 1, 2):
            individual_product_psi = ApplyGate(UXX, [q, q + 1], individual_product_psi)

        # apply the regular UXX end qubit gate
        individual_product_psi = ApplyGate(
            UXX, [n_qubits - 1, 0], individual_product_psi
        )

        psi_result = psi_result + individual_product_psi

        # now do the same thing except for the derivative of the UXX gate
        individual_product_psi = psi.clone()
        # apply the regular UYY gates
        for q in range(odd, n_qubits - 1, 2):
            individual_product_psi = ApplyGate(UYY, [q, q + 1], individual_product_psi)

        # apply the regular UYY gate
        individual_product_psi = ApplyGate(
            UYY, [n_qubits - 1, 0], individual_product_psi
        )

        # apply the regular UXX gates
        for q in range(odd, n_qubits - 1, 2):
            individual_product_psi = ApplyGate(UXX, [q, q + 1], individual_product_psi)

        # apply the derivative UXX end qubit gate
        individual_product_psi = ApplyGate(
            d_UXX, [n_qubits - 1, 0], individual_product_psi
        )

        psi_result = psi_result + individual_product_psi

    return psi_result


def layer_paulixpaul_it_numeric(
    n_qubits, psi_list, parameters, rotation, current_param, gradient_index, dtheta
):
    raise NotImplementedError


def layer_x_rot_HVA(n_qubits, psi_list, param):
    for q in range(n_qubits):
        for i in range(len(psi_list)):
            psi_list[i] = ApplyGate(rot(param, 2), [q], psi_list[i])


def layer_paulipauli(n_qubits, psi, theta, odd, pauli, pbc=False):
    if pauli == "xx":
        M = np.mul(-1j * theta / 2, xx)
    elif pauli == "yy":
        M = np.mul(-1j * theta / 2, yy)
    elif pauli == "zz":
        M = np.mul(-1j * theta / 2, zz)
    else:
        raise ValueError("double pauli gate pauli is invalid")

    U = np.matrix_exp(M).reshape(2, 2, 2, 2)

    for q in range(n_qubits - 1):
        if q % 2 == odd:
            psi = ApplyGate(U, [q, q + 1], psi)
    if pbc and n_qubits % 2 != odd:
        psi = ApplyGate(U, [n_qubits - 1, 0], psi)

    return psi


def dividing_measurement_gates(n_qubits, n_layers, n_sub_circuits):
    step = n_layers // n_sub_circuits
    points = [[step * i - 1, n_qubits // 2 - 1] for i in range(1, n_sub_circuits)]
    return points


def renyi_entropy(psi, n_qubits, qubits_to_keep):
    """
    returns the renyi entropy given psi.
    """

    basis_spin = quspin.basis.spin_basis_general(n_qubits, S="1/2")
    psi = np.reshape(psi, (2 ** (len(psi.shape))))
    rdm = basis_spin.partial_trace(psi, qubits_to_keep)
    return -np.log2(np.trace(rdm**2).real)


def generate_entropies(psi, n_qubits, entropy_regions):
    entropy_results = []

    for qubit_region in entropy_regions:
        entropy_results.append(renyi_entropy(psi, n_qubits, qubit_region))

    return entropy_results


# This code down here is kept

# import openfermion as of

# Initialize an empty QubitOperator to store the full Hamiltonian
# full_hamiltonian = of.QubitOperator()

# # Loop through each adjacent pair of qubits in the 12-qubit chain
# for i in range(n_qubits-1):  # from 0 to 10
#     xx_term = of.QubitOperator(f"X{i} X{i+1}", 1)
#     yy_term = of.QubitOperator(f"Y{i} Y{i+1}", 1)
#     zz_term = of.QubitOperator(f"Z{i} Z{i+1}", .5)

#     # Add these terms to the full Hamiltonian
#     full_hamiltonian += xx_term
#     full_hamiltonian += yy_term
#     full_hamiltonian += zz_term

# xx_term = of.QubitOperator(f"X{1} X{0}", 1)
# yy_term = of.QubitOperator(f"Y{1} Y{0}", 1)
# zz_term = of.QubitOperator(f"Z{1} Z{0}", .5)

# full_hamiltonian += xx_term
# full_hamiltonian += yy_term
# full_hamiltonian += zz_term
# # Convert to a sparse array
# full_ham_array = of.get_sparse_operator(full_hamiltonian)

# from openfermion.linalg import get_ground_state

# ground_energy, a = get_ground_state(full_ham_array)

# print(f"The ground state energy is {ground_energy}")
