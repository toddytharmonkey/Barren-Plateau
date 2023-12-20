from scipy.optimize import minimize
import ast
import openfermion as of
from dask.diagnostics import ProgressBar
from dask import delayed, compute
import matplotlib.pyplot as plt
from scipy.stats import bootstrap
from tqdm import tqdm  # assuming my code is going to be run in notebook
from collections import namedtuple
import torch
import random
import os
import numpy as np
from dask.distributed import Client, as_completed
import quspin

code_version = "1.2"

# device = torch.device("cuda" if torch.cuda.is_available() else
#                       SystemError("Cuda was not chosen"))
device = torch.device("cpu")

ham_1_1_4 = of.get_sparse_operator(of.QubitOperator(
    'X0 X1', 1)+of.QubitOperator('Y0 Y1', 1)+of.QubitOperator('Z0 Z1', 4)).toarray()
ham_1_1_4 = torch.reshape(torch.tensor(ham_1_1_4, dtype=torch.complex64),
                          tuple([2 for i in range(int(np.log2(ham_1_1_4 .size)))]))

ham_1_1_05 = of.get_sparse_operator(of.QubitOperator(
    'X0 X1', 1)+of.QubitOperator('Y0 Y1', 1)+of.QubitOperator('Z0 Z1', .5)).toarray()
ham_1_1_05 = torch.reshape(torch.tensor(ham_1_1_05, dtype=torch.complex64),
                           tuple([2 for i in range(int(np.log2(ham_1_1_05.size)))]))

"""
Generate gradient variance shots using 3 possible methods for different ansatz.

Code can run on the GPU if a device is available.

Sonny Rappaport, Gaurav Gyawali, Michael Lawler, March 2023
"""
# Basic Wavefunction Manipulation/Creations-----------------------------------


def ApplyGate(U, qubits, psi):
    ''''Multiplies a state psi by gate U acting on qubits'''

    indices = ''.join([chr(97+q) for q in qubits])
    indices += ''.join([chr(65+q) for q in qubits])
    indices += ','
    indices += ''.join([chr(97+i-32*qubits.count(i))
                       for i in range(len(psi.shape))])

#     print("U", U)
#     print("qubits", qubits)
#     print("indices",indices)

    return torch.einsum(indices, U, psi)


def Inner(psi_1, psi_2):
    '''<psi_1|psi_2>'''
    indices = ''.join([chr(97+q) for q in range(len(psi_1.shape))])
    indices += ','
    indices += ''.join([chr(97+q) for q in range(len(psi_2.shape))])
    return torch.einsum(indices, psi_1.conj(), psi_2)


def Basis(n):
    '''Returns the computational basis states for n qubits'''
    def i2binarray(i):
        return [int(c) for c in bin(i)[2:].zfill(n)]
    return [i2binarray(i) for i in range(2**n)]


def initial_state(n_qubits):
    '''Initializes the qubits on the computational basis'''
    zero = torch.tensor([1, 0], dtype=torch.complex64, device=device)
    psi = torch.tensor([1, 0], dtype=torch.complex64, device=device)
    for i in range(n_qubits-1):
        psi = torch.kron(psi, zero)
    return psi.reshape((2,)*n_qubits)


# Gate Definitons--------------------------------------------------------------

def pauli(i):
    '''Pauli matrix. i = 0 for I, 1 for X, 2 for Y, 3 for Z'''
    if i == 0:
        return torch.eye(2)
    elif i == 1:
        return torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
    elif i == 2:
        return torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
    elif i == 3:
        return torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)
    else:
        return ValueError("i=0,1,2,3 only")


CNOT = torch.reshape(torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [
    0, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.complex64, device=device), (2, 2, 2, 2))

H = 1/np.sqrt(2)*torch.tensor([[1, 1], [1, -1]],
                              dtype=torch.complex64, device=device)

zz = torch.kron(pauli(3), pauli(3))
xx = torch.kron(pauli(1), pauli(1))
yy = torch.kron(pauli(2), pauli(2))


def paulipauli(i):
    '''Pauli matrix. i = 0 for I, 1 for XX, 2 for YY, 3 for ZZ'''

    if i == 0:
        return torch.kron(torch.eye(2), torch.eye(2)).reshape(2, 2, 2, 2)
    elif i == 1:
        return xx
    elif i == 2:
        return yy
    elif i == 3:
        return zz
    else:
        return ValueError("i=0,1,2,3 only")


def rot(theta, i, grad=0):
    '''Rotation gate. i = 1 for x, 2 for y, 3 for z'''
    if not grad:
        return np.cos(theta/2)*torch.eye(2, dtype=torch.complex64,
                                         device=device) - 1j*np.sin(theta/2)*pauli(i)
    else:
        return -.5*np.sin(theta/2)*torch.eye(2, dtype=torch.complex64,
                                             device=device) - .5j*np.cos(theta/2)*pauli(i)


def rotrot(theta, i, grad=0):
    '''Rotation gate. i = 1 for x, 2 for y, 3 for z'''
    if not grad:
        return (np.cos(theta/2)*torch.eye(4, dtype=torch.complex64,
                                          device=device) - 1j*np.sin(theta/2)*paulipauli(i)).reshape(2, 2, 2, 2)
    else:
        return (-.5*np.sin(theta/2)*torch.eye(4, dtype=torch.complex64,
                                              device=device) - .5j*np.cos(theta/2)*paulipauli(i)).reshape(2, 2, 2, 2)


def measure(i):
    '''measurement operator on 0 or 1'''
    if i == 0:
        return torch.tensor([[1., 0], [0, 0]], dtype=torch.complex64, device=device)
    elif i == 1:
        return torch.tensor([[0, 0], [0, 1.]], dtype=torch.complex64, device=device)
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
    'applies the XXZ hamiltonian to psi'
    psi_original = psi
    psi_cost = ApplyGate(ham_1_1_4, [0, 1], psi_original)

#     print("n_qubits", n_qubits)

    for i in range(1, n_qubits-1):
        psi_cost = psi_cost + ApplyGate(ham_1_1_4, [i, i+1], psi_original)

    if periodic:
        psi_cost = psi_cost + \
            ApplyGate(ham_1_1_4, [n_qubits-1, 0], psi_original)

    return psi_cost


def apply_XXZ_1_1_05(psi, periodic, n_qubits):
    'applies the XXZ hamiltonian to psi'
    psi_original = psi
    psi_cost = ApplyGate(ham_1_1_05, [0, 1], psi_original)

#     print("n_qubits", n_qubits)

    for i in range(1, n_qubits-1):
        psi_cost = psi_cost + ApplyGate(ham_1_1_05, [i, i+1], psi_original)

    if periodic:
        psi_cost = psi_cost + \
            ApplyGate(ham_1_1_05, [n_qubits-1, 0], psi_original)

    return psi_cost


def apply_Z0Z1(psi):
    'applies z0z1 to the first two qubits'
    return ApplyGate(zz.reshape(2, 2, 2, 2), [0, 1], psi)


def prob_rounder(p0, i):
    '''
    adjusts rounding errors, particularly when the result is very close to 0
    or 1. the code in this python file is vulnrible to this error.
    '''

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
    if ansatz == 'HEA2':
        return n_layers * 2
    elif ansatz == 'HEA1':
        return n_qubits * n_layers * 2
    elif ansatz == 'HEA2_uber_parameters':
        return n_qubits * n_layers * 2 
    elif ansatz == 'HVA':
        return n_layers * 4  # HVA
    else:
        raise ValueError(f'Input ansatz not identified, you entered {ansatz}')


def random_parameters(num):
    """
    returns returns every individual parameter for the HEA ansatz. This means
    every individual parameter in the returned list is repeated over a whole
    row of qubits.
    """

    return np.random.uniform(low=-4*np.pi, high=4*np.pi,
                             size=num)


def random_rotations(num):
    return np.random.randint(low=1, high=4, size=num)


def random_measurements_prob(n_layers, n_qubits, chance):
    '''generate a random list of measurements, with 'chance' that one is
    placed at a given location'''

    measure_list = []

    for depth in range(n_layers-1):
        for qubit_num in range(n_qubits):

            if chance > np.random.rand():
                measure_list.append([depth, qubit_num])

    if len(measure_list) == 0:
        return None
    else:
        return measure_list


def random_measurements_num(n_layers, n_qubits, size):
    '''generate a random list of 'size' measurements.'''

    measure_list = []

    for depth in range(n_layers-1):
        for qubit_num in range(n_qubits):

            measure_list.append([depth, qubit_num])

    measure_list = random.sample(measure_list, k=size)

    if len(measure_list) == 0:
        return None
    else:
        return measure_list


def layer_measure(psi_list, p, gradient_technique, post_selected, measurements, layer, specific_measurement, measurement_index):
    if measurements is not None:
        measurements = np.array(measurements)
        relevant_qubits = measurements[np.where(
            measurements[:, 0] == layer)][:, 1:]
        if len(relevant_qubits):
            return apply_measure(psi_list, relevant_qubits.tolist(),
                          p, gradient_technique, post_selected, specific_measurement, measurement_index)
        else:
            return []


def apply_measure(psi_list, measurements, pM, gradient_technique,
                  post_selected=True, specific_measurements=None, measurement_index=0):
    '''project the wavefunctions onto some randomly sampled basis
    returns the unnormalized (conditional) probabiliy
    This is the most subtle part of the entire code so be careful
    We do not normalize psi at each step so the output wavefunctions are
    \tilde{\psi}'''

    outcomes = []

    for q in measurements:

        outcome = 0  # default outcome

        original_p = 1

        for i in range(len(psi_list)):

            p0 = torch.abs(
                Inner(ApplyGate(measure(0), q, psi_list[i]), psi_list[i])).to("cpu").numpy()

            p0, p1 = prob_rounder(p0, i)

            # if not post-selected, make the outcome from the first (normal) circuit
            if i == 0 and specific_measurements != None:
                outcome = specific_measurements[measurement_index]
                measurement_index = measurement_index + 1 
            elif i == 0 and post_selected == False:
                outcome = np.random.choice([0, 1], 1, p=[p0, p1])[0]
            elif i == 0 and post_selected == True:
                outcome = 0

            if i == 0:
                outcomes.append(outcome)

            if i == 0:
                original_p = [p0, p1][outcome]

            p_i = [p0, p1][outcome]

            pM[i] = pM[i]*p_i

            if gradient_technique == "analytic":
                psi_list[i] = ApplyGate(
                    measure(outcome), q, psi_list[i])/np.sqrt(original_p)
            else:
                psi_list[i] = ApplyGate(
                    measure(outcome), q, psi_list[i])/np.sqrt(p_i)
                
    return outcomes


def gradients_by_layer(n_qubits, n_layers, parameters, gradient_technique="numeric",
                       gradient_index=0, measurements=None, dtheta=.00001,
                       return_analytic_suite=False, post_selected=False,
                       entropy_regions=[[]], periodic=False,
                       get_layered_results=False, ham_type="z0z1",
                       ansatz="GG", rotations=None,specific_measurement= None):
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

    if gradient_technique != "shift" and gradient_technique != "analytic" and gradient_technique != "numeric":
        raise ValueError("gradient_technique is not a valid type")

    if ansatz == "HEA2":

        return HEA_gradient_by_layer(n_qubits, n_layers, parameters,
                                     gradient_technique,
                                     gradient_index, measurements, dtheta,
                                     return_analytic_suite, post_selected,
                                     entropy_regions, periodic,
                                     get_layered_results, ham_type,specific_measurement)

    if ansatz == "HEA2_uber_parameters":

        return HEA_uber_gradient_by_layer(n_qubits, n_layers, parameters,
                                     gradient_technique,
                                     gradient_index, measurements, dtheta,
                                     return_analytic_suite, post_selected,
                                     entropy_regions, periodic,
                                     get_layered_results, ham_type,)
    elif ansatz == "HEA1":

        return GG_gradient_by_layer(n_qubits, n_layers, parameters, rotations,
                                    gradient_technique,
                                    gradient_index, measurements, dtheta,
                                    return_analytic_suite, post_selected,
                                    entropy_regions, periodic,
                                    get_layered_results, ham_type, )

    elif ansatz == "HVA":
        return HVA_gradient_by_layer(n_qubits, n_layers, parameters,
                                     gradient_technique,
                                     gradient_index, measurements, dtheta,
                                     return_analytic_suite, post_selected,
                                     entropy_regions, periodic,
                                     get_layered_results, ham_type, )

# HEA2 uber parameters specific code ----------------------------------------------------------

def HEA_uber_gradient_by_layer(n_qubits, n_layers, parameters, gradient_technique="numeric",
                         gradient_index=0, measurements=None, dtheta=.00001,
                         return_analytic_suite=False, post_selected=False,
                         entropy_regions=[[]], periodic=False,
                         get_layered_results=False, ham_type="z0z1", return_psi_list = False):

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
        layer_rot_GG(n_qubits, psi_list, parameters, 2, current_param,
                     gradient_index, one_parameter=True)
        current_param += n_qubits

        # 2: CNOT between qth and (q+1)th qubits where q is even
        layer_CNOT_ladder(n_qubits, psi_list, periodic)

        # 3: Rotation gates on all qubits
        # with analytical gradient
        layer_rot_GG(n_qubits, psi_list, parameters,
                     1, current_param, gradient_index, one_parameter = True)
        current_param += n_qubits

        layer_measure(psi_list, psi_list_probabilities, gradient_technique,
                      post_selected, measurements, l)

        if get_layered_results:
            layer_results.append(gradients_GG(
                n_qubits, psi_list, gradient_technique, psi_list_probabilities,
                dtheta, return_analytic_suite, entropy_regions,
                ham_type, periodic))
    if return_psi_list:
        return psi_list
    if get_layered_results:
        return layer_results
    else:
        #         print(psi_list)
        return gradients_GG(n_qubits, psi_list, gradient_technique, psi_list_probabilities,
                             dtheta, return_analytic_suite, entropy_regions,
                             ham_type, periodic)

# HEA VQE specific code------------------------------------------------------------------------


def GG_gradient_by_layer(n_qubits, n_layers, parameters, rotations, gradient_technique="numeric",
                         gradient_index=0, measurements=None, dtheta=.00001,
                         return_analytic_suite=False, post_selected=False,
                         entropy_regions=[[]], periodic=False,
                         get_layered_results=False, ham_type="z0z1",):

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
        layer_rot_GG(n_qubits, psi_list, parameters, rotations, current_param,
                     gradient_index)
        current_param += n_qubits

        # 2: CNOT between qth and (q+1)th qubits where q is even
        layer_CNOT_even(n_qubits, psi_list)

        # 3: Rotation gates on all qubits
        # TODO- align this with the other layer_rots, also only currently works
        # with analytical gradient
        layer_rot_GG(n_qubits, psi_list, parameters,
                     rotations, current_param, gradient_index)
        current_param += n_qubits

        # 4: CNOT between qth and (q+1)th qubits where q is even
        layer_CNOT_odd(n_qubits, psi_list)

        layer_measure(psi_list, psi_list_probabilities, gradient_technique,
                      post_selected, measurements, l)
        
        if get_layered_results:
            layer_results.append(gradients_GG(
                n_qubits, psi_list, gradient_technique, psi_list_probabilities,
                dtheta, return_analytic_suite, entropy_regions,
                ham_type, periodic))

    if get_layered_results:
        return layer_results
    else:
        #         print(psi_list)
        return gradients_GG(n_qubits, psi_list, gradient_technique, psi_list_probabilities,
                             dtheta, return_analytic_suite, entropy_regions,
                             ham_type, periodic)


    return layer_results


def gradients_GG(n_qubits, psi_list, gradient_technique, p, dtheta,
                 return_analytic_suite, entropy_regions, ham_type,
                 periodic):
    '''
    Given a psi_list and a gradient_technique, returns various useful inner
    products such as the gradients or cost function.

    If return_analytic_suite is true, if the gradient_technique is analytic then
    the renyi entropy and cost function are returned as well as the gradients.
    '''

    psi = psi_list[0]
    cost_psi = ApplyHam(psi, ham_type, n_qubits, periodic)

    if gradient_technique == 'numeric':
        psi_plus = psi_list[1]
        cost_psi_plus = ApplyHam(psi, ham_type, n_qubits, periodic)

        psi_minus = psi_list[2]
        cost_psi_minus = ApplyHam(psi, ham_type, n_qubits, periodic)

        return (Inner(psi_plus, cost_psi_plus).real-Inner(psi_minus,
                                                          cost_psi_minus).real)/(2*dtheta)

    elif gradient_technique == 'shift':

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
        unaware_gradient = 2*term1.real
        aware_gradient = 2*term2.real

        if return_analytic_suite:
            return C.cpu().numpy(), unaware_gradient.cpu().numpy(), aware_gradient.cpu().numpy()
        else:
            return unaware_gradient.cpu().numpy(), aware_gradient.cpu().numpy()


def layer_rot_GG(n_qubits, psi_list, parameters, rotations, current_param,
                 gradient, one_parameter = False):

    for q in range(n_qubits):
        for i in range(len(psi_list)):
            param = parameters[current_param]
            take_grad = 1 if (current_param == gradient and i == 1) else 0
            if not one_parameter:
                psi_list[i] = ApplyGate(
                    rot(param, rotations[current_param], take_grad), [q],
                    psi_list[i])
            else: 
                psi_list[i] = ApplyGate(
                    rot(param, rotations, take_grad), [q],
                    psi_list[i])
        current_param += 1


def layer_y_rot_GG(n_qubits, psi_list, param):
    for q in range(n_qubits):
        for i in range(len(psi_list)):
            psi_list[i] = ApplyGate(rot(param, 2), [q], psi_list[i])


def layer_CNOT_even(n_qubits, psi_list, periodic_boundary=False):
    '''apply a CNOT between an qth and (q+1)th qubits where q is even'''
    for i in range(len(psi_list)):
        for q in range(n_qubits-1):
            if q % 2 == 0:
                psi_list[i] = ApplyGate(CNOT, [q, q+1], psi_list[i])
        if periodic_boundary == True and n_qubits % 2 == 1:
            psi_list[i] = ApplyGate(CNOT, [n_qubits-1, 0], psi_list[i])


def layer_CNOT_odd(n_qubits, psi_list, periodic_boundary=False):
    '''apply a CNOT between an qth and (q+1)th qubits where q is odd'''
    for i in range(len(psi_list)):
        for q in range(n_qubits-1):
            if q % 2 != 0:
                psi_list[i] = ApplyGate(CNOT, [q, q+1], psi_list[i])
        if periodic_boundary == True and n_qubits % 2 == 0:
            psi_list[i] = ApplyGate(CNOT, [n_qubits-1, 0], psi_list[i])



# HEA2 VQE specific code---------------------------------------------------------

def HEA_gradient_by_layer(n_qubits, n_layers, parameters, gradient_technique="numeric",
                          gradient_index=0, measurements=None, dtheta=.00001,
                          return_analytic_suite=False, post_selected=False,
                          entropy_regions=[[]], periodic=False,
                          get_layered_results=False, ham_type="z0z1",specific_measurement=None):

    layer_results = []
    outcomes = []
    outcome_index = 0 

    if gradient_technique == "shift":
        # create nonshifted, -, + versions of psi , for each individually shifted psi
        psi_list = torch.stack([initial_state(n_qubits)
                               for _ in range(3*n_qubits)])
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

    p_by_layer = []

    for l in range(n_layers):
        # 1: paramaterized Ry rotation gates on all qubits

        # print("New Layer")
        # print(psi_list)

        layer_rot(n_qubits, psi_list, parameters, 2, current_param,
                  gradient_technique, gradient_index, dtheta)
        current_param += 1

        # # 2: CNOTs connecting from the top to bottom qubit
        layer_CNOT_ladder(n_qubits, psi_list, periodic)

        # 3: Rotation gates on all qubits
        layer_rot(n_qubits, psi_list, parameters, 1, current_param,
                  gradient_technique, gradient_index, dtheta)
        current_param += 1

        if get_layered_results:
            layer_results.append(gradients_HEA(
                n_qubits, psi_list, gradient_technique, p,
                dtheta, return_analytic_suite, entropy_regions,
                ham_type, periodic))

        # If measurements are provided apply them
        measurement_outcome = layer_measure(psi_list, p, gradient_technique,
                      post_selected, measurements, l, specific_measurement, outcome_index)
        if measurement_outcome != None:
            outcomes.extend(measurement_outcome)
            #print(measurement_outcome)
            outcome_index = outcome_index + len(measurement_outcome)
        p_by_layer.append(p[0])
        
    if get_layered_results:
        return layer_results, outcomes,p_by_layer
    else:
        return gradients_HEA(n_qubits, psi_list, gradient_technique, p,
                             dtheta, return_analytic_suite, entropy_regions,
                             ham_type, periodic), outcomes


def layer_H(n_qubits, psi_list, qubit_range):
    '''apply a rotation layer of Hadamard gates to all qubits'''
    for i in range(len(psi_list)):
        for q in range(n_qubits):
            psi_list[i] = ApplyGate(H, [q], psi_list[i])


def gradients_HEA(n_qubits, psi_list, gradient_technique, p, dtheta,
                  return_analytic_suite, entropy_regions, ham_type,
                  periodic):
    '''
    Given a psi_list and a gradient_technique, returns various useful inner
    products such as the gradients or cost function.

    If return_analytic_suite is true, if the gradient_technique is analytic then
    the renyi entropy and cost function are returned as well as the gradients.
    '''

    if entropy_regions == True:
        return generate_entropies(psi_list[0],n_qubits, [[q for q in range(n_qubits//2)],[0],[n_qubits//2]])

    psi = psi_list[0]
    cost_psi = ApplyHam(psi, ham_type, n_qubits, periodic)

    if gradient_technique == 'numeric':

        psi_plus = psi_list[1]
        cost_psi_plus = ApplyHam(psi, ham_type, n_qubits, periodic)

        psi_minus = psi_list[2]
        cost_psi_minus = ApplyHam(psi, ham_type, n_qubits, periodic)

        return (Inner(psi_plus, cost_psi_plus).real-Inner(psi_minus, cost_psi_minus).real)/(2*dtheta)

    elif gradient_technique == 'shift':

        aware_gradient = 0
        unaware_gradient = 0

        grouped_psi_list = torch.reshape(psi_list, tuple(
            [n_qubits, 3, *[2 for _ in range(n_qubits)]]))

        grouped_p = np.reshape(p, (n_qubits, 3))

        for k, psi_list in enumerate(grouped_psi_list):
            O = []
            for psi in psi_list:
                cost_psi = ApplyHam(psi, ham_type, n_qubits, periodic)
                O.append(Inner(psi, cost_psi).real)
            cost_p, cost_p_plus, cost_p_minus = O
            prob, prob_plus, prob_minus = grouped_p[k]

            aware_gradient += 0.5 * \
                ((cost_p_plus*prob_plus - cost_p_minus*prob_minus) /
                 prob - (prob_plus-prob_minus)*cost_p/prob)
            unaware_gradient += 0.5 * \
                (cost_p_plus*prob_plus - cost_p_minus*prob_minus)/prob

        return unaware_gradient, aware_gradient

    else:  # analytical gradient

        C = Inner(psi, cost_psi).real
        return C

        # Calculate the gradients using eqn 22 in the notes. Note we are not multiplying by p because of MC sampling
        term1 = Inner(psi_list[1], cost_psi)  # first term
        # first term - second term
        term2 = term1 - C * Inner(psi_list[1], psi_list[0])

        # Note that the second term is being divided by only prob instead of prob^2 because the cost function is normalized here
        unaware_gradient = 2*term1.real
        aware_gradient = 2*term2.real

        if return_analytic_suite:
            return C.cpu().numpy(), unaware_gradient.cpu().numpy(), aware_gradient.cpu().numpy()
        else:
            return unaware_gradient.cpu().numpy(), aware_gradient.cpu().numpy()


def layer_rot(n_qubits, psi_list, parameters, rotation, current_param,
              gradient_technique, gradient_index, dtheta,):

    if gradient_technique == "shift":
        layer_rot_shift(n_qubits, psi_list, parameters,
                        rotation, current_param, gradient_index)
    elif gradient_technique == "analytic":
        layer_rot_analytical(n_qubits, psi_list, parameters,
                             rotation, current_param, gradient_index)
    else:
        layer_rot_numeric(n_qubits, psi_list, parameters,
                          rotation, current_param, gradient_index, dtheta)


def layer_rot_shift(n_qubits, psi_list, parameters, rotation, current_param, gradient_index):

    p = (0, np.pi/2, -np.pi/2)

    # group the psis into n_qubit groups of threes to account for the product rule
    grouped_psi_list = torch.reshape(psi_list, tuple(
        [n_qubits, 3, *[2 for _ in range(n_qubits)]]))

    # for every kth psi whose derivative is taken in the kth row:
    for k in range(len(grouped_psi_list)):
        for q in range(n_qubits):  # over every qubit:
            for i in range(3):  # over each nonshifted, -, + psi
                # if the gradient index is the current parameter and this is the correct psi
                if gradient_index == current_param and k == q:
                    grouped_psi_list[k, i] = ApplyGate(
                        rot(parameters[current_param] + p[i], rotation), [q],
                        grouped_psi_list[k, i])
                # otherwise, just apply the regular (identical) parameter over the whole column
                else:
                    grouped_psi_list[k, i] = ApplyGate(rot(parameters[current_param], rotation), [
                                                       q], grouped_psi_list[k, i])


def row_with_one_gradient(n_qubits, psi, parameter, rotation, gradient_qubit):

    for q in range(n_qubits):
        psi = ApplyGate(rot(parameter, rotation, q ==
                        gradient_qubit), [q], psi)
    return psi


def product_rule_psi(n_qubits, psi, parameter, rotation):

    original_psi = psi.clone()
    psi_result = torch.zeros(psi.shape, device=device)
    for q in range(n_qubits):
        gradient_psi_term = row_with_one_gradient(
            n_qubits, original_psi, parameter, rotation, q)

        psi_result = psi_result + gradient_psi_term
    return psi_result


def layer_rot_analytical(n_qubits, psi_list, parameters, rotation,
                         current_param, gradient_index):
    for i in range(len(psi_list)):
        # if this is del_psi and it's time to take a derivative:
        if current_param == gradient_index and i == 1:

            psi_list[i] = product_rule_psi(
                n_qubits, psi_list[i], parameters[current_param], rotation)

        else:
            psi_list[i] = rotation_layer(
                n_qubits, psi_list[i], parameters[current_param], rotation)


def layer_CNOT_ladder(n_qubits, psi_list, periodic_boundary=False):
    '''apply a CNOT between an qth and (q+1)th qubits where q is even'''
    for i in range(len(psi_list)):
        for q in range(n_qubits-1):
            psi_list[i] = ApplyGate(CNOT, [q, q+1], psi_list[i])
        if periodic_boundary == True:
            psi_list[i] = ApplyGate(CNOT, [n_qubits-1, 0], psi_list[i])


def layer_rot_numeric(n_qubits, psi_list, parameters, rotation, current_param, gradient_index, dtheta):
    if gradient_index != current_param:
        for i in range(len(psi_list)):
            psi_list[i] = rotation_layer(
                n_qubits, psi_list[i], parameters[current_param], rotation)
    else:
        p = (parameters[current_param], parameters[current_param] +
             dtheta, parameters[current_param] - dtheta)
        for q in range(n_qubits):
            for i in range(len(psi_list)):
                psi_list[i] = ApplyGate(rot(p[i], rotation), [q], psi_list[i])


def rotation_layer(n_qubits, psi, parameter, rotation):
    '''
    applies an identical rotation gate over all the qubits in psi.
    '''
    for q in range(n_qubits):
        psi = ApplyGate(rot(parameter, rotation), [q], psi)

    return psi

# HVA VQE specific code----------------------------------------------------------


def HVA_gradient_by_layer(n_qubits, n_layers, parameters, gradient_technique="numeric",
                          gradient_index=0, measurements=None, dtheta=.00001,
                          return_analytic_suite=False, post_selected=False,
                          entropy_regions=[[]], periodic=False,
                          get_layered_results=False, ham_type="z0z1",):

    layer_results = []

    if gradient_technique == "shift":
        # create nonshifted, -, + versions of psi , for each individually shifted psi
        psi_list = torch.stack([initial_state(n_qubits)
                               for _ in range(3*n_qubits)])
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

        layer_paulixpauli(n_qubits, psi_list, parameters, 0,
                          current_param, gradient_technique, gradient_index,
                          dtheta)
        current_param += 1

        layer_paulixpauli(n_qubits, psi_list, parameters, 1,
                          current_param, gradient_technique, gradient_index,
                          dtheta)
        current_param += 1

        layer_paulixpauli(n_qubits, psi_list, parameters, 2,
                          current_param, gradient_technique, gradient_index,
                          dtheta)
        current_param += 1

        layer_paulixpauli(n_qubits, psi_list, parameters, 3,
                          current_param, gradient_technique, gradient_index,
                          dtheta)
        current_param += 1

        # If measurements are provided apply them
        layer_measure(psi_list, p, gradient_technique,
                      post_selected, measurements, l)

        if get_layered_results:
            layer_results.append(gradients_HVA(
                n_qubits, psi_list, gradient_technique, p,
                dtheta, return_analytic_suite, entropy_regions,
                ham_type, periodic))

    if get_layered_results:
        return layer_results
    else:
        return gradients_HVA(n_qubits, psi_list, gradient_technique, p,
                             dtheta, return_analytic_suite, entropy_regions,
                             ham_type, periodic)


def gradients_HVA(n_qubits, psi_list, gradient_technique, p, dtheta,
                  return_analytic_suite, entropy_regions, ham_type,
                  periodic):

    psi = psi_list[0]
    cost_psi = ApplyHam(psi, ham_type, n_qubits, periodic)

    C = Inner(psi, cost_psi).real

    # Calculate the gradients using eqn 22 in the notes. Note we are not multiplying by p because of MC sampling
    term1 = Inner(psi_list[1], cost_psi)  # first term
    # first term - second term
    term2 = term1 - C * Inner(psi_list[1], psi_list[0])

    # Note that the second term is being divided by only prob instead of prob^2 because the cost function is normalized here
    unaware_gradient = 2*term1.real
    aware_gradient = 2*term2.real

    if return_analytic_suite:
        return C.cpu().numpy(), unaware_gradient.cpu().numpy(), aware_gradient.cpu().numpy()
    else:
        return unaware_gradient.cpu().numpy(), aware_gradient.cpu().numpy()


def layer_paulixpauli(n_qubits, psi_list, parameters, type_gates,
                      current_param, gradient_technique, gradient_index,
                      dtheta):
    """

    Puts down a layer of double pauli gates in the fashion of the Roeland paper.
    Type 1 is odd double ZZ gates, Type 2 is odd double YY and XX gates,
    Type 3 is even ZZ gates, Type 3 is even YY and XX

    """

    if gradient_technique == "shift":
        layer_paulixpauli_shift(n_qubits, psi_list, parameters,
                                type_gates, current_param, gradient_index)
    elif gradient_technique == "analytic":
        layer_paulixpauli_analytical(n_qubits, psi_list, parameters,
                                     type_gates, current_param, gradient_index)
    else:
        layer_paulixpaul_it_numeric(n_qubits, psi_list, parameters,
                                    type_gates, current_param, gradient_index, dtheta)


def layer_paulixpauli_shift(n_qubits, psi_list, parameters,
                            rotation, current_param, gradient_index):
    raise NotImplementedError


def rotation_layer_HVA_gradient(n_qubits, psi, parameter, type_gate):

    if type_gate == 0:
        psi = layer_paulipauli_derivative_zz(
            n_qubits, psi, parameter, True, pbc=True)
    elif type_gate == 1:
        psi = layer_paulipauli_derivative_yyxx(n_qubits, psi, parameter,
                                               True, pbc=True)
    elif type_gate == 2:
        psi = layer_paulipauli_derivative_zz(
            n_qubits, psi, parameter, False, pbc=True)
    elif type_gate == 3:
        psi = layer_paulipauli_derivative_yyxx(n_qubits, psi, parameter,
                                               False, pbc=True)
    else:
        raise ValueError("type_gate is not a number from 0-3 inclusive")

    return psi


def rotation_layer_HVA(n_qubits, psi, parameter, type_gate, gradient=False):

    if type_gate == 0:
        psi = layer_paulipauli(n_qubits, psi, parameter,
                               True, 'zz', pbc=True)
    elif type_gate == 1:
        psi = layer_paulipauli(n_qubits, psi, parameter,
                               True, 'yy', pbc=True)
        psi = layer_paulipauli(n_qubits, psi, parameter,
                               True, 'xx', pbc=True)
    elif type_gate == 2:
        psi = layer_paulipauli(n_qubits, psi, parameter,
                               False, 'zz', pbc=True)
    elif type_gate == 3:
        psi = layer_paulipauli(n_qubits, psi, parameter,
                               False, 'yy', pbc=True)
        psi = layer_paulipauli(n_qubits, psi, parameter,
                               False, 'xx', pbc=True)
    else:
        raise ValueError("type_gate is not a number from 0-3 inclusive")

    return psi


def layer_paulixpauli_analytical(n_qubits, psi_list, parameters,
                                 type_gates, current_param, gradient_index):
    psi, del_psi = psi_list

    psi_list[0] = rotation_layer_HVA(
        n_qubits, psi, parameters[current_param], type_gates)

    if gradient_index == current_param:
        psi_list[1] = rotation_layer_HVA_gradient(
            n_qubits, del_psi, parameters[current_param], type_gates)

    else:
        psi_list[1] = rotation_layer_HVA(
            n_qubits, del_psi, parameters[current_param], type_gates)


def layer_paulipauli_derivative_zz(n_qubits, psi, theta, odd, pbc=True):

    M = torch.mul(-1j*theta/2, zz)
    U = torch.matrix_exp(M).reshape(2, 2, 2, 2)
    d_U = torch.mul(torch.mul(-1j/2, zz),
                    torch.matrix_exp(M)).reshape(2, 2, 2, 2)

    psi_result = torch.zeros(psi.shape, device=device)

    for del_q in range(odd, n_qubits-1, 2):

        individual_product_psi = psi.clone()

        # product rule over the individual paulixpauli gates
        for q in range(odd, n_qubits-1, 2):

            if del_q == q:
                individual_product_psi = ApplyGate(
                    d_U, [q, q+1], individual_product_psi)
            else:
                individual_product_psi = ApplyGate(
                    U, [q, q+1], individual_product_psi)

        if pbc and n_qubits % 2 != odd:
            individual_product_psi = ApplyGate(
                U, [n_qubits-1, 0], individual_product_psi)

        psi_result = psi_result + individual_product_psi

    # add the product rule term over the last gate, if it exists
    if pbc and n_qubits % 2 != odd:

        individual_product_psi = psi.clone()

        for q in range(odd, n_qubits-1, 2):
            individual_product_psi = ApplyGate(
                U, [q, q+1], individual_product_psi)

        individual_product_psi = ApplyGate(
            d_U, [n_qubits-1, 0], individual_product_psi)

        psi_result = psi_result + individual_product_psi

    return psi_result


def layer_paulipauli_derivative_yyxx(n_qubits, psi, theta, odd, pbc=False):

    UYY = rotrot(theta, 2, False)
    d_UYY = rotrot(theta, 2, True)

#     print("UYY", UYY)
#     print(d_UYY)

    UXX = rotrot(theta, 1, False)
    d_UXX = rotrot(theta, 1, True)

    psi_result = torch.zeros(psi.shape, device=device)

    # index over all the gates with the same parameter
    for derivative_UXX_gate in range(2):
        for derivative_index in range(odd, (n_qubits-1), 2):
            individual_product_psi = psi.clone()

            # apply first all the UYY gates over the middle qubits
            for q in range(odd, n_qubits-1, 2):

                # apply the special product rule UYY gate if applicable
                if derivative_index == q and derivative_UXX_gate == 0:
                    individual_product_psi = ApplyGate(
                        d_UYY, [q, q+1], individual_product_psi)
                else:
                    individual_product_psi = ApplyGate(
                        UYY, [q, q+1], individual_product_psi)

            if pbc and n_qubits % 2 != odd:
                individual_product_psi = ApplyGate(
                    UYY, [n_qubits-1, 0], individual_product_psi)

            # now apply all the UXX gates over the middle qubits
            for q in range(odd, n_qubits-1, 2):

                # apply the special product rule UXX gate if applicable
                if derivative_index == q and derivative_UXX_gate == 1:
                    individual_product_psi = ApplyGate(
                        d_UXX, [q, q+1], individual_product_psi)

                else:
                    individual_product_psi = ApplyGate(
                        UXX, [q, q+1], individual_product_psi)

            if pbc and n_qubits % 2 != odd:
                individual_product_psi = ApplyGate(
                    UXX, [n_qubits-1, 0], individual_product_psi)

            psi_result = psi_result + individual_product_psi

    # add the product rule term over the last gate, if it exists
    if pbc and n_qubits % 2 != odd:

        # first, find the product rule term for where the derivative of the end
        # UYY gate is taken
        individual_product_psi = psi.clone()
        # apply the regular UYY gates
        for q in range(odd, n_qubits-1, 2):

            individual_product_psi = ApplyGate(
                UYY, [q, q+1], individual_product_psi)

        # apply the product rule UYY gate
        individual_product_psi = ApplyGate(
            d_UYY, [n_qubits-1, 0], individual_product_psi)

        # apply the regular UXX gates
        for q in range(odd, n_qubits-1, 2):

            individual_product_psi = ApplyGate(
                UXX, [q, q+1], individual_product_psi)

        # apply the regular UXX end qubit gate
        individual_product_psi = ApplyGate(
            UXX, [n_qubits-1, 0], individual_product_psi)

        psi_result = psi_result + individual_product_psi

        # now do the same thing except for the derivative of the UXX gate
        individual_product_psi = psi.clone()
        # apply the regular UYY gates
        for q in range(odd, n_qubits-1, 2):

            individual_product_psi = ApplyGate(
                UYY, [q, q+1], individual_product_psi)

        # apply the regular UYY gate
        individual_product_psi = ApplyGate(
            UYY, [n_qubits-1, 0], individual_product_psi)

        # apply the regular UXX gates
        for q in range(odd, n_qubits-1, 2):

            individual_product_psi = ApplyGate(
                UXX, [q, q+1], individual_product_psi)

        # apply the derivative UXX end qubit gate
        individual_product_psi = ApplyGate(
            d_UXX, [n_qubits-1, 0], individual_product_psi)

        psi_result = psi_result + individual_product_psi

    return psi_result


def layer_paulixpaul_it_numeric(n_qubits, psi_list, parameters,
                                rotation, current_param, gradient_index, dtheta):
    raise NotImplementedError


def layer_x_rot_HVA(n_qubits, psi_list, param):
    for q in range(n_qubits):
        for i in range(len(psi_list)):
            psi_list[i] = ApplyGate(rot(param, 2), [q], psi_list[i])


def layer_paulipauli(n_qubits, psi, theta, odd, pauli, pbc=False):
    if pauli == 'xx':
        M = torch.mul(-1j*theta/2, xx)
    elif pauli == 'yy':
        M = torch.mul(-1j*theta/2, yy)
    elif pauli == 'zz':
        M = torch.mul(-1j*theta/2, zz)
    else:
        raise ValueError("double pauli gate pauli is invalid")

    U = torch.matrix_exp(M).reshape(2, 2, 2, 2)

    for q in range(n_qubits-1):
        if q % 2 == odd:
            psi = ApplyGate(U, [q, q+1], psi)
    if pbc and n_qubits % 2 != odd:
        psi = ApplyGate(U, [n_qubits-1, 0], psi)

    return psi

# Generating Statistics From Individual Shots------------------------------------
from dask.distributed import Client, as_completed
from tqdm.auto import tqdm
import dask


def generate_sample_group_dask_tqdm(n_qubits, n_layers, gradient_index, n_samples,
                                    measurement_prob, ansatz, periodic, ham_type):
    """
    Generate a sample group of n_samples shots in parallel using Dask with a tqdm progress bar.
    Uses Dask delayed for deferred execution of tasks.
    """
    results = []

    num_param = num_parameters(n_qubits, n_layers, ansatz)

    # Use Dask delayed for deferred execution
    delayed_tasks = []
    for _ in range(n_samples):
        if ansatz == "GG":
            rotations = random_rotations(num_param)
        else:
            rotations = None

        parameters = random_parameters(num_param)
        measurements = random_measurements_prob(n_layers, n_qubits, measurement_prob)

        # Create a delayed task
        task = dask.delayed(gradients_by_layer)(n_qubits, n_layers,
                                                parameters, rotations=rotations,
                                                gradient_technique="analytic",
                                                gradient_index=gradient_index, measurements=measurements,
                                                return_analytic_suite=False, post_selected=False,
                                                entropy_regions=[[]], periodic=periodic,
                                                get_layered_results=True, ham_type=ham_type,
                                                ansatz=ansatz)
        delayed_tasks.append(task)

    # Compute all tasks in parallel and track progress with tqdm
    for result in dask.compute(*delayed_tasks):
        results.append(result)

    return results


def generate_sample_group(n_qubits, n_layers, gradient_index, n_samples,
                          measurement_prob, ansatz, periodic, ham_type):
    """
    Generate a sample group of n_sample shots. Calculates them using the
    analytic method as it is computationally the cheapest. Also just uses the
    z0z1 hamiltonian.
    """

    results = []

    num_param = num_parameters(n_qubits, n_layers, ansatz)

    for _ in tqdm(range(n_samples), leave=False):

        if ansatz == "GG":
            rotations = random_rotations(num_param)
        else:
            rotations = None

        parameters = random_parameters(num_param)
        measurements = random_measurements_prob(n_layers, n_qubits,
                                                measurement_prob)

        results.append(gradients_by_layer(n_qubits, n_layers,
                                          parameters,
                                          rotations=rotations,
                                          gradient_technique="analytic",
                       gradient_index=gradient_index, measurements=measurements,
                       return_analytic_suite=False, post_selected=False,
                       entropy_regions=[[]], periodic=periodic,
                       get_layered_results=True, ham_type=ham_type,
                       ansatz=ansatz))

    return results

def generate_entropy_sample_group(n_qubits, n_layers, n_samples,
                          measurement_prob, ansatz, periodic, ham_type):
    """
    Generate a sample group of n_sample shots. Calculates them using the
    analytic method as it is computationally the cheapest. Also just uses the
    z0z1 hamiltonian.
    """

    results = []

    num_param = num_parameters(n_qubits, n_layers, ansatz)

    for _ in tqdm(range(n_samples), leave=False):

        if ansatz == "GG":
            rotations = random_rotations(num_param)
        else:
            rotations = None

        parameters = random_parameters(num_param)
        measurements = random_measurements_prob(n_layers, n_qubits,
                                                measurement_prob)

        results.append(gradients_by_layer(n_qubits, n_layers,
                                          parameters,
                                          rotations=rotations,
                                          gradient_technique="analytic",
                       gradient_index=0, measurements=measurements,
                       return_analytic_suite=False, post_selected=False,
                       entropy_regions=True, periodic=periodic,
                       get_layered_results=True, ham_type=ham_type,
                       ansatz=ansatz))

    return results

def bootstrapped_variance_intervals(n_qubits, n_layers, gradient_index, n_samples,
                                    measurement_prob, significance_level,
                                    ansatz, periodic, file_name, ham_type,
                                    sample_group_function):
    """
    Function to calculate bootstrapped variance intervals using either serial or parallel sample generation.
    """

    rng = np.random.default_rng()

    # Generate samples using the provided sample group function (serial or parallel)
    samples = np.asarray(sample_group_function(n_qubits, n_layers, gradient_index, n_samples,
                                               measurement_prob, ansatz, periodic, ham_type))

    np.save(file_name + "samples", samples)

    # Rest of the function remains the same
    unaware_samples = samples[:, :, 0]
    aware_samples = samples[:, :, 1]

    unaware_bootstrap = bootstrap(
        (unaware_samples,), np.var, confidence_level=significance_level, random_state=rng)
    aware_bootstrap = bootstrap(
        (aware_samples,), np.var, confidence_level=significance_level, random_state=rng)

    var = np.var(samples, axis=0)
    reshaped_var = np.vstack((var[:, 0], var[:, 1]))

    return np.asarray([reshaped_var, unaware_bootstrap.confidence_interval, aware_bootstrap.confidence_interval])

def entropy_means(n_qubits, n_layers, n_samples,
                                    measurement_prob, 
                                    ansatz, periodic, file_name, ham_type):
    """
    Function to calculate bootstrapped variance intervals using either serial or parallel sample generation.
    """

    rng = np.random.default_rng()

    # Generate samples using the provided sample group function (serial or parallel)
    samples = np.asarray(generate_entropy_sample_group(n_qubits, n_layers, n_samples,
                                               measurement_prob, ansatz, periodic, ham_type))

    np.save(file_name + "samples", samples)
    
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0) / np.sqrt(len(samples))

    return np.asarray([mean, std])


def generate_results(qubit_range, n_layers, n_samples,
                     probability_range, ansatz, file_name, ham_type="xxz_1_1_05", parallel=False):
    """
    Generate results either in parallel using Dask or serially based on the parallel flag.
    """

    full_file_name = file_name + '.npy'
    if os.path.exists(full_file_name):
        user_input = input(f"The file {full_file_name} already exists. Overwrite? (Y/N): ").strip().upper()
        if user_input != 'Y':
            print("Operation cancelled by the user.")
            return
        else:
            print(f"Overwriting the file {full_file_name}...")

    results = np.zeros((
        1, len(probability_range), len(qubit_range), 3, 2, n_layers))

    client = Client() if parallel else None

    # Choose the function based on the parallel flag
    sample_group_function = generate_sample_group_dask_tqdm if parallel else generate_sample_group

    for n_q, q in enumerate(tqdm(qubit_range, leave=True)):
        num_p = num_parameters(q, n_layers, "HEA2")
        gradient_range = [0]

        for n_p, p in enumerate(tqdm(probability_range, leave=False)):
            for n_g, g in enumerate(tqdm(gradient_range, leave=False)):

                bootstrap_result = bootstrapped_variance_intervals(q, n_layers, g,
                                                                   n_samples,
                                                                   p,
                                                                   .67,
                                                                   ansatz, True, file_name, ham_type,
                                                                   sample_group_function)

                results[n_g, n_p, n_q, :] = bootstrap_result

                np.save(file_name, results)

    if client:
        client.close()

    return results

def generate_entropy_results(qubit_range, n_layers, n_samples,
                     probability_range, ansatz, file_name, ham_type="xxz_1_1_05", parallel=False):
    """
    Generate results either in parallel using Dask or serially based on the parallel flag.
    """

    full_file_name = file_name + '.npy'
    if os.path.exists(full_file_name):
        user_input = input(f"The file {full_file_name} already exists. Overwrite? (Y/N): ").strip().upper()
        if user_input != 'Y':
            print("Operation cancelled by the user.")
            return
        else:
            print(f"Overwriting the file {full_file_name}...")

    for n_q, q in enumerate(tqdm(qubit_range, leave=True)):
        gradient_range = [0]

        for n_p, p in enumerate(tqdm(probability_range, leave=False)):

                result = entropy_means(q, n_layers, n_samples, p, ansatz, True, file_name, ham_type)
                print(result.shape)
                return

                results[n_g, n_p, n_q, :] = bootstrap_result

                np.save(file_name, results)

    return results

def prepare_random_unitary_expectation():
    return NotImplementedError

# Mutual entropy code ----------------------------------

def probability_to_measure_one_given_parameters(n_qubits, n_layers, parameters, measurements, specific_outcome= None, final_outcome_in = None):

        z0z1_func, outcomes,pM = gradients_by_layer(n_qubits, n_layers,
                                          parameters,
                                          rotations=None,
                                          gradient_technique="analytic",
                       gradient_index=0, measurements=measurements,
                       return_analytic_suite=True, post_selected=False,
                       entropy_regions=False, periodic=True,
                       get_layered_results=True, ham_type="z0z1",
                       ansatz="HEA2", specific_measurement=specific_outcome) # TODO get rid of everything related to gradient in this calculation
        z0z1_func = np.asarray(z0z1_func)
        prob_1 = (1+z0z1_func)/2
        prob_minus_1 = (1-z0z1_func)/2

        #print(prob_1, prob_minus_1)

        prob_1 = np.where(prob_1 < 0, 0, prob_1)
       
        prob_minus_1 = np.where(prob_minus_1<0,0,prob_minus_1)

        results = []
        final_outcomes = []

        for i in range(n_layers):
            #print([prob_1[i],prob_minus_1[i]])
            if final_outcome_in == None:
                final_outcome = np.random.choice([0,1],p=[prob_1[i],1-prob_1[i]])
            else:
                final_outcome = final_outcome_in[i]
            final_prob = [prob_1[i],1-prob_1[i]][final_outcome]*pM[i]
            final_outcomes.append(final_outcome)
            results.append(final_prob)

       # results are in shape 
        return results, outcomes, final_outcomes

@dask.delayed
def probability_to_measure_one_given_parameters_delayed(n_qubits, n_layers, parameters, measurements, specific_outcome=None, final_outcome=None):
    return probability_to_measure_one_given_parameters(n_qubits, n_layers, parameters, measurements, specific_outcome,final_outcome)


def generate_mutual_info_samples_dask_change_all_parameters(n_qubits, n_layers,  n_a, n_c, measurements): 

    rng = np.random.default_rng()
    #generate list of random theta_a
    samples = [] 
    parameters_original = random_parameters(num_parameters(n_qubits, n_layers, "HEA2")) # appending a tuple (n_c,2, n_layers) to samples 
    original_sample, outcome, final_outcome = probability_to_measure_one_given_parameters(n_qubits, n_layers, parameters_original, measurements,specific_outcome=None,final_outcome_in=None)  
    print(outcome,final_outcome)
    for _ in range(n_a):
        parameters = random_parameters(num_parameters(n_qubits, n_layers, "HEA2")) # appending a tuple (n_c,2, n_layers) to samples 
        samples.append(probability_to_measure_one_given_parameters_delayed(n_qubits, n_layers, parameters, measurements, outcome, final_outcome))

    results = dask.compute(*samples)

    results_final = []

    for sample, _, _ in results:
        results_final.append(sample)

    results_final.append(original_sample)

    results_final = np.reshape(results_final, (n_a+1, n_layers))

    # overall shape (n_a, n_layers) 
    return results_final

def mutual_information_change_all_parameters(n_qubits,n_layers,measurements, n_a,n_c):

    #uses dask 
    samples = generate_mutual_info_samples_dask_change_all_parameters(n_qubits, n_layers, n_a, n_c, measurements)

    print(samples.shape)

    p_i_m_given_thetas = samples[:]

    print("p_i_m given thetas", p_i_m_given_thetas)

    p_bi_aware = np.mean(samples[:,:], axis=(0))

    print("p_bi_aware", p_bi_aware)
    print("p_bi_aware_variance", np.std(samples[:,:], axis=(0)))

    # print("numerator", p_i_m_given_thetas)

    # # print("denominator", p_bi_aware)

    # p_bi_unaware = np.mean(p_i_m_given_thetas, axis=(0,1))

    # p_i_given_thetas = np.mean(p_i_m_given_thetas, axis=(1))

    # sum over every axis except for the number of layers, for both aware + unawareuuu

    mutual_info_aware = - np.sum(np.log(p_bi_aware/p_i_m_given_thetas),axis=0)
    #mutual_info_unaware = - np.sum(np.log(p_bi_unaware/p_i_given_thetas),axis=(0))

    return mutual_info_aware #mutual_info_unaware

# END OF MUTUAL INFORMATION CALCUALTIONS ---------------------------------------------------------------------------------------------------------------

def dividing_measurement_gates(n_qubits, n_layers, n_sub_circuits):
    step = n_layers // n_sub_circuits
    points = [[step * i-1, n_qubits // 2 - 1]
              for i in range(1, n_sub_circuits)]
    return points


def gradient_descent_optimize_with_schedule_and_cost(ansatz, n_qubits, n_layers, theta, learning_rate, Niter, measurements, n_shots, post_selected, dir_name, parallel, ham_type):

    # Create directory if it doesn't exist
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    log_file_path = os.path.join(dir_name, 'params.log')
    with open(log_file_path, 'w') as log_file:
        log_file.write(f"ansatz: {ansatz}, n_qubits: {n_qubits}, n_layers: {n_layers}, theta: {theta}, learning_rate: {learning_rate}, Niter: {Niter}, measurements: {measurements}, n_shots: {n_shots}, post_selected: {post_selected}, dir_name: {dir_name}, parallel: {parallel}, ham_type: {ham_type}\n")

    if n_shots < 1:
        raise ValueError("Number of shots should be greater than 1")

    cost_values_unaware = []
    cost_values_aware = []

    theta_unaware = theta.copy()
    theta_aware = theta.copy()

    for i in tqdm(range(Niter), desc='iteration', leave=False):

        c_unaware, unaware_gradients, _ = cost_and_grad(ansatz, n_qubits,
                                                        n_layers, theta_unaware,
                                                        measurements, n_shots, os.path.join(dir_name, f"{i}_unaware"), post_selected, parallel, ham_type)

        c_aware, _,  aware_gradients = cost_and_grad(ansatz, n_qubits,
                                                     n_layers, theta_unaware,
                                                     measurements, n_shots, os.path.join(dir_name, f"{i}_aware"), post_selected, parallel, ham_type)

        cost_values_unaware.append(c_unaware)
        cost_values_aware.append(c_aware)

        theta_unaware = theta_unaware - learning_rate * unaware_gradients
        theta_aware = theta_aware - learning_rate * aware_gradients

    return theta_unaware, theta_aware, cost_values_unaware, cost_values_aware


def gradient_descent_multiple_optimization_runs(ansatz, n_qubits, n_layers, initial_theta, learning_rate, Niter, measurements, n_shots, post_selected, dir_name, parallel, ham_type, n_theta=5):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    else:
        raise FileExistsError(f"Directory {dir_name} already exists.")

    all_aware_costs = []
    all_unaware_costs = []

    # Plot for Aware Costs
    plt.figure(figsize=(12, 8))

    for run in tqdm(range(n_theta), desc="run"):
        # Assuming you have these functions defined
        theta = random_parameters(num_parameters(n_qubits, n_layers, ansatz))
        run_dir = os.path.join(dir_name, f"run_{run}")

        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        _, _, cost_values_unaware, cost_values_aware = optimize_with_schedule_and_cost(
            ansatz, n_qubits, n_layers, theta, learning_rate, Niter, measurements, n_shots, post_selected, run_dir, parallel, ham_type)

        final_value_aware = cost_values_aware[-1]
        plt.plot(cost_values_aware,
                 label=f'Run {run + 1} (Final Value: {final_value_aware:.4f})')
        all_aware_costs.append(cost_values_aware)

    plt.xlabel('Iterations')
    plt.ylabel('Aware Cost Values')
    plt.title('Aware Cost Values for Different Runs')
    plt.legend()
    plt.show()

    # Plot for Unaware Costs
    plt.figure(figsize=(12, 8))

    for run, cost_values_unaware in enumerate(all_unaware_costs):
        final_value_unaware = cost_values_unaware[-1]
        plt.plot(cost_values_unaware,
                 label=f'Run {run + 1} (Final Value: {final_value_unaware:.4f})')

    plt.xlabel('Iterations')
    plt.ylabel('Unaware Cost Values')
    plt.title('Unaware Cost Values for Different Runs')
    plt.legend()
    plt.show()

    return np.array(all_unaware_costs), np.array(all_aware_costs)


def parallel_cost_and_grad(ansatz, n_qubits, n_layers, theta, measurements, n_shots, post_selected=False, ham_type="z0z1"):
    tasks = []

    # Creating a list of delayed tasks
    for shot in range(n_shots):
        for d_i in range(len(theta)):
            task = delayed_gradients_by_layer(n_qubits, n_layers, theta, gradient_technique="analytic",
                                              gradient_index=d_i, measurements=measurements,
                                              return_analytic_suite=True, periodic=True,
                                              get_layered_results=False, ham_type=ham_type,
                                              ansatz=ansatz, rotations=None, post_selected=post_selected)
            tasks.append(task)

    # Using Dask's compute to execute the tasks in parallel
    with ProgressBar():
        results = compute(*tasks, scheduler='processes')

#     np.save(file_name, results)

    # Reshape results
    cost_functions = np.array(
        [result[0] for result in results]).reshape(n_shots, len(theta))
    unaware_gradients = np.array(
        [result[1] for result in results]).reshape(n_shots, len(theta))
    aware_gradients = np.array(
        [result[2] for result in results]).reshape(n_shots, len(theta))

    # Compute the mean over shots for each parameter
    mean_cost_functions = cost_functions.mean(axis=0)
    mean_unaware_gradients = unaware_gradients.mean(axis=0)
    mean_aware_gradients = aware_gradients.mean(axis=0)

    return mean_cost_functions[0], mean_unaware_gradients, mean_aware_gradients


def non_parallel_cost_and_grad(ansatz, n_qubits, n_layers, theta, measurements, n_shots, post_selected=False, ham_type="z0z1"):

    tasks = []

    #print("length of theta", len(theta))

    # Creating a list of delayed tasks
    for shot in range(n_shots):
        for d_i in range(len(theta)):
            task = gradients_by_layer(n_qubits, n_layers, theta, gradient_technique="analytic",
                                      gradient_index=d_i, measurements=measurements,
                                      return_analytic_suite=True, periodic=True,
                                      get_layered_results=False, ham_type=ham_type,
                                      ansatz=ansatz, rotations=None, post_selected=post_selected)
            tasks.append(task)

    results = tasks

    #print("results shape", np.asarray(results).shape)

#     np.save(file_name, results)

    # Reshape results
    cost_functions = np.array(
        [result[0] for result in results]).reshape(n_shots, len(theta))
    unaware_gradients = np.array(
        [result[1] for result in results]).reshape(n_shots, len(theta))
    aware_gradients = np.array(
        [result[2] for result in results]).reshape(n_shots, len(theta))

    # Compute the mean over shots for each parameter
    mean_cost_functions = cost_functions.mean(axis=0)
    mean_unaware_gradients = unaware_gradients.mean(axis=0)
    mean_aware_gradients = aware_gradients.mean(axis=0)

    return mean_cost_functions[0], mean_unaware_gradients, mean_aware_gradients


def schedule(Niter, schedule):

    if schedule == 'linear':
        return np.linspace(1, 0, Niter)
    elif schedule == 'linear2':
        return np.linspace(.2, 0, Niter)
    elif schedule == "none":
        return np.zeros(Niter)
    else:
        raise NotImplementedError


# MISC functions----------------------------------------------------------------


def renyi_entropy(psi, n_qubits, qubits_to_keep):
    '''
    returns the renyi entropy given psi.
    '''

    basis_spin = quspin.basis.spin_basis_general(n_qubits, S='1/2')
    psi = np.reshape(psi, (2**(len(psi.shape))))
    rdm = basis_spin.partial_trace(psi, qubits_to_keep)
    return -np.log2(np.trace(rdm**2).real)


def generate_entropies(psi, n_qubits, entropy_regions):

    entropy_results = []

    for qubit_region in entropy_regions:
        entropy_results.append(renyi_entropy(psi, n_qubits, qubit_region))

    return entropy_results


# Wrapping the gradients_by_layer function with Dask's delayed for lazy evaluation
@ delayed
def delayed_gradients_by_layer(n_qubits, n_layers, parameters, gradient_technique="numeric",
                               gradient_index=0, measurements=None, dtheta=.00001,
                               return_analytic_suite=False, post_selected=False,
                               entropy_regions=[[]], periodic=False,
                               get_layered_results=False, ham_type="z0z1",
                               ansatz="GG", rotations=None,):
    return gradients_by_layer(n_qubits, n_layers, parameters, gradient_technique, gradient_index, measurements,
                              return_analytic_suite=return_analytic_suite, periodic=periodic, get_layered_results=get_layered_results,
                              ham_type=ham_type, ansatz=ansatz, rotations=rotations)


def cost_and_grad(ansatz, n_qubits, n_layers, theta, measurements, n_shots, post_selected=False, parallel=False, ham_type="z0z1"):

    if parallel:
        return parallel_cost_and_grad(ansatz, n_qubits, n_layers, theta, measurements, n_shots, post_selected, ham_type)
    else:
        return non_parallel_cost_and_grad(ansatz, n_qubits, n_layers, theta, measurements, n_shots, post_selected, ham_type)


def gather_run_data(root_directory):
    """
    Gathers all run data stored in .npy files in a given root directory and its subdirectories.

    Parameters:
        root_directory (str): The directory where the run data folders are located.

    Returns:
        dict: A dictionary containing the data from each run. The keys are the run folder names and the values are the loaded numpy arrays.
    """
    all_run_data = {}

    # Loop over each folder in the root directory
    for folder_name in os.listdir(root_directory):
        folder_path = os.path.join(root_directory, folder_name)

        # If it's a directory, we assume it's a run folder
        if os.path.isdir(folder_path):

            # Initialize a dictionary for this specific run
            run_data = {}

            # Loop over each file in the run folder
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)

                # Only read .npy files
                if file_path.endswith(".npy"):
                    # Load the data and store it in the dictionary
                    data = np.load(file_path)
                    run_data[file_name] = data

            # Add this run's data to the overall data dictionary
            all_run_data[folder_name] = run_data

    return all_run_data


def scipy_cost_and_grad(theta, ansatz, n_qubits, n_layers, measurements, n_shots, post_selected, ham_type, parallel, gradient):
    if parallel:
        cost, unaware_gradients, aware_gradients = parallel_cost_and_grad(
            ansatz, n_qubits, n_layers, theta, measurements, n_shots, post_selected, ham_type)
    else:
        cost, unaware_gradients, aware_gradients = non_parallel_cost_and_grad(
            ansatz, n_qubits, n_layers, theta, measurements, n_shots, post_selected, ham_type)

    if gradient == "aware":
        return np.array(cost, dtype=np.float64), np.array(aware_gradients, dtype=np.float64)
    elif gradient == "unaware":
        return np.array(cost, dtype=np.float64), np.array(unaware_gradients, dtype=np.float64)
    else:
        raise ValueError(
            "neither aware nor unaware gradient input into scipy_cost_and_grad")


def optimize_with_scipy(ansatz, n_qubits, n_layers, initial_theta, measurements, n_shots, post_selected, dir_name, parallel, ham_type, gradient):
    global costs_at_each_iteration
    costs_at_each_iteration = []
    iteration_ticker = 0

    def callback(x):
        nonlocal iteration_ticker
        cost, _ = scipy_cost_and_grad(x, ansatz, n_qubits, n_layers,
                                      measurements, n_shots, post_selected, ham_type, parallel, gradient)
        costs_at_each_iteration.append(cost)
        iteration_ticker = iteration_ticker + 1
        #print(iteration_ticker)

    result = minimize(
        fun=scipy_cost_and_grad,
        x0=initial_theta,
        args=(ansatz, n_qubits, n_layers, measurements, n_shots,
              post_selected, ham_type, parallel, gradient),
        jac=True,
        method='L-BFGS-B',
        callback=callback,
        options={'maxiter': 200}
    )

    return result.x, result.fun, costs_at_each_iteration


def multiple_optimization_runs(ansatz, n_qubits, n_layers, measurements, n_shots, post_selected, dir_name, parallel, ham_type, gradient, thetas):
    
    # Creating the directory name based on provided parameters and global variable
    full_dir_path = f"{dir_name}_{ansatz}_q{n_qubits}_l{n_layers}_shots{n_shots}_post{post_selected}_{ham_type}_{gradient}_thetas{len(thetas)}_version{code_version}"
    
    if not os.path.exists(full_dir_path):
        os.makedirs(full_dir_path)
    elif not os.listdir(full_dir_path):
        print(f"Note: Directory {full_dir_path} already exists and is empty. Proceeding to use it.")
    else:
        raise FileExistsError(f"Directory {full_dir_path} already exists and is not empty.")

    all_run_costs = []
    all_run_info = []

    plt.figure(figsize=(12, 8))

    for run, theta in enumerate(tqdm(thetas, desc="run", leave=False)):
        run_dir = os.path.join(full_dir_path, f"run_{run}")

        final_parameters, final_value, costs_at_each_iteration = optimize_with_scipy(
            ansatz, n_qubits, n_layers, theta, measurements, n_shots, post_selected, run_dir, parallel, ham_type, gradient)

        all_run_costs.append(costs_at_each_iteration)
        all_run_info.append(
            (final_parameters, final_value, costs_at_each_iteration))
        
        # Save the costs at each iteration for this run to a .npy file
        np.save(os.path.join(full_dir_path, f"run_{run}.npy"), costs_at_each_iteration)

        if costs_at_each_iteration:
            plt.plot(costs_at_each_iteration, label=f'Run {run + 1} (Final Value: {costs_at_each_iteration[-1]:.4f})')
        else:
            print(f"Run {run + 1} has an empty costs_at_each_iteration list!")

    plt.xlabel('Iterations')
    plt.ylabel('Cost Values')
    plt.title('Cost Values for Different Runs')
    plt.legend()
    plt.savefig(os.path.join(full_dir_path, "all_run_plot.png"))

    np.save(os.path.join(full_dir_path, "all_run_info.npy"), all_run_info)

    return all_run_info

# ansatz = "HEA2"
# n_qubits = 6
# n_layers = 10
# learning_rate = np.pi/200
# Niter = 100
# measurements = None
# initial_theta = np.array(random_parameters(num_parameters(n_qubits, n_layers, ansatz)), dtype=np.float64)
# n_shots = 1 #change this line
# post_selected = True #change this line
# dir_name = "xxz_1_1_4_6qubits_none_measurement_scipy"
# parallel=False
# ham_type = "xxz"
# n_theta = 10
# gradient = "aware"

# multiple_optimization_runs(ansatz, n_qubits, n_layers, initial_theta, measurements, n_shots, post_selected, dir_name, parallel, ham_type, gradient, n_theta)
import openfermion as of

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