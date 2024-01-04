from MILT_core import *
import dask


def probability_to_measure_one_given_parameters(
    n_qubits, n_layers, parameters, measurements
):
    z0z1_func = np.array(
        gradients_by_layer(
            n_qubits,
            n_layers,
            parameters,
            rotations=None,
            gradient_technique=None,
            gradient_index=0,
            measurements=measurements,
            return_analytic_suite=True,
            post_selected=True,
            entropy_regions=False,
            periodic=True,
            get_layered_results=True,
            ham_type="z0z1",
            ansatz="HEA2",
        )
    )  # TODO get rid of everything related to gradient in this calculation

    prob_1 = (1 + z0z1_func) / 2
    prob_minus_1 = (1 - z0z1_func) / 2

    # print(prob_1, prob_minus_1)

    prob_1 = np.where(prob_1 < 0, 0, prob_1)

    prob_minus_1 = np.where(prob_minus_1 < 0, 0, prob_minus_1)

    # this is in shape (2, n_layers)
    return [prob_1, prob_minus_1]


@dask.delayed
def probability_to_measure_one_given_parameters_delayed(
    n_qubits, n_layers, parameters, measurements
):
    return probability_to_measure_one_given_parameters(
        n_qubits, n_layers, parameters, measurements
    )


def generate_mutual_info_sample_a(n_qubits, n_layers, parameters, measurements, n_c):
    # n_c is the number of measurements

    samples = []

    for _ in tqdm(range(n_c), leave=False, desc="Measurement samples"):
        samples.append(
            probability_to_measure_one_given_parameters(
                n_qubits, n_layers, parameters, measurements
            )
        )

    # shape (n_c, 2, n_layers)
    return samples


def generate_mutual_info_samples_dask(
    n_qubits, n_layers, parameters, measurements, n_a, n_c
):
    rng = np.random.default_rng()
    # generate list of random theta_a
    random_a = rng.uniform(low=-np.pi, high=np.pi, size=n_a)

    samples = []

    for theta_a in random_a:
        parameters_copy = np.copy(parameters)
        parameters_copy[0] = theta_a
        for _ in tqdm(range(n_c), leave=False, desc="Measurement samples"):
            # appending a tuple (n_c,2, n_layers) to samples
            samples.append(
                probability_to_measure_one_given_parameters_delayed(
                    n_qubits, n_layers, parameters, measurements
                )
            )

    results = dask.compute(*samples)

    results = np.reshape(results, (n_c, n_a, 2, n_layers))

    # overall shape (n_a, n_c, 2, n_layers)
    return results


def generate_mutual_info_samples(
    n_qubits, n_layers, parameters, measurements, n_a, n_c
):
    rng = np.random.default_rng()
    # generate list of random theta_a
    random_a = rng.uniform(low=-np.pi, high=np.pi, size=n_a)

    samples = []

    for theta_a in tqdm(random_a, leave=False, desc="theta_a"):
        parameters_copy = np.copy(parameters)
        parameters_copy[0] = theta_a
        # appending a tuple (n_c,2, n_layers) to samples
        samples.append(
            generate_mutual_info_sample_a(
                n_qubits, n_layers, parameters_copy, measurements, n_c
            )
        )

    # overall shape (n_a, n_c, 2, n_layers)
    return samples


def mutual_information_for_one_phi_vec(
    n_qubits, n_layers, parameters, measurements, n_a, n_c, dask=False
):
    # samples are differently drawn p(i,m|theta)
    if dask:
        # p_i_m_given_theta is of shaape (n_c, n_a, 2, n_layers)
        p_i_m_given_theta = generate_mutual_info_samples_dask(
            n_qubits, n_layers, parameters, measurements, n_a, n_c
        )
    else:
        # p_i_m_given_theta is of shaape (n_a, n_c, 2, n_layers)
        p_i_m_given_theta = generate_mutual_info_samples(
            n_qubits, n_layers, parameters, measurements, n_a, n_c
        )
    # print(p_i_m_given_theta)

    # print(np.asarray(p_i_m_given_theta).shape)

    # for unaware, working with the measurements averaged out
    # this is then of shape (n_a, 2, n_layers) and is also the shape of mutual_info_unaware
    p_i_given_theta = np.mean(p_i_m_given_theta, axis=1)

    p_bi_m = np.mean(p_i_m_given_theta, axis=0)
    p_bi = np.mean(p_i_m_given_theta, axis=(0, 1))

    # sum over every axis except for the number of layers, for both aware + unaware
    mutual_info_aware = -np.sum(
        p_i_m_given_theta * np.log(p_bi_m / p_i_m_given_theta), axis=(0, 1, 2)
    ) / (n_a * n_c)
    mutual_info_unaware = -np.sum(
        p_i_given_theta * np.log(p_bi / p_i_given_theta), axis=(0, 1)
    ) / (n_a * n_c)

    return mutual_info_aware, mutual_info_unaware


def overall_mutual_information(
    n_qubits, n_layers, measurements, n_p, n_a, n_c, dask=False
):
    if measurements == None or measurements == [[]]:
        n_c = 1  # don't accidentally run measurement outcome shots if there are no measurements

    # we are collecting a list over the same calculation but over different layers of the circuit
    mutual_info = []

    for _ in tqdm(range(n_p)):
        parameters = random_parameters(num_parameters(n_qubits, n_layers, "HEA2"))
        mutual_info.append(
            mutual_information_for_one_phi_vec(
                n_qubits, n_layers, parameters, measurements, n_a, n_c, dask
            )
        )

    return np.mean(mutual_info, axis=0)


def generate_mutual_info_samples_dask_change_all_parameters(
    n_qubits, n_layers, n_a, measurements
):
    rng = np.random.default_rng()
    # generate list of random theta_a
    samples = []
    for _ in range(n_a):
        parameters = random_parameters(num_parameters(n_qubits, n_layers, "HEA2"))
        # appending a tuple (n_c,2, n_layers) to samples
        samples.append(
            probability_to_measure_one_given_parameters_delayed(
                n_qubits, n_layers, parameters, measurements
            )
        )

    results = dask.compute(*samples)
    results = np.reshape(results, (n_a, 2, n_layers))

    # overall shape (n_a, 2, n_layers)
    return results


def generate_mutual_info_samples_dask_change_all_parameters_and_measurements(
    n_qubits, n_layers, n_a, n_p, p
):
    rng = np.random.default_rng()
    # generate list of random theta_a
    samples = []

    for _ in range(n_p):  # loop over different measurement configurations
        measurements = random_measurements_prob(n_layers, n_qubits, p)
        for _ in range(n_a):
            parameters = random_parameters(num_parameters(n_qubits, n_layers, "HEA2"))
            # appending a tuple (2, n_layers) to samples
            samples.append(
                probability_to_measure_one_given_parameters_delayed(
                    n_qubits, n_layers, parameters, measurements
                )
            )

    results = dask.compute(*samples)
    results = np.reshape(results, (n_p, n_a, 2, n_layers))

    # overall shape (n_p, n_a, 2, n_layers)
    return results


def mutual_information_only_parameters(n_qubits, n_layers, n_a, measurements):
    """
    Generate mutual information samples across all parameters but only across different thetas for one measurement configuration. Uses dask. n_a is the number of thetas averaged over.
    """

    p_i_m_given_thetas = generate_mutual_info_samples_dask_change_all_parameters(
        n_qubits, n_layers, n_a, measurements
    )

    p_bi = np.mean(p_i_m_given_thetas, axis=(0))

    # sum over every axis except for the number of layers, for both aware + unaware
    mutual_info_mean = -np.sum(
        p_i_m_given_thetas * np.log(p_bi / p_i_m_given_thetas), axis=(0, 1)
    ) / (n_a)

    mutual_info_error = np.std(
        -np.sum(p_i_m_given_thetas * np.log(p_bi / p_i_m_given_thetas), axis=(1)),
        axis=0,
    ) / np.sqrt(n_a)

    return mutual_info_mean, mutual_info_error


def mutual_info_different_measurements(n_qubits, n_layers, n_a, n_p, p):
    """
    For every layer, generate the mutual information, averaging over different thetas for different measurement configurations at probability p. n_a is the number of thetas averaged over and n_p is the number of measurements averaged over.
    """

    # generate samples in shape (n_p, n_a, 2, n_layers )
    p_i_m_given_thetas = (
        generate_mutual_info_samples_dask_change_all_parameters_and_measurements(
            n_qubits, n_layers, n_a, n_p, p
        )
    )

    # average over n_a
    p_bi = np.mean(p_i_m_given_thetas, axis=(1))

    # sum over n_a and pos/neg outcome
    mutual_info_measurement_groups = -np.sum(
        p_i_m_given_thetas * np.log(p_bi / p_i_m_given_thetas), axis=(1, 2)
    ) / (n_a)

    return np.mean(mutual_info_measurement_groups, axis=0), np.std(
        mutual_info_measurement_groups, axis=0
    ) / np.sqrt(n_p)
