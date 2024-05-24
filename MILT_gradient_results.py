from MILT_core import *
from dask.distributed import Client, as_completed
from tqdm.auto import tqdm
import dask


def generate_sample_group_dask_tqdm(
    n_qubits,
    n_layers,
    gradient_index,
    n_samples,
    measurement_prob,
    ansatz,
    periodic,
    ham_type,
):
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
        task = dask.delayed(gradients_by_layer)(
            n_qubits,
            n_layers,
            parameters,
            rotations=rotations,
            gradient_technique="analytic",
            gradient_index=gradient_index,
            measurements=measurements,
            return_analytic_suite=False,
            post_selected=False,
            entropy_regions=[[]],
            periodic=periodic,
            get_layered_results=True,
            ham_type=ham_type,
            ansatz=ansatz,
        )
        delayed_tasks.append(task)

    # Compute all tasks in parallel and track progress with tqdm
    for result in dask.compute(*delayed_tasks):
        results.append(result)

    return results


def generate_sample_group(
    n_qubits,
    n_layers,
    gradient_index,
    n_samples,
    measurement_prob,
    ansatz,
    periodic,
    ham_type,
):
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
        measurements = random_measurements_prob(n_layers, n_qubits, measurement_prob)

        results.append(
            gradients_by_layer(
                n_qubits,
                n_layers,
                parameters,
                rotations=rotations,
                gradient_technique="analytic",
                gradient_index=gradient_index,
                measurements=measurements,
                return_analytic_suite=False,
                post_selected=False,
                entropy_regions=[[]],
                periodic=periodic,
                get_layered_results=True,
                ham_type=ham_type,
                ansatz=ansatz,
            )
        )

    return results

def generate_sample_group_one_circuit(
    n_qubits,
    n_layers,
    gradient_index,
    n_samples,
    measurement_prob,
    ansatz,
    periodic,
    ham_type,
):
    """
    Generates samples, where the only thing changing is the measurement gate configuration. 
    """

    results = []

    num_param = num_parameters(n_qubits, n_layers, ansatz)

    parameters = random_parameters(num_param)

    for _ in tqdm(range(n_samples), leave=False):
        if ansatz == "GG":
            rotations = random_rotations(num_param)
        else:
            rotations = None

        measurements = random_measurements_prob(n_layers, n_qubits, measurement_prob)

        results.append(
            gradients_by_layer(
                n_qubits,
                n_layers,
                parameters,
                rotations=rotations,
                gradient_technique="analytic",
                gradient_index=gradient_index,
                measurements=measurements,
                return_analytic_suite=False,
                post_selected=False,
                entropy_regions=[[]],
                periodic=periodic,
                get_layered_results=True,
                ham_type=ham_type,
                ansatz=ansatz,
            )
        )

    return results


def generate_entropy_sample_group(
    n_qubits, n_layers, n_samples, measurement_prob, ansatz, periodic, ham_type
):
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
        measurements = random_measurements_prob(n_layers, n_qubits, measurement_prob)

        results.append(
            gradients_by_layer(
                n_qubits,
                n_layers,
                parameters,
                rotations=rotations,
                gradient_technique="analytic",
                gradient_index=0,
                measurements=measurements,
                return_analytic_suite=False,
                post_selected=False,
                entropy_regions=True,
                periodic=periodic,
                get_layered_results=True,
                ham_type=ham_type,
                ansatz=ansatz,
            )
        )

    return results


def bootstrapped_variance_intervals(
    n_qubits,
    n_layers,
    gradient_index,
    n_samples,
    measurement_prob,
    significance_level,
    ansatz,
    periodic,
    file_name,
    ham_type,
    sample_group_function,
):
    """
    Function to calculate bootstrapped variance intervals using either serial or parallel sample generation.
    """

    rng = np.random.default_rng()

    # Generate samples using the provided sample group function (serial or parallel)
    samples = np.asarray(
        sample_group_function(
            n_qubits,
            n_layers,
            gradient_index,
            n_samples,
            measurement_prob,
            ansatz,
            periodic,
            ham_type,
        )
    )

    np.save(file_name + "samples", samples)

    # Rest of the function remains the same
    unaware_samples = samples[:, :, 0]
    aware_samples = samples[:, :, 1]

    unaware_bootstrap = bootstrap(
        (unaware_samples,),
        np.var,
        confidence_level=significance_level,
        random_state=rng,
    )
    aware_bootstrap = bootstrap(
        (aware_samples,), np.var, confidence_level=significance_level, random_state=rng
    )

    var = np.var(samples, axis=0)
    reshaped_var = np.vstack((var[:, 0], var[:, 1]))

    return np.asarray(
        [
            reshaped_var,
            unaware_bootstrap.confidence_interval,
            aware_bootstrap.confidence_interval,
        ]
    )


def entropy_means(
    n_qubits,
    n_layers,
    n_samples,
    measurement_prob,
    ansatz,
    periodic,
    file_name,
    ham_type,
):
    """
    Function to calculate bootstrapped variance intervals using either serial or parallel sample generation.
    """

    rng = np.random.default_rng()

    # Generate samples using the provided sample group function (serial or parallel)
    samples = np.asarray(
        generate_entropy_sample_group(
            n_qubits, n_layers, n_samples, measurement_prob, ansatz, periodic, ham_type
        )
    )

    np.save(file_name + "samples", samples)

    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0) / np.sqrt(len(samples))

    return np.asarray([mean, std])

def generate_results_nonvarying(
    qubit_range,
    n_layers,
    n_samples,
    probability_range,
    ansatz,
    file_name,
    ham_type="xxz_1_1_05",
    parallel=False,
):
    """
    Generate results either in parallel using Dask or serially based on the parallel flag for our 2024-05-23 results, where we are testing to see what happens when we var ythe position of the measurement gates for all parameters except the first being the same.
    """

    if parallel is True:
        raise NotImplementedError 

    full_file_name = file_name + ".npy"
    if os.path.exists(full_file_name):
        user_input = (
            input(f"The file {full_file_name} already exists. Overwrite? (Y/N): ")
            .strip()
            .upper()
        )
        if user_input != "Y":
            print("Operation cancelled by the user.")
            return
        else:
            print(f"Overwriting the file {full_file_name}...")

    results = np.zeros((1, len(probability_range), len(qubit_range), 3, 2, n_layers))

    client = Client() if parallel else None

    # Choose the function based on the parallel flag
    sample_group_function = (
        generate_sample_group_dask_tqdm if parallel else generate_sample_group_one_circuit
    )

    for n_q, q in enumerate(tqdm(qubit_range, leave=True)):
        num_p = num_parameters(q, n_layers, "HEA2")
        gradient_range = [0]

        for n_p, p in enumerate(tqdm(probability_range, leave=False)):
            for n_g, g in enumerate(tqdm(gradient_range, leave=False)):
                bootstrap_result = bootstrapped_variance_intervals(
                    q,
                    n_layers,
                    g,
                    n_samples,
                    p,
                    0.67,
                    ansatz,
                    True,
                    file_name,
                    ham_type,
                    sample_group_function,
                )

                results[n_g, n_p, n_q, :] = bootstrap_result

                np.save(file_name, results)

    if client:
        client.close()

    return results

def generate_results(
    qubit_range,
    n_layers,
    n_samples,
    probability_range,
    ansatz,
    file_name,
    ham_type="xxz_1_1_05",
    parallel=False,
):
    """
    Generate results either in parallel using Dask or serially based on the parallel flag.
    """

    full_file_name = file_name + ".npy"
    if os.path.exists(full_file_name):
        user_input = (
            input(f"The file {full_file_name} already exists. Overwrite? (Y/N): ")
            .strip()
            .upper()
        )
        if user_input != "Y":
            print("Operation cancelled by the user.")
            return
        else:
            print(f"Overwriting the file {full_file_name}...")

    results = np.zeros((1, len(probability_range), len(qubit_range), 3, 2, n_layers))

    client = Client() if parallel else None

    # Choose the function based on the parallel flag
    sample_group_function = (
        generate_sample_group_dask_tqdm if parallel else generate_sample_group
    )

    for n_q, q in enumerate(tqdm(qubit_range, leave=True)):
        num_p = num_parameters(q, n_layers, "HEA2")
        gradient_range = [0]

        for n_p, p in enumerate(tqdm(probability_range, leave=False)):
            for n_g, g in enumerate(tqdm(gradient_range, leave=False)):
                bootstrap_result = bootstrapped_variance_intervals(
                    q,
                    n_layers,
                    g,
                    n_samples,
                    p,
                    0.67,
                    ansatz,
                    True,
                    file_name,
                    ham_type,
                    sample_group_function,
                )

                results[n_g, n_p, n_q, :] = bootstrap_result

                np.save(file_name, results)

    if client:
        client.close()

    return results


def generate_entropy_results(
    qubit_range,
    n_layers,
    n_samples,
    probability_range,
    ansatz,
    file_name,
    ham_type="xxz_1_1_05",
    parallel=False,
):
    """
    Generate results either in parallel using Dask or serially based on the parallel flag.
    """

    full_file_name = file_name + ".npy"
    if os.path.exists(full_file_name):
        user_input = (
            input(f"The file {full_file_name} already exists. Overwrite? (Y/N): ")
            .strip()
            .upper()
        )
        if user_input != "Y":
            print("Operation cancelled by the user.")
            return
        else:
            print(f"Overwriting the file {full_file_name}...")

    for n_q, q in enumerate(tqdm(qubit_range, leave=True)):
        gradient_range = [0]

        for n_p, p in enumerate(tqdm(probability_range, leave=False)):
            result = entropy_means(
                q, n_layers, n_samples, p, ansatz, True, file_name, ham_type
            )
            print(result.shape)
            return

            results[n_g, n_p, n_q, :] = bootstrap_result

            np.save(file_name, results)

    return results
