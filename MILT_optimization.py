from MILT_core import *


def gradient_descent_optimize_with_schedule_and_cost(
    ansatz,
    n_qubits,
    n_layers,
    theta,
    learning_rate,
    Niter,
    measurements,
    n_shots,
    post_selected,
    dir_name,
    parallel,
    ham_type,
):
    # Create directory if it doesn't exist
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    log_file_path = os.path.join(dir_name, "params.log")
    with open(log_file_path, "w") as log_file:
        log_file.write(
            f"ansatz: {ansatz}, n_qubits: {n_qubits}, n_layers: {n_layers}, theta: {theta}, learning_rate: {learning_rate}, Niter: {Niter}, measurements: {measurements}, n_shots: {n_shots}, post_selected: {post_selected}, dir_name: {dir_name}, parallel: {parallel}, ham_type: {ham_type}\n"
        )

    if n_shots < 1:
        raise ValueError("Number of shots should be greater than 1")

    cost_values_unaware = []
    cost_values_aware = []

    theta_unaware = theta.copy()
    theta_aware = theta.copy()

    for i in tqdm(range(Niter), desc="iteration", leave=False):
        c_unaware, unaware_gradients, _ = cost_and_grad(
            ansatz,
            n_qubits,
            n_layers,
            theta_unaware,
            measurements,
            n_shots,
            os.path.join(dir_name, f"{i}_unaware"),
            post_selected,
            parallel,
            ham_type,
        )

        c_aware, _, aware_gradients = cost_and_grad(
            ansatz,
            n_qubits,
            n_layers,
            theta_unaware,
            measurements,
            n_shots,
            os.path.join(dir_name, f"{i}_aware"),
            post_selected,
            parallel,
            ham_type,
        )

        cost_values_unaware.append(c_unaware)
        cost_values_aware.append(c_aware)

        theta_unaware = theta_unaware - learning_rate * unaware_gradients
        theta_aware = theta_aware - learning_rate * aware_gradients

    return theta_unaware, theta_aware, cost_values_unaware, cost_values_aware


def gradient_descent_multiple_optimization_runs(
    ansatz,
    n_qubits,
    n_layers,
    initial_theta,
    learning_rate,
    Niter,
    measurements,
    n_shots,
    post_selected,
    dir_name,
    parallel,
    ham_type,
    n_theta=5,
):
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
            ansatz,
            n_qubits,
            n_layers,
            theta,
            learning_rate,
            Niter,
            measurements,
            n_shots,
            post_selected,
            run_dir,
            parallel,
            ham_type,
        )

        final_value_aware = cost_values_aware[-1]
        plt.plot(
            cost_values_aware,
            label=f"Run {run + 1} (Final Value: {final_value_aware:.4f})",
        )
        all_aware_costs.append(cost_values_aware)

    plt.xlabel("Iterations")
    plt.ylabel("Aware Cost Values")
    plt.title("Aware Cost Values for Different Runs")
    plt.legend()
    plt.show()

    # Plot for Unaware Costs
    plt.figure(figsize=(12, 8))

    for run, cost_values_unaware in enumerate(all_unaware_costs):
        final_value_unaware = cost_values_unaware[-1]
        plt.plot(
            cost_values_unaware,
            label=f"Run {run + 1} (Final Value: {final_value_unaware:.4f})",
        )

    plt.xlabel("Iterations")
    plt.ylabel("Unaware Cost Values")
    plt.title("Unaware Cost Values for Different Runs")
    plt.legend()
    plt.show()

    return np.array(all_unaware_costs), np.array(all_aware_costs)


def parallel_cost_and_grad(
    ansatz,
    n_qubits,
    n_layers,
    theta,
    measurements,
    n_shots,
    post_selected=False,
    ham_type="z0z1",
):
    tasks = []

    # Creating a list of delayed tasks
    for shot in range(n_shots):
        for d_i in range(len(theta)):
            task = delayed_gradients_by_layer(
                n_qubits,
                n_layers,
                theta,
                gradient_technique="analytic",
                gradient_index=d_i,
                measurements=measurements,
                return_analytic_suite=True,
                periodic=True,
                get_layered_results=False,
                ham_type=ham_type,
                ansatz=ansatz,
                rotations=None,
                post_selected=post_selected,
            )
            tasks.append(task)

    # Using Dask's compute to execute the tasks in parallel
    with ProgressBar():
        results = compute(*tasks, scheduler="processes")

    #     np.save(file_name, results)

    # Reshape results
    cost_functions = np.array([result[0] for result in results]).reshape(
        n_shots, len(theta)
    )
    unaware_gradients = np.array([result[1] for result in results]).reshape(
        n_shots, len(theta)
    )
    aware_gradients = np.array([result[2] for result in results]).reshape(
        n_shots, len(theta)
    )

    # Compute the mean over shots for each parameter
    mean_cost_functions = cost_functions.mean(axis=0)
    mean_unaware_gradients = unaware_gradients.mean(axis=0)
    mean_aware_gradients = aware_gradients.mean(axis=0)

    return mean_cost_functions[0], mean_unaware_gradients, mean_aware_gradients


def non_parallel_cost_and_grad(
    ansatz,
    n_qubits,
    n_layers,
    theta,
    measurements,
    n_shots,
    post_selected=False,
    ham_type="z0z1",
):
    tasks = []

    # print("length of theta", len(theta))

    # Creating a list of delayed tasks
    for shot in range(n_shots):
        for d_i in range(len(theta)):
            task = gradients_by_layer(
                n_qubits,
                n_layers,
                theta,
                gradient_technique="analytic",
                gradient_index=d_i,
                measurements=measurements,
                return_analytic_suite=True,
                periodic=True,
                get_layered_results=False,
                ham_type=ham_type,
                ansatz=ansatz,
                rotations=None,
                post_selected=post_selected,
            )
            tasks.append(task)

    results = tasks

    # print("results shape", np.asarray(results).shape)

    #     np.save(file_name, results)

    # Reshape results
    cost_functions = np.array([result[0] for result in results]).reshape(
        n_shots, len(theta)
    )
    unaware_gradients = np.array([result[1] for result in results]).reshape(
        n_shots, len(theta)
    )
    aware_gradients = np.array([result[2] for result in results]).reshape(
        n_shots, len(theta)
    )

    # Compute the mean over shots for each parameter
    mean_cost_functions = cost_functions.mean(axis=0)
    mean_unaware_gradients = unaware_gradients.mean(axis=0)
    mean_aware_gradients = aware_gradients.mean(axis=0)

    return mean_cost_functions[0], mean_unaware_gradients, mean_aware_gradients


def schedule(Niter, schedule):
    if schedule == "linear":
        return np.linspace(1, 0, Niter)
    elif schedule == "linear2":
        return np.linspace(0.2, 0, Niter)
    elif schedule == "none":
        return np.zeros(Niter)
    else:
        raise NotImplementedError


# Wrapping the gradients_by_layer function with Dask's delayed for lazy evaluation
@dask.delayed
def delayed_gradients_by_layer(
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
    return gradients_by_layer(
        n_qubits,
        n_layers,
        parameters,
        gradient_technique,
        gradient_index,
        measurements,
        return_analytic_suite=return_analytic_suite,
        periodic=periodic,
        get_layered_results=get_layered_results,
        ham_type=ham_type,
        ansatz=ansatz,
        rotations=rotations,
    )


def cost_and_grad(
    ansatz,
    n_qubits,
    n_layers,
    theta,
    measurements,
    n_shots,
    post_selected=False,
    parallel=False,
    ham_type="z0z1",
):
    if parallel:
        return parallel_cost_and_grad(
            ansatz,
            n_qubits,
            n_layers,
            theta,
            measurements,
            n_shots,
            post_selected,
            ham_type,
        )
    else:
        return non_parallel_cost_and_grad(
            ansatz,
            n_qubits,
            n_layers,
            theta,
            measurements,
            n_shots,
            post_selected,
            ham_type,
        )


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


def scipy_cost_and_grad(
    theta,
    ansatz,
    n_qubits,
    n_layers,
    measurements,
    n_shots,
    post_selected,
    ham_type,
    parallel,
    gradient,
):
    if parallel:
        cost, unaware_gradients, aware_gradients = parallel_cost_and_grad(
            ansatz,
            n_qubits,
            n_layers,
            theta,
            measurements,
            n_shots,
            post_selected,
            ham_type,
        )
    else:
        cost, unaware_gradients, aware_gradients = non_parallel_cost_and_grad(
            ansatz,
            n_qubits,
            n_layers,
            theta,
            measurements,
            n_shots,
            post_selected,
            ham_type,
        )

    if gradient == "aware":
        return np.array(cost, dtype=np.float64), np.array(
            aware_gradients, dtype=np.float64
        )
    elif gradient == "unaware":
        return np.array(cost, dtype=np.float64), np.array(
            unaware_gradients, dtype=np.float64
        )
    else:
        raise ValueError(
            "neither aware nor unaware gradient input into scipy_cost_and_grad"
        )


def optimize_with_scipy(
    ansatz,
    n_qubits,
    n_layers,
    initial_theta,
    measurements,
    n_shots,
    post_selected,
    dir_name,
    parallel,
    ham_type,
    gradient,
):
    global costs_at_each_iteration
    costs_at_each_iteration = []
    iteration_ticker = 0

    def callback(x):
        nonlocal iteration_ticker
        cost, _ = scipy_cost_and_grad(
            x,
            ansatz,
            n_qubits,
            n_layers,
            measurements,
            n_shots,
            post_selected,
            ham_type,
            parallel,
            gradient,
        )
        costs_at_each_iteration.append(cost)
        iteration_ticker = iteration_ticker + 1
        # print(iteration_ticker)

    result = minimize(
        fun=scipy_cost_and_grad,
        x0=initial_theta,
        args=(
            ansatz,
            n_qubits,
            n_layers,
            measurements,
            n_shots,
            post_selected,
            ham_type,
            parallel,
            gradient,
        ),
        jac=True,
        method="L-BFGS-B",
        callback=callback,
        options={"maxiter": 200},
    )

    return result.x, result.fun, costs_at_each_iteration


def multiple_optimization_runs(
    ansatz,
    n_qubits,
    n_layers,
    measurements,
    n_shots,
    post_selected,
    dir_name,
    parallel,
    ham_type,
    gradient,
    thetas,
):
    # Creating the directory name based on provided parameters and global variable
    full_dir_path = f"{dir_name}_{ansatz}_q{n_qubits}_l{n_layers}_shots{n_shots}_post{post_selected}_{ham_type}_{gradient}_thetas{len(thetas)}_version{code_version}"

    if not os.path.exists(full_dir_path):
        os.makedirs(full_dir_path)
    elif not os.listdir(full_dir_path):
        print(
            f"Note: Directory {full_dir_path} already exists and is empty. Proceeding to use it."
        )
    else:
        raise FileExistsError(
            f"Directory {full_dir_path} already exists and is not empty."
        )

    all_run_costs = []
    all_run_info = []

    plt.figure(figsize=(12, 8))

    for run, theta in enumerate(tqdm(thetas, desc="run", leave=False)):
        run_dir = os.path.join(full_dir_path, f"run_{run}")

        final_parameters, final_value, costs_at_each_iteration = optimize_with_scipy(
            ansatz,
            n_qubits,
            n_layers,
            theta,
            measurements,
            n_shots,
            post_selected,
            run_dir,
            parallel,
            ham_type,
            gradient,
        )

        all_run_costs.append(costs_at_each_iteration)
        all_run_info.append((final_parameters, final_value, costs_at_each_iteration))

        # Save the costs at each iteration for this run to a .npy file
        np.save(os.path.join(full_dir_path, f"run_{run}.npy"), costs_at_each_iteration)

        if costs_at_each_iteration:
            plt.plot(
                costs_at_each_iteration,
                label=f"Run {run + 1} (Final Value: {costs_at_each_iteration[-1]:.4f})",
            )
        else:
            print(f"Run {run + 1} has an empty costs_at_each_iteration list!")

    plt.xlabel("Iterations")
    plt.ylabel("Cost Values")
    plt.title("Cost Values for Different Runs")
    plt.legend()
    plt.savefig(os.path.join(full_dir_path, "all_run_plot.png"))

    np.save(os.path.join(full_dir_path, "all_run_info.npy"), all_run_info)

    return all_run_info
