from loguru import logger
from federated_learning.arguments import Arguments
from federated_learning.utils import generate_data_loaders_from_distributed_dataset
from federated_learning.datasets.data_distribution import distribute_batches_equally
from federated_learning.utils import average_nn_parameters
from federated_learning.utils import convert_distributed_data_into_numpy
from federated_learning.utils import poison_data
from federated_learning.utils import identify_random_elements
from federated_learning.utils import save_results
from federated_learning.utils import load_train_data_loader
from federated_learning.utils import load_test_data_loader
from federated_learning.utils import generate_experiment_ids
from federated_learning.utils import convert_results_to_csv
from federated_learning.utils.random_noise import apply_random_noise
from client import Client
from federated_learning.utils.fang_attack_trimmed_mean import apply_fang_trim_mean

import os

def train_subset_of_clients(epoch, args, clients, poisoned_workers):
    """
    Train a subset of clients per round.

    :param epoch: epoch
    :type epoch: int
    :param args: arguments
    :type args: Arguments
    :param clients: clients
    :type clients: list(Client)
    :param poisoned_workers: indices of poisoned workers
    :type poisoned_workers: list(int)
    """
    kwargs = args.get_round_worker_selection_strategy_kwargs()
    kwargs["current_epoch_number"] = epoch

    random_workers = args.get_round_worker_selection_strategy().select_round_workers(
        list(range(args.get_num_workers())),
        poisoned_workers,
        kwargs)

    for client_idx in random_workers:
        args.get_logger().info("Training epoch #{} on client #{}", str(epoch), str(clients[client_idx].get_client_index()))
        if client_idx in poisoned_workers and args.get_attack_type() == "sign_flipping":
            clients[client_idx].train(epoch,True)
        else:
            clients[client_idx].train(epoch,False)
    args.get_logger().info("Averaging client parameters")
    parameters = [clients[client_idx].get_nn_parameters() for client_idx in random_workers]

    if (args.get_attack_type() == "random_noise_update" or args.get_attack_type() == "random_noise_addition"):
        apply_random_noise(random_workers, poisoned_workers, parameters, args.get_attack_type(), args.get_target(), args.get_is_targeted)

    if args.get_attack_type() == "fang_trim":
        apply_fang_trim_mean(random_workers, poisoned_workers, parameters)
    
    new_nn_params = average_nn_parameters(parameters, len(parameters))

    for client in clients:
        #args.get_logger().info("Updating parameters on client #{}", str(client.get_client_index()))
        client.update_nn_parameters(new_nn_params)

    return clients[0].test(), random_workers, parameters

def create_clients(args, train_data_loaders, test_data_loader):
    """
    Create a set of clients.
    """
    clients = []
    for idx in range(args.get_num_workers()):
        clients.append(Client(args, idx, train_data_loaders[idx], test_data_loader))

    return clients

def run_machine_learning(clients, args, poisoned_workers):
    """
    Complete machine learning over a series of clients.
    """
    epoch_test_set_results = []
    epoch_backdoor_results = []
    worker_selection = []
    update = []
    for epoch in range(1, args.get_num_epochs() + 1):
        results, workers_selected, parameters = train_subset_of_clients(epoch, args, clients, poisoned_workers)

        epoch_test_set_results.append(results[:4])
        epoch_backdoor_results.append(results[4:])
        worker_selection.append(workers_selected)

    return convert_results_to_csv(epoch_test_set_results), convert_results_to_csv(epoch_backdoor_results), worker_selection#, convert_results_to_csv(update)

def run_exp(replacement_method, num_poisoned_workers, KWARGS, client_selection_strategy, idx):
    log_files, results_files, models_folders, worker_selections_files = generate_experiment_ids(idx, 1)

    # Initialize logger
    args = Arguments(logger)
    if not os.path.exists(args.get_attack_type()):
        os.mkdir(args.get_attack_type())
    handler = logger.add(args.get_attack_type() + "/" +log_files[0], enqueue=True)

    
    args.set_model_save_path(models_folders[0])
    args.set_num_poisoned_workers(num_poisoned_workers)
    args.set_round_worker_selection_strategy_kwargs(KWARGS)
    args.set_client_selection_strategy(client_selection_strategy)
    args.log()

    train_data_loader = load_train_data_loader(logger, args)
    test_data_loader = load_test_data_loader(logger, args)

    # Distribute batches equal volume IID
    distributed_train_dataset = distribute_batches_equally(train_data_loader, args.get_num_workers())
    distributed_train_dataset = convert_distributed_data_into_numpy(distributed_train_dataset)

    poisoned_workers = identify_random_elements(args.get_num_workers(), args.get_num_poisoned_workers())
    distributed_train_dataset = poison_data(logger, distributed_train_dataset, args.get_num_workers(), poisoned_workers, replacement_method, args.get_attack_type())
    train_data_loaders = generate_data_loaders_from_distributed_dataset(distributed_train_dataset, args.get_batch_size())


    clients = create_clients(args, train_data_loaders, test_data_loader)

    results, results_backdoor, worker_selection = run_machine_learning(clients, args, poisoned_workers)
    

    
    if (args.get_attack_type() == "dba" or args.get_attack_type() == "backdoor"):
        save_results(results_backdoor, args.get_attack_type() + "/" + results_files[0])
    else:    
        save_results(results, args.get_attack_type() + "/" + results_files[0])
    save_results(worker_selection, args.get_attack_type() + "/" + worker_selections_files[0])

    logger.remove(handler)
