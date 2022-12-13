from .label_replacement import apply_class_label_replacement
from .backdoor import apply_backdoor
from .dba_attack import apply_dba
from .client_utils import log_client_data_statistics
from federated_learning.arguments import Arguments

def poison_data(logger, distributed_dataset, num_workers, poisoned_worker_ids, replacement_method, attack_type):
    """
    Poison worker data

    :param logger: logger
    :type logger: loguru.logger
    :param distributed_dataset: Distributed dataset
    :type distributed_dataset: list(tuple)
    :param num_workers: Number of workers overall
    :type num_workers: int
    :param poisoned_worker_ids: IDs poisoned workers
    :type poisoned_worker_ids: list(int)
    :param replacement_method: Replacement methods to use to replace
    :type replacement_method: list(method)
    :attack_type: labelflipping or backdoor : string
    """
    # TODO: Add support for multiple replacement methods?
    poisoned_dataset = []

    class_labels = list(set(distributed_dataset[0][1]))
    args = Arguments(logger)
    logger.info("Poisoning data for workers: {}".format(str(poisoned_worker_ids)))

    for worker_idx in range(num_workers):
        if worker_idx in poisoned_worker_ids:
            if attack_type == "label_flipping":
                poisoned_dataset.append(apply_class_label_replacement(distributed_dataset[worker_idx][0], distributed_dataset[worker_idx][1], replacement_method))
            elif attack_type == "backdoor":
                poisoned_dataset.append(apply_backdoor(distributed_dataset[worker_idx][0], distributed_dataset[worker_idx][1], args.get_backdoor_target(),args.get_backdoor_intense()))
            elif attack_type == "dba":
                index = 0
                for i in range(len(poisoned_worker_ids)):
                    if poisoned_worker_ids[i] == worker_idx:
                        index = i
                remainder = index % 4
                poisoned_dataset.append(apply_dba(distributed_dataset[worker_idx][0], distributed_dataset[worker_idx][1], args.get_backdoor_target(), args.get_backdoor_intense(),remainder))
            else:
                poisoned_dataset.append(distributed_dataset[worker_idx])
        else:
            poisoned_dataset.append(distributed_dataset[worker_idx])

    log_client_data_statistics(logger, class_labels, poisoned_dataset)

    return poisoned_dataset
