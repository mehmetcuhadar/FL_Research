import numpy as np
def apply_random_noise(random_workers, poisoned_workers, parameters, attack_type, target):
    index = []
    counter = 0
    for r_w in random_workers:
        if r_w in poisoned_workers:
            index.append(counter)
        counter += 1
    counter = 0
    target_counter = -1
    for params in parameters:
        if counter in index:
            for param in params["fc.weight"].data:
                target_counter += 1
                if target_counter != target: continue
                for j in range(len(param)):
                    if attack_type == "random_noise_update" : param[j] = np.random.normal()
                    else: param[j] += np.random.normal()
                
            target_counter = -1
        counter += 1