def apply_sign_flipping(random_workers, poisoned_workers, parameters):
    index = []
    counter = 0
    for r_w in random_workers:
        if r_w in poisoned_workers:
            index.append(counter)
        counter += 1
    counter = 0
    for params in parameters:
        if counter in index: 
            q = params["fc.weight"].data
            params["fc.weight"].data = -1 * q         
        counter += 1