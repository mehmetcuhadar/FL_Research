import numpy as np
import torch
from scipy.stats import norm
import math

def apply_fang_trim_mean(random_workers, poisoned_workers, parameters):
   
    benign_update = []

    index = []
    counter = 0
    for r_w in random_workers:
        if r_w in poisoned_workers:
            index.append(counter)
        counter += 1

    counter = 0

    for w in random_workers:
        if not counter in index:
            tensor = torch.tensor(np.nan_to_num(w))
            benign_update.append(tensor)
        counter += 1
            
    
    num_byzantine = len(poisoned_workers)

    benign_update = torch.stack(benign_update, 0)
    
    benign_update_long = benign_update.float()
    
    agg_grads = torch.mean(benign_update_long, 0)
    deviation = torch.tensor([torch.sign(agg_grads)])
    
    device = benign_update_long.device
    b = 2
    
    inter = torch.max(benign_update_long, 0)[0]
    inter = torch.tensor([inter])
    inter2 = torch.min(benign_update_long, 0)[0]
    inter2 = torch.tensor([inter2])
    
    max_vector = inter
    min_vector = inter2

    max_ = (max_vector > 0).type(torch.FloatTensor).to(device)
    min_ = (min_vector < 0).type(torch.FloatTensor).to(device)

    max_[max_ == 1] = b
    max_[max_ == 0] = 1 / b
    min_[min_ == 1] = b
    min_[min_ == 0] = 1 / b

    max_range = torch.cat(
        (max_vector[:, None], (max_vector * max_)[:, None]), dim=1
    )
    min_range = torch.cat(
        ((min_vector * min_)[:, None], min_vector[:, None]), dim=1
    )

    rand = (
        torch.from_numpy(
            np.random.uniform(0, 1, [len(deviation), num_byzantine])
        )
        .type(torch.FloatTensor)
        .to(benign_update.device)
    )

    max_rand = (
        torch.stack([max_range[:, 0]] * rand.shape[1]).T
        + rand * torch.stack([max_range[:, 1] - max_range[:, 0]] * rand.shape[1]).T
    )
    min_rand = (
        torch.stack([min_range[:, 0]] * rand.shape[1]).T
        + rand * torch.stack([min_range[:, 1] - min_range[:, 0]] * rand.shape[1]).T
    )

    mal_vec = (
        torch.stack(
            [(deviation < 0).type(torch.FloatTensor)] * max_rand.shape[1]
        ).T.to(device)
        * max_rand
        + torch.stack(
            [(deviation > 0).type(torch.FloatTensor)] * min_rand.shape[1]
        ).T.to(device)
        * min_rand
    ).T
    
    counter = 0

    for i, client in enumerate(parameters):
        if counter in index:
            client["fc.weight"].data = mal_vec[i]
        counter += 1