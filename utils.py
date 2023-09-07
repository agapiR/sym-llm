import torch

# data loading
def get_batch(data, split, block_size, batch_size, device='cpu'):
    # generate a small batch of data of inputs x and targets y
    data = data[split]
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def calculate_cooc(sequence, N, K, device='cpu'):
    cooc_tensor = torch.zeros((N, N, K))
    for i,d in enumerate(sequence):
        for k in range(K):
            if i+k+1<len(sequence):
                nd = sequence[i+k+1]
                cooc_tensor[d,nd,k] += 1
    return cooc_tensor.to(device)