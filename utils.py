import torch
import matplotlib.pyplot as plt

# data loading
def get_batch(data, split, block_size, batch_size, device='cpu'):
    # generate a small batch of data of inputs x and targets y
    data = data[split]
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# data loading
def get_batch_(data, split, block_size, batch_size, device='cpu'):
    # generate a small batch of data of inputs x and targets y
    in_data, out_data = data[split]
    ix = torch.randint(len(in_data) - block_size, (batch_size,))
    x = torch.stack([in_data[i:i+block_size] for i in ix])
    y = torch.stack([out_data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def calculate_cooc(sequence, N, K, device='cpu', index_dict=None):
    cooc_tensor = torch.zeros((N, N, K))
    for i,d in enumerate(sequence):
        if index_dict is not None:
            d = index_dict[d.item()]
        for k in range(K):
            if i+k+1<len(sequence):
                nd = sequence[i+k+1]
                if index_dict is not None:
                    nd = index_dict[nd.item()]
                cooc_tensor[d,nd,k] += 1
    return cooc_tensor.to(device)

@torch.no_grad()
def estimate_loss_(model, data_dict, eval_iters, block_size, batch_size, device='cpu'):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_(data_dict, split, block_size, batch_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

@torch.no_grad()
def estimate_loss(model, data_dict, eval_iters, block_size, batch_size, device='cpu'):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data_dict, split, block_size, batch_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out