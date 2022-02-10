import torch

def inverse_transform_sampling(X, pdf, n, device=None):
    '''
    X: (Batch_size, Nc, 1)
    pdf: (Batch_size, Nc, 1)
    '''
    X = torch.concat((torch.tensor([0.0], device=device).expand(X.shape[0], 1), X.squeeze(-1)), dim=1) # (Batch_size, Nc + 1)
    U = torch.rand(pdf.shape[0], n, device=device) # (Batch_size, n)
    cdf = torch.cumsum(pdf.squeeze(-1), -1) # (Batch_size, Nc)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1], device=device), cdf], dim=-1) # (Batch_size, Nc + 1)

    inds = torch.searchsorted(cdf, U, right=True)
    #samples = (X[inds] - X[inds - 1]) * torch.rand(pdf.shape[0], n, device=device) + X[inds - 1]
    samples = torch.zeros(pdf.shape[0], n)
    for i in range(samples.shape[0]):
        samples[i] = (X[i,inds[i]] - X[i,inds[i] - 1]) * torch.rand(n, device=device) + X[i,inds[i] - 1]

    return samples # (Batch_size, n)

def integrate_color(N, density, color, ts, device=None):
    '''
    N: int
    density: Tensor [Batch_size, N, 1]
    color: Tensor [Batch_size, N, 3]
    ts: Tesnor [Batch_size, N, 1]
    '''
    batch_size = density.shape[0]
    ts = ts.squeeze(-1)
    delta = ts[:,1:] - ts[:,:-1]
    T = torch.zeros(batch_size, N, device=device)
    T[...,1:] = density[...,:-1] * delta
    T = torch.cumsum(T, dim=1)
    T = torch.exp(-T) # (Batch_size, N)

    weights = T * torch.concat(((1 - torch.exp(-T))[...,1:], torch.zeros(batch_size, 1, device=device)), dim=-1) # (Batch_size, N)

    output_color = torch.sum(weights.unsqueeze(-1) * color, dim=1) # (Batch_size, 3)
    
    return output_color, weights
