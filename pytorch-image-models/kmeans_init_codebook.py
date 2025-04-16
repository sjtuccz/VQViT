import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack, unpack
from sklearn.decomposition import PCA
import numpy as np

def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device = device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device = device)

    return samples[indices]

def batched_sample_vectors(samples, num):
    return torch.stack([sample_vectors(sample, num) for sample in samples.unbind(dim = 0)], dim = 0)
def noop(*args, **kwargs):
    pass
def cdist(x, y):
    x2 = reduce(x ** 2, 'b n d -> b n', 'sum')
    y2 = reduce(y ** 2, 'b n d -> b n', 'sum')
    xy = einsum('b i d, b j d -> b i j', x, y) * -2
    return (rearrange(x2, 'b i -> b i 1') + rearrange(y2, 'b j -> b 1 j') + xy).clamp(min = 0).sqrt()
def batched_bincount(x, *, minlength):
    batch, dtype, device = x.shape[0], x.dtype, x.device
    target = torch.zeros(batch, minlength, dtype = dtype, device = device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values)
    return target

def l2norm(t, dim = -1,  eps = 1e-6):
    return F.normalize(t, p = 2, dim = dim, eps = eps)
def kmeans(
    samples,
    num_clusters,
    num_iters = 10,
    use_cosine_sim = False,
    sample_fn = batched_sample_vectors,
    all_reduce_fn = noop
):
    num_codebooks, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device

    means = sample_fn(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ rearrange(means, 'h n d -> h d n')
        else:
            dists = -cdist(samples, means)

        buckets = torch.argmax(dists, dim = -1)
        bins = batched_bincount(buckets, minlength = num_clusters)
        all_reduce_fn(bins)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype = dtype)

        new_means.scatter_add_(1, repeat(buckets, 'h n -> h n d', d = dim), samples)
        new_means = new_means / rearrange(bins_min_clamped, '... -> ... 1')
        all_reduce_fn(new_means)

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(
            rearrange(zero_mask, '... -> ... 1'),
            means,
            new_means
        )

    return means, bins
def kmeans_pca_fit_dim(token,
    codebook_num = 256,
    codebook_dim = 2,
    num_iters = 10,use_cosine_sim = False):
    '''token shape (b,h,d)'''
    token = token.cpu()
    samples = rearrange(token, 'b h d -> 1 (b h) d') 
    centroids, counts = kmeans(samples, num_clusters=codebook_num, num_iters=num_iters,use_cosine_sim=use_cosine_sim)
    centroids = centroids.squeeze(dim=0)
    if centroids.shape[-1] != codebook_dim:
        x_np = centroids.numpy() if isinstance(centroids, torch.Tensor) else centroids
        pca = PCA(n_components=codebook_dim)
        x_reduced = pca.fit_transform(x_np)
        centroids = torch.from_numpy(x_reduced)
    return centroids, counts



if __name__ == '__main__':
    x = torch.randn(128, 193, 384)
    centroids, counts = kmeans_pca_fit_dim(x)
    print(centroids.shape)