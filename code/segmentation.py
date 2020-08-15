import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind as ttest

import torch
import torch.nn as nn

from utility import get_label_image, get_tensor_slice, get_prime_factors, get_local_median, cosine_similarity
from models import UNet

# from optical_electrophysiology import extract_traces
# def get_traces(mat, softmask, label_image=None, regions=None, min_pixels=20, percentile=50, label_size_max_portion=0.5, min_thresh=0.1):
#     """Extract traces from mat and softmask (usually correlation map)
#     Args:
#         mat: 3D torch.Tensor with shape (nframe, nrow, ncol)
#         softmask: 2D torch.Tensor with shape (nrow, ncol)
#         label_image: default None, call get_label_image; otherwise 2D torch.Tensor with shape (nrow, ncol)

#     Returns:
#         label_image: 2D torch.Tensor with shape (nrow, ncol)
#         traces: a list of 1D torch.Tensors

#     """
#     if label_image is None:
#         label_image, regions = get_label_image(softmask, min_pixels=min_pixels, label_size_max_portion=label_size_max_portion, min_thresh=min_thresh)
#     submats, traces = extract_traces(mat, softmask=softmask, label_image=label_image, regions=regions, percentile=percentile)
#     return label_image, traces
    
def get_high_conf_mask(cor_map, low_percentile=25, high_percentile=5, min_cor=0.1, min_pixels=20, exclude_boundary_width=2):
    """Return a high-confidence mask
    """
    label_image, regions = get_label_image(cor_map, min_pixels=min_pixels)
    fg_mask = ((cor_map >= np.percentile(cor_map.cpu(), high_percentile)) & (cor_map.new_tensor(label_image)>0) & (cor_map > min_cor))
    bg_mask = ((cor_map <= np.percentile(cor_map.cpu(), low_percentile)) & (cor_map.new_tensor(label_image)==0) & (cor_map < min_cor))
    mask = cor_map.new_full(cor_map.size(), -1)
    mask[bg_mask] = 0
    mask[fg_mask] = 1
    mask[:exclude_boundary_width] = -1
    mask[-exclude_boundary_width:] = -1
    mask[:, :exclude_boundary_width] = -1
    mask[:, -exclude_boundary_width:] = -1
    return mask

def semi_supervised_segmentation(mat, cor_map=None, model=None, out_channels=[8,16,32], kernel_size=3, frames_per_iter=100, 
                                 num_iters=200, print_every=1, select_frames=False, return_model=False,
                                 optimizer_fn=torch.optim.AdamW, optimizer_fn_args = {'lr': 1e-2, 'weight_decay': 1e-3}, 
                                 loss_threshold=0, save_loss_folder=None, reduction='max', last_out_channels=None,
                                 verbose=False, device=torch.device('cuda')):
    """Semi-supervised semantic segmentation
    Args: 
        mat: torch.Tensor
    
    Returns:
        soft_mask: 2D torch.Tensor
        model: learned model if return_model is True
    """
    if cor_map is None:
        cor_map = get_cor_map_4d(mat, select_frames=select_frames)
    high_conf_mask = get_high_conf_mask(cor_map)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([(high_conf_mask==1).sum(), (high_conf_mask==0).sum()]).float().to(device))
    loss_history = []
    if model is None:
#         nrow, ncol = mat.shape[-2:]
#         pool_kernel_size_row = get_prime_factors(nrow)[:3]
#         pool_kernel_size_col = get_prime_factors(ncol)[:3]
#         model = UNet(in_channels=mat.shape[0], num_classes=2, out_channels=out_channels, num_conv=2, n_dim=3, 
#                      kernel_size=[3, (3, 3, 3), (3, 3, 3), (3, 3, 3), (1, 3, 3), (1, 3, 3), (1, 3, 3)], 
#                      padding=[1, (0, 1, 1), (0, 1, 1), (0, 1, 1), (0, 1, 1), (0, 1, 1), (0, 1, 1)], 
#                      pool_kernel_size=[(2, pool_kernel_size_row[0], pool_kernel_size_col[0]), 
#                                        (2, pool_kernel_size_row[1], pool_kernel_size_col[1]), 
#                                        (2, pool_kernel_size_row[2], pool_kernel_size_col[2])], 
#                      use_adaptive_pooling=True, same_shape=False,
#                      transpose_kernel_size=[(1, pool_kernel_size_row[2], pool_kernel_size_col[2]), 
#                                             (1, pool_kernel_size_row[1], pool_kernel_size_col[1]), 
#                                             (1, pool_kernel_size_row[0], pool_kernel_size_col[0])], 
#                      transpose_stride=[(1, pool_kernel_size_row[2], pool_kernel_size_col[2]), 
#                                        (1, pool_kernel_size_row[1], pool_kernel_size_col[1]), 
#                                        (1, pool_kernel_size_row[0], pool_kernel_size_col[0])],
#                      padding_mode='zeros', normalization='layer_norm',
#                      activation=nn.LeakyReLU(negative_slope=0.01, inplace=True)).to(device)
        if mat.ndim == 3:
            mat = mat.unsqueeze(0)
        in_channels = mat.shape[0]
        num_classes = 2
        if isinstance(kernel_size, int):
            padding = (kernel_size-1)//2
            padding_row = padding_col = padding
        elif isinstance(kernel_size, tuple):
            assert len(kernel_size)==3
            padding, padding_row, padding_col = [(k-1)//2 for k in kernel_size]
        encoder_depth = len(out_channels)
        nframe, nrow, ncol = mat.shape[-3:]
        if isinstance(kernel_size, int):
            assert nframe > 4*encoder_depth*(kernel_size-1)
        else:
            assert nframe > 4*encoder_depth*(kernel_size[0]-1)
        pool_kernel_size_row = get_prime_factors(nrow)[:encoder_depth]
        pool_kernel_size_col = get_prime_factors(ncol)[:encoder_depth]
        model = UNet(in_channels=in_channels, num_classes=num_classes, out_channels=out_channels, num_conv=2, n_dim=3, 
                     kernel_size=kernel_size, 
                     padding=[padding] + [(0, padding, padding)]*encoder_depth*2, 
                     pool_kernel_size=[(1, pool_kernel_size_row[i], pool_kernel_size_col[i]) for i in range(encoder_depth)], 
                     use_adaptive_pooling=True, same_shape=False,
                     transpose_kernel_size=[(1, pool_kernel_size_row[i], pool_kernel_size_col[i]) 
                                            for i in reversed(range(encoder_depth))], 
                     transpose_stride=[(1, pool_kernel_size_row[i], pool_kernel_size_col[i]) 
                                       for i in reversed(range(encoder_depth))],
                     padding_mode='zeros', normalization='layer_norm', last_out_channels=last_out_channels,
                     activation=nn.LeakyReLU(negative_slope=0.01, inplace=True)).to(device)
    optimizer = optimizer_fn(filter(lambda p: p.requires_grad, model.parameters()), **optimizer_fn_args)
    
    idx = torch.nonzero(high_conf_mask!=-1, as_tuple=True)
    y_true = high_conf_mask[idx].long()
    for i in range(num_iters):
#         if (i+1) == num_iters//2:
#             optimizer_fn_args['lr'] /= 10
#             optimizer = optimizer_fn(filter(lambda p: p.requires_grad, model.parameters()), **optimizer_fn_args)
        x = get_tensor_slice(mat, dims=[1], sizes=[frames_per_iter])
        y_pred = model(x)
        if reduction == 'max':
            y_pred = y_pred.max(1)[0]
        elif reduction == 'mean':
            y_pred = y_pred.mean(1)
        elif reduction.startswith('top'):
            n = y_pred.size(1)
            if reduction.endswith('percent'):
                k = max(int(int(reduction[3:-7])/100. * n), 1)
            else:
                k = min(int(reduction[3:]), n)
            y_pred = y_pred.topk(k, dim=1)[0].mean(1)
        else:
            raise ValueError(f'reduction = {reduction} not handled!')
        y_pred = y_pred[:, idx[0], idx[1]].T
        loss = loss_fn(y_pred, y_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if verbose and ((i+1)%print_every == 0 or i==0 or i==num_iters-1):
            print(f'{i+1} loss={loss.item()}')
        if loss_threshold>0 and (i+1)%print_every==0 and np.mean(loss_history[-print_every:])<loss_threshold:
            break
    if verbose:
        plt.title('Training loss')
        plt.plot(loss_history, 'ro-', markersize=2)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.show()
    if save_loss_folder is not None and os.path.exists(save_loss_folder):
        np.save(f'{save_loss_folder}/loss__semi_supervised_segmentation.npy', loss_history)
    with torch.no_grad():
        soft_masks = []
        num_iters = mat.shape[1] // frames_per_iter + 1
        for i in range(num_iters):
            x = get_tensor_slice(mat, dims=[1], sizes=[frames_per_iter])
            y_pred = model(x)
            if reduction == 'max':
                y_pred = y_pred.max(1)[0]
            elif reduction == 'mean':
                y_pred = y_pred.mean(1)
            elif reduction.startswith('top'):
                n = y_pred.size(1)
                if reduction.endswith('percent'):
                    k = max(int(int(reduction[3:-7])/100. * n), 1)
                else:
                    k = min(int(reduction[3:]), n)
                y_pred = y_pred.topk(k, dim=1)[0].mean(1)
            else:
                raise ValueError(f'reduction = {reduction} not handled!')
            soft_mask = torch.softmax(y_pred, dim=0)[1]
            soft_masks.append(soft_mask)
        soft_mask = torch.stack(soft_masks, dim=0).mean(0)
    if return_model:
        return soft_mask, model 
    else:
        return soft_mask
    
    
def simple_cosine_similarity(x, y=None):
    """
    Args:
        x: (n_x, dim)
        y: (n_y, dim)
    
    Returns:
        cos: (n_x, n_y)
    """
    if y is None:
        y = x
    x = x - x.mean(1, keepdim=True)
    y = y - y.mean(1, keepdim=True)
    cos = torch.mm(x, y.T) / x.norm(dim=1, keepdim=True) / y.norm(dim=1)
    return cos


def manhattan_distance(coords):
    """
    Args:
        coords: (n_pts, ndim)
        
    Returns:
        dist: (n_pts, n_pts)
    """
    return (coords.unsqueeze(2) - coords.T).abs().sum(1)


def graph_laplacian(weight, normalized=True):
    """
    Args:
        weight: non-negative symmetric 2D matrix
       
    Returns:
        L: symetric graph laplacian
        
    """
    assert weight.ndim==2 and weight.min() >= 0
    diagonal = weight.sum(1)
    L = torch.diag(diagonal) - weight
    if normalized and diagonal.min() > 0:
        d = 1./torch.sqrt(diagonal)
        L = d.unsqueeze(0) * L * d.unsqueeze(1)
    return L


def pairwise_dist(x, y=None, sqrt=False):
    """
    Args:
        x: (n_pts, ndim)
        y: if None, set to be x
        
    Returns:
        dist: (n_pts, n_pts)
        
    """
    if y is None:
        y = x
    dist = (x*x).sum(1, keepdim=True) + (y*y).sum(1) - 2*torch.mm(x, y.T)
    dist[dist<0] = 0
    if sqrt:
        return torch.sqrt(dist)
    else:
        return dist

    
def k_means(points, n_clusters=2, random_init=False, return_centers=False):
    """
    Args:
        points: (n_pts, ndim)
    
    Returns:
        if return_centers is True:
            indices: (n_pts, ), LongTensor, labels
            values: (n_pts, ), distance to center
            centers: (n_clusters, ndim)
        
    Examples:
        good_colors = get_good_colors()
        points = torch.randn((100, 2))
        indices, values, centers = k_means(points, n_clusters=10, return_centers=True)
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.scatter(points[:, 0].cpu().numpy(), points[:, 1].cpu().numpy(), c=[good_colors[i.item()] for i in indices], marker='o')
        ax.scatter(centers[:, 0].cpu(), centers[:, 1].cpu(), c=good_colors[:10], marker='v', s=50)
        plt.show()
    """
    n_pts, ndim = points.shape
    if random_init:
        centers = points[torch.randperm(n_pts)[:n_clusters]]
    else:
        sel_idx = np.random.choice(n_pts)
        centers = points[sel_idx].unsqueeze(0)
        for _ in range(n_clusters-1):
            dist = pairwise_dist(centers, points).min(dim=0)[0]
            sel_idx = torch.multinomial(dist, 1)
            centers = torch.cat([centers, points[sel_idx]], dim=0)
    labels = torch.zeros(n_pts).long().to(points.device)
    indices = torch.ones(n_pts).long().to(points.device)
    while not torch.equal(labels, indices):
        labels = indices
        dist = pairwise_dist(centers, points)
        values, indices = dist.min(dim=0) #ToDo: check empty clusters
        centers = torch.stack([points[indices==i].mean(0) for i in range(n_clusters)], dim=0)
    if return_centers:
        return indices, values, centers
    else:
        return indices, values.sum()

    
def get_fft(mat, signal_dim=0, max_freq=200, normalized=True, reduction='norm'):
    """Calculate FFT for one dimension
    """
    if max_freq is None:
        max_freq = mat.size(signal_dim) // 2 + 1
    signal_dim = (signal_dim + mat.ndim) % mat.ndim # handle the case where signal_dim == -1
    if signal_dim != mat.ndim - 1: 
        # if signal_dim is not the last dimension
        mat_fft = mat.transpose(signal_dim, -1)
    else:
        mat_fft = mat
    mat = torch.rfft(mat_fft, signal_ndim=1, normalized=normalized)[..., :max_freq, :]
    if reduction == 'cat':
        mat = mat.reshape(*mat.shape[:-2], max_freq*2)         
    elif reduction == 'norm':
        mat = mat.norm(dim=-1)
    elif reduction == 'real':
        mat = mat[..., 0]
    elif reduction == 'imaginary':
        mat = mat[..., 1]
    if signal_dim != mat.ndim - 1:
        mat = mat.transpose(signal_dim, -1)
    return mat
 
    
def spectral_clustering(graph, k=2):
    """
    Args:
        graph: 2D torch.Tensor, non-negative symetric weighted adjacent matrix for a graph
        
    Returns:
        labels: cluster assignment starting from 0
    """
    laplacian = graph_laplacian(graph)
    eigenvalues, eigenvectors = torch.symeig(laplacian, eigenvectors=True)
    embedding = eigenvectors[:, :k]
    embedding = embedding / torch.norm(embedding, dim=1, keepdim=True)
    labels, dist = k_means(embedding, k)
    return labels


def cal_cor(x, y, median_filter=True, fft=True, window_size=50, fft_max_freq=100):
    """
    Args:
        x: (n_pnts, n_dim)
        y: (n_pnts, n_dim)
        
    Returns:
        cor: (n_pnts, )
        torch.cat([x, y], dim=0)
    """
    if median_filter:
        x = x - get_local_median(x, window_size=window_size, dim=1)
        y = y - get_local_median(y, window_size=window_size, dim=1)
    if fft:
        if fft_max_freq is None or fft_max_freq > x.size(1)//2 + 1:
            fft_max_freq = x.size(1)//2 + 1
        x = torch.rfft(x, signal_ndim=1, normalized=True)[..., :fft_max_freq, :].norm(dim=-1)
        y = torch.rfft(y, signal_ndim=1, normalized=True)[..., :fft_max_freq, :].norm(dim=-1)
    cor = cosine_similarity(x, y, dim=1)
    return cor, torch.cat([x, y], dim=0)

def paired_cor(coords, mat, num_samples=20, num_pts_per_sample=50, exponent=6):
    """Select two regions that are far away from each other and calculate correlation between these regions
    Args:
        coords: (n_pts, 2), each row specifies (row, col)
        mat: torch.Tensor with shape (nframe, nrow, ncol)
        num_samples: default 20, how many random samples to choose
        num_pts_per_sample: default 50, how many points are chosen for calculating mean trace for each region
        exponent: default 6, a parameter to make random selections favor farest points
    
    Returns:
        cor: torch.Tensor with shape (n_samples, )
        val: torch.Tensor with shape (2*num_samples, nframe)
    """
    coords = coords.float()
    # randomly select one "farest" point from the center
    weight = torch.pow(coords - coords.mean(dim=0), exponent=exponent).sum(dim=1).unsqueeze(0).expand(num_samples, len(coords))
    idx1 = torch.multinomial(weight, 1) # first batch center indices
    weight = torch.pow(coords - coords[idx1], exponent=exponent).sum(dim=-1) # weight for selecting second batch centers
    idx2 = torch.multinomial(weight, 1) # second batch center indices
    k = min(num_pts_per_sample, len(coords)//2)
    idx1 = (-weight).topk(k=k, dim=1)[1]
    idx2 = (-(coords - coords[idx2]).abs().sum(dim=-1)).topk(k=k, dim=1)[1]
    
    m = mat[..., coords[:, 0].long(), coords[:, 1].long()].T
    x, y = m[idx1].mean(dim=1), m[idx2].mean(dim=1)
    cor, val = cal_cor(x, y)
    return cor, val

def split_clusters(sel_label_idx, mat, label_image, cor_threshold=0.8, soft_threshold=False, min_num_pixels=50, max_num_pixels=1200, 
                   max_dist=2, median_detrend=True, apply_fft=True, fft_max_freq=200, verbose=True, plot=False, soma_coords=None):
    """Given a cluster, split it into more clusters if criteria are met
    Args:
        sel_label_idx: int, the cluster id to be splitted; note 0 is background in label_image, labels are numbered starting from 1
        mat: torch.Tensor of shape (nframe, nrow, ncol)
        label_image: torch.Tensor of shape (nrow, ncol)
        cor_threshold: correlation threshold for splitting a cluster
        soft_threshold: if True, use statistical test to automatically split clusters
        min_num_pixels: minimal number of pixels for a cluster
        max_num_pixels: maximal number of pixels for a cluster
        max_dist: used to build a nearest neigbhor graph based on Manhattan distance smaller than 2
        median_detrend: if true, apply median filter to traces before applying FFT or calculating correlation
        apply_fft: if True, apply FFT before calculating correlation
        fft_max_freq: threshold to filter out high-frequency noise (assuming signals mainly have low frequency)
        soma_coords: only for plot with ground truths
    
    Returns:
        inplace modification of label_image
    """
    def show_plot():
        image = torch.zeros_like(cor)
        image[coords[:, 0], coords[:, 1]] = image.new_tensor(labels+1)
        minr, minc = coords.min(dim=0)[0]
        maxr, maxc = coords.max(dim=0)[0]
        fig, ax = plt.subplots()
        ax.imshow(image[minr:maxr, minc:maxc].cpu().numpy())
        ax.set_title(f'')
        for i, (x, y) in enumerate(soma_coords):
            if x >= minr and x < maxr and y >= minc and y < maxr:
                ax.scatter([(x-minr).item()], [(y-minc).item()], s=30, c='r')
                ax.text((x-minr).item()+1, (y-minc).item(), i+1, color='r', fontweight='bold')
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()
    coords = torch.nonzero(label_image==sel_label_idx)
    if len(coords) < min_num_pixels:
        if verbose:
            print(f'Stop splitting: Cluster must have at least {min_num_pixels} pixels but have {len(coords)} pixels instead!')
        return
    dist = manhattan_distance(coords)
    dist.fill_diagonal_(max_dist + 1)
    adj_mat = (dist <= max_dist)
    
    dtype = mat.dtype
    if dtype == torch.float16:
        mat = mat.float()
    m = mat[:, coords[:, 0], coords[:, 1]]
    if median_detrend:
        m = m - get_local_median(m, window_size=50, dim=0)
    if apply_fft:
        m_transformed = get_fft(m.T, signal_dim=1, max_freq=fft_max_freq, normalized=True, reduction='norm')
    else:
        m_transformed = m.T
    m_transformed = m_transformed - m_transformed.mean(1, keepdim=True)
    norm = torch.norm(m_transformed, dim=1)
    cor = torch.mm(m_transformed, m_transformed.T) / (norm.unsqueeze(1) * norm)
    cor[~adj_mat] = 0
    cor[cor < 0] = 0
#     assert torch.equal(cor, cor.T)

    # spectral clustering
    labels = spectral_clustering(graph=cor, k=2)
    bincount = torch.bincount(labels)
    val_cnt, idx_cnt = bincount.sort()
    if val_cnt[0] <= min_num_pixels and val_cnt[1] <= max_num_pixels:
        if verbose:
            print(f'Stop spliting: One cluster is too small and the other cluster is not too big: {val_cnt.tolist()}')
        return
    if val_cnt[0] <= min_num_pixels//2 and val_cnt[1] > max_num_pixels:
        # one cluster is too small and the other cluster is too big
        row_idx, col_idx = coords[labels==idx_cnt[0]].T
        label_image[row_idx, col_idx] = 0
        if verbose:
            print(f'Reset {val_cnt[0]} pixels to background')
        return split_clusters(sel_label_idx, mat, label_image, cor_threshold=cor_threshold, soft_threshold=soft_threshold, 
                              min_num_pixels=min_num_pixels, max_dist=max_dist, median_detrend=median_detrend, apply_fft=apply_fft, 
                              fft_max_freq=fft_max_freq, verbose=verbose)
    if soft_threshold:
        cor1, val1 = paired_cor(coords[labels==idx_cnt[0]], mat)
        cor2, val2 = paired_cor(coords[labels==idx_cnt[1]], mat)
        cor3 = cosine_similarity(val1, val2, dim=1)
        statistic, p_val = ttest(torch.cat([cor1, cor2], dim=0).cpu(), cor3.cpu())
        if verbose:
            statistic1, p_val1 = ttest(cor1.cpu(), cor3.cpu())
            statistic2, p_val2 = ttest(cor2.cpu(), cor3.cpu())
            print(f'cor1-cor3: statistic={statistic1}, p_val={p_val1}')
            print(f'cor2-cor3: statistic={statistic2}, p_val={p_val2}')
            print(f'[cor1, cor2]-cor3: statistic={statistic}, p_val={p_val}')
            print('Correlation mean: sample1={:.2f} sample2={:.2f} sample1:sample2={:.2f}'.format(
                cor1.mean().item(), cor2.mean().item(), cor3.mean().item()))
            print('Correlation std:  sample1={:.2f} sample2={:.2f} sample1:sample2={:.2f}'.format(
                cor1.std().item(), cor2.std().item(), cor3.std().item()))
        if statistic < 0 and p_val < 0.05 and cor3.mean() > 0.5:
            if verbose:
                print(f'Stop splitting: statistic={statistic} and p_val={p_val} and cor3.mean()={cor3.mean()}')
            return
    else:
        x = torch.stack([m[:, labels==i].mean(1) for i in range(2)], dim=0)
        if median_detrend:
            x = x - get_local_median(x, window_size=50, dim=-1)
        if apply_fft:
            x = torch.rfft(x, signal_ndim=1, normalized=True)[..., :fft_max_freq, :].reshape(x.size(0), -1)
        cor_val = simple_cosine_similarity(x[:1], x[1:]).item()
        if cor_val > cor_threshold:
            if verbose:
                print(f'Stop splitting: cor_val ({cor_val}) > cor_threshold ({cor_threshold})')
            return
    row_idx, col_idx = coords[labels==idx_cnt[0]].T
    label_image[row_idx, col_idx] = label_image.max() + 1
    if verbose:
        print(f'Split {sel_label_idx} into {sel_label_idx} ({val_cnt[1]}) and {label_image.max().item()} ({val_cnt[0]})')
    if plot:
        show_plot()
    if (labels==idx_cnt[1]).sum() > min_num_pixels:
        return split_clusters(sel_label_idx, mat, label_image, cor_threshold=cor_threshold, soft_threshold=soft_threshold, 
                              min_num_pixels=min_num_pixels, max_dist=max_dist, median_detrend=median_detrend, apply_fft=apply_fft, 
                              fft_max_freq=fft_max_freq, verbose=verbose)