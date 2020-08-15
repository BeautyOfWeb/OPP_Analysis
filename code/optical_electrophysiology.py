import os, functools, subprocess, json, time, shutil, re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas
from sklearn.cluster import KMeans
from skimage.measure import regionprops
from skimage.color import label2rgb

import torch
import torch.nn as nn

from utility import linear_regression, power_series, empty_cache, get_label_image, neighbor_cor, svd, get_local_median, get_cor_map, get_cor_map_4d
from utility import empty_cache
from models import get_bg_mat, UNet
from visualization import plot_tensor, plot_image_label_overlay, imshow, plot_hist, plot_curves, plot_singular_values, save_gif_file, get_good_colors
from train import step_decompose
from denoise import get_denoised_mat
from segmentation import split_clusters


def get_size_from_txt(filepath):
    meta_data = pandas.read_csv(filepath, sep='\t', header=None, index_col=0)
    size = int(meta_data.loc['frames']), int(meta_data.loc['ywidth']), int(meta_data.loc['xwidth'])
    return size

def load_file(filepath, size=-1, dtype=np.uint16, astype='float32', device=torch.device('cuda')):
    array = np.fromfile(filepath, dtype=np.uint16).reshape(size).astype(astype)
    mat = torch.from_numpy(array).to(device)
    return mat

def load_mat(exp_id, meta_data, folder, astype='float32', device=torch.device('cuda')):
    file_meta = meta_data[exp_id]
    ncol, nrow, L = file_meta['xwidth'], file_meta['ywidth'], file_meta['frames']
    mat = load_file(filepath=os.path.join(folder, exp_id + '.bin'), size=[L, nrow, ncol], dtype=np.uint16, astype=astype, device=device)
    return mat

def plot_mean_intensity(mat, detrended=None, plot_detrended=False, plot_segments=False, num_frames=3000, period=500, signal_length=100, 
                        figsize=(20, 10)):
    array = mat.mean(-1).mean(-1).cpu()
    if detrended is not None and plot_detrended:
        array = array - array.min()
    xs = [array]
    colors = ['b']
    labels = ['mean intensity']
    if detrended is not None:
        detrended = detrended.mean(-1).mean(-1).cpu()
        if plot_detrended:
            detrended = detrended - detrended.min()
        xs.append(array - detrended)
        colors.append('k')
        labels.append('trend')
        if plot_detrended:
            xs.append(detrended)
            colors.append('g')
            labels.append('detrended')
    plot_curves(xs, colors, labels, show=False, marker='o', linestyle='--', linewidth=1, markersize=1)
    for i in range(0, num_frames, period):
        plt.axvline(x=i, color='r', linestyle='-.')
        plt.axvline(x=i+signal_length, color='r', linestyle='-.')
        plt.axvline(x=i, color='g', linestyle='--')
        plt.axvline(x=i+period, color='g', linestyle='--')
    plt.show()
    if plot_segments:
        for i in range(0, num_frames, period):
            plot_curves([x[i:i+period] for x in xs], colors, labels, show=False, marker='o', linestyle='--', linewidth=1, markersize=1)
            plt.axvline(x=0, color='r', linestyle='-.')
            plt.axvline(x=signal_length, color='r', linestyle='-.')
            plt.title(f'segment {i//period}')
            plt.show()

def detrend_linear(mat, train_idx=None, linear_order=3, input_min=-2, input_max=2, return_trend=False,
                   input_transformation=None, device=torch.device('cuda')):
    input_aug = torch.linspace(input_min, input_max, mat.shape[0], device=device)
    if input_transformation is not None:
        input_aug = input_transformation(input_aug)
    if train_idx is None:
        train_idx = range(mat.shape[0])
    beta, trend = linear_regression(X=input_aug[train_idx], Y=mat.reshape(mat.shape[0], -1)[train_idx], order=linear_order,
                                    X_test=input_aug)
    trend = trend.reshape(mat.shape)
    mat_adj = mat - trend
    if return_trend:
        for k in [k for k in locals().keys() if k not in ['mat_adj', 'trend']]:
            del locals()[k]
        torch.cuda.empty_cache()
        return mat_adj, trend
    else:
        for k in [k for k in locals().keys() if k not in ['mat_adj']]:
            del locals()[k]
        torch.cuda.empty_cache()
        return mat_adj

def detrend(mat, start0, end0, train_size_left, train_size_right, linear_order=3, use_mean_bg=False, plot=False, test_left=None, test_right=None, 
            device=torch.device('cuda'), exp_id=None, meta_data=None, folder=None, show_singular_values=False, **kwargs):
    if mat is None:
        mat = load_mat(exp_id, meta_data, folder) # from **kwargs
    _, nrow, ncol = mat.shape
    mat_list = [mat[s:e] for s, e in zip(start0, end0)]
    length0 = end0[0] - start0[0]
    input_aug = torch.linspace(-2, 2, length0, device=device)
    x_train = torch.cat([input_aug[:train_size_left], input_aug[-train_size_right:]])
    beta_left = linear_regression(x_train, order=linear_order)
    
    mat_adj = []
    if use_mean_bg:
        kernel_size = 7 # odd number (>=3) to maintain the same shape
        bg_mat = get_bg_mat(mat, kernel_size)

    for i in range(len(start0)): 
        # Pixel-level detrending
        y_bg = bg_mat if use_mean_bg else mat
        y_bg = y_bg[start0[i]:end0[i]].reshape(length0, -1)
        y_train = torch.cat([y_bg[:train_size_left], y_bg[-train_size_right:]])
        y_train_mean = y_train.mean()
        y_train_std = y_train.std()
        y_train = (y_train - y_train_mean) / y_train_std
        beta = torch.matmul(beta_left, y_train)
        y_pred = torch.matmul(power_series(input_aug, order=linear_order), beta)
        y_pred = y_pred * y_train_std + y_train_mean
        y_true = mat[start0[i]:end0[i]].reshape(length0, -1)
        y_adj = y_true - y_pred
        mat_adj.append(y_adj.reshape(length0, nrow, ncol))

        if plot:
            if show_singular_values:
                plot_singular_values(y_true, marker='o--', linewidth=1, markersize=2, use_cpu=True, end=20, show=True,
                                     title=f'Segment {i+1} singular values BEFORE detrending')
                plot_singular_values(y_adj, marker='o--', linewidth=1, markersize=2, use_cpu=True, end=20, color='orange',
                                     title=f'Segment {i+1} singular values AFTER detrending')
            plt.scatter(range(length0), y_true.mean(1).cpu(), marker='o', 
                        c=['r']*train_size_left + ['b']*(length0-train_size_left-train_size_right) + ['r']*train_size_right, 
                        s=1, alpha=0.5)
            plot_tensor(y_pred.mean(1), marker='g--', alpha=0.5)
            plt.axvline(test_left, color='g', linestyle='-.')
            plt.axvline(test_right, color='g', linestyle='-.')
            plt.title(f'Segment {i+1}')
            plt.show()
            plot_tensor(y_adj.mean(1), marker='o-', markersize=1, alpha=0.8)
            plt.axvline(test_left, color='g', linestyle='-.')
            plt.axvline(test_right, color='g', linestyle='-.')
            plt.title(f'Detrended segment {i+1}: min={y_adj.min().item():.0f} mean={y_adj.mean().item():.0f} max={y_adj.max().item():.0f}')
            plt.show()
    return mat_adj

def detrend_high_magnification(mat, skip_segments=1, num_segments=6, period=500, train_size_left=0, train_size_right=350, 
                               linear_order=3, plot=False, signal_start=0, signal_end=100, filepath=None, size=(-1, 180, 300), 
                               device=torch.device('cuda'), start0=None, end0=None, return_mat=False, **kwargs):
    if mat is None:
        mat = load_file(filepath=filepath, size=size, dtype=np.uint16, device=device)
    L, nrow, ncol = mat.size()
    if period == 'unknown':
        period = L
    if signal_end == 'period':
        signal_end = period
    train_idx = ([range(skip_segments*period)] +
                 [range(i*period, train_size_left+i*period) for i in range(skip_segments, num_segments)] + 
                 [range((i+1)*period - train_size_right, (i+1)*period) for i in range(skip_segments, num_segments)])
    train_idx = functools.reduce(lambda x,y: list(x)+list(y), train_idx)
    input_aug = torch.linspace(-2, 2, L, device=device)
    beta, trend = linear_regression(X=input_aug[train_idx], Y=mat.reshape(L, -1)[train_idx], order=linear_order, X_test=input_aug)
    mat_adj = mat - trend.reshape(L, nrow, ncol)
    if plot:
        frame_mean = mat.mean(-1).mean(-1)
        plt.figure(figsize=(20, 10))
        plt.plot(frame_mean.cpu(), 'o--', linewidth=1, markersize=2)
        for i in range(num_segments):
            plt.axvline(i*period+signal_start, color='g', linestyle='-.')
            plt.axvline(i*period+signal_end, color='g', linestyle='-.')
        plt.title('Frame mean intensity')
        plt.show()
        imshow(mat.mean(0), title='Mean intensity')
        imshow(trend.mean(0).reshape(nrow, ncol), title='Trend')
        imshow(mat_adj.mean(0).reshape(nrow, ncol), title='Detrended')

        cor = neighbor_cor(mat, neighbors=8, plot=True, choice='max', title='cor mat')
        plot_hist(cor, show=True)
        cor = neighbor_cor(trend.reshape(-1, nrow, ncol), neighbors=8, plot=True, choice='max', title='cor trend')
        plot_hist(cor, show=True)
        cor = neighbor_cor(mat_adj.reshape(-1, nrow, ncol), neighbors=8, plot=True, choice='max', title='cor mat_adj')
        plot_hist(cor, show=True)
    
        plot_mean_intensity(mat, detrended=mat_adj, plot_detrended=True, plot_segments=True, num_frames=L, period=period, 
                            signal_length=signal_end-signal_start)
    if start0 is not None and end0 is not None:
        mat_adj = [mat_adj[s:e] for s, e in zip(start0, end0)]
    if return_mat:
        return mat, mat_adj
    else:
        return mat_adj


def extract_super_pixels(mat_adj=None, test_left=None, test_right=None, mat_cat=None, num_neighbors=8, cor_choice='mean', connectivity=None, 
                         min_pixels=50, image=None, plot=False, use_mean_image=False):
    if image is None:
        if mat_cat is None:
            mat_cat = torch.cat([m[test_left:test_right] for m in mat_adj], dim=0)
        cor_global = neighbor_cor(mat_cat, neighbors=num_neighbors, choice=cor_choice, plot=plot, 
                                  title='correlation map')
        if use_mean_image:
            cor_global = mat_cat.mean(0) * cor_global
            cor_global = cor_global / cor_global.max()
        image = cor_global.detach().cpu().numpy()
    else:
        cor_global = image # for backward compatibility
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    label_image, regions = get_label_image(image, min_pixels=min_pixels, connectivity=connectivity, plot=False)
    if plot:
        plot_image_label_overlay(image, label_image)
    return cor_global, label_image, regions

def get_percentile(a, percentile):
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    return np.percentile(a.reshape(-1), q=percentile)
    
def extract_single_trace(mat, label_mask, percentile=50):
    binary_mask = label_mask > 0
    if percentile > 0:
        binary_mask = label_mask > get_percentile(label_mask[binary_mask], percentile)
    trace = (mat*label_mask*binary_mask).sum(-1).sum(-1) / label_mask[binary_mask].sum()
    return trace

def extract_traces(mat, softmask, label_image, regions=None, percentile=50, median_detrend=False):
    """
    Args:
        label_image: background: 0, labels: 1, 2, 3, ... (no skipping)
    """
    # assert len(np.unique(label_image)) == label_image.max()
    if mat.dtype == torch.float16:
        mat = mat.float()
    if regions is None:
        if isinstance(label_image, torch.Tensor):
            label_image_ = label_image.detach().cpu().numpy()
        else:
            label_image_ = label_image
        regions = regionprops(label_image_)
    if not isinstance(label_image, torch.Tensor):
        label_image = torch.from_numpy(label_image).to(mat.device)
    submats = []
    traces = []
    for i, region in enumerate(regions):
        minr, minc, maxr, maxc = region.bbox
        submat = mat[:, minr:maxr, minc:maxc]
        sub_image = label_image[minr:maxr, minc:maxc]
        label_mask = softmask[minr:maxr, minc:maxc].clone()
        label_mask[sub_image!=i+1] = 0
        trace = extract_single_trace(submat, label_mask, percentile=percentile)
        submats.append(submat)
        traces.append(trace)
    for k in [k for k in locals().keys() if k not in ['submats', 'traces']]:
        del locals()[k]
    if len(traces) > 0:
        traces = torch.stack(traces, dim=0)
    torch.cuda.empty_cache()
    if median_detrend:
        traces = traces - get_local_median(traces, window_size=50, dim=1)
    return submats, traces

def get_submat_traces(regions, label_image, seg_idx=0, mat_adj=None, sig_list=None, mat_list=None, mat=None, cor=None,
                      weighted_denominator=True, 
                      weight_percentile=50, return_name='all', linear_order=3, input_aug=None, beta_left=None, train_size_left=None, 
                      train_size_right=None, compare=False, test_left=None, test_right=None, plot_singular_values=False, 
                      device=torch.device('cuda'), **kwargs):
    """Use four different methods to calculate traces
    
    'mat_adj': use pre-calculated detrended matrices with linear regression
    'mean_bg': use the mean background values to detrend
    'y_adj': use the background to detrend with linear regression
    'sig_list': use the original values without detrending
    
    
    """
    def get_trace(mat=None, mat_adj=None, seg_idx=None, submat=None, weight_percentile=weight_percentile, 
                  plot_singular_values=plot_singular_values):
        if submat is None:
            if mat is None:
                mat = mat_adj[seg_idx]
            submat = mat[:, minr:maxr, minc:maxc]
        if cor is None:
            cor_ = neighbor_cor(submat, neighbors=8, plot=False, choice='max', title='cor', return_adj_list=False)
        else:
            cor_ = cor[minr:maxr, minc:maxc]
        if weight_percentile is None:
            weight = cor_
        else:
            weight = nn.functional.threshold(cor_, np.percentile(cor_[label_mask.bool()].cpu(), weight_percentile), 0, inplace=False)
        if weight.sum() == 0:
            weight = 1
        denominator = (label_mask*weight).sum() if weighted_denominator else label_mask.sum()
        trace = ((submat * label_mask * weight).sum(-1).sum(-1) / denominator)
        if plot_singular_values:
            u, s, v = torch.svd(submat.reshape(len(submat), -1))
            plot_tensor(s, marker='o--', linewidth=1, figsize=(20, 10), show=True)
        return submat, trace, weight
        
    is_grad_enabled = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    traces = {'mat_adj': [], 'sig_list': [], 'mean_bg': [], 'y_adj': []}
    submats = {'mat_adj': [], 'sig_list': [], 'mean_bg': [], 'y_adj': []}
    if mat is not None:
        traces['mat'] = []
        submats['mat'] = []
    num_labels = len(regions)
    if isinstance(label_image, np.ndarray):
        label_image = torch.from_numpy(label_image).to(device=device)
    nrow, ncol = label_image.shape
    
    for label_idx in range(1, num_labels+1):
        minr, minc, maxr, maxc = regions[label_idx-1].bbox
        label_mask = (label_image==label_idx).float()[minr:maxr, minc:maxc] # there can be other labels
        bg_mask = (label_image==0).float()[minr:maxr, minc:maxc]
        if mat is not None:
            submat, trace, weight = get_trace(mat=mat)
            submats['mat'].append([submat, label_mask, weight])
            traces['mat'].append(trace)
        else:
            if mat_adj is not None and return_name == 'all' or return_name == 'mat_adj':
                if seg_idx is None:
                    # this will consume a lot of gpu memory; never used
                    submat =  torch.cat([m[test_left:test_right] for m in mat_adj], dim=0)[:, minr:maxr, minc:maxc]
                    submat, trace, weight = get_trace(submat=submat)
                else:
                    submat, trace, weight = get_trace(mat_adj=mat_adj, seg_idx=seg_idx)
                submats['mat_adj'].append([submat, label_mask, weight])
                if compare:
                    traces['mat_adj'].append(trace[test_left:test_right])
                else:
                    traces['mat_adj'].append(trace)

            if sig_list is not None and return_name == 'all' or return_name == 'sig_list':
                if seg_idx is None:
                    # this will consume a lot of gpu memory; never used
                    submat =  torch.cat(sig_list, dim=0)[:, minr:maxr, minc:maxc]
                    submat, trace, weight = get_trace(submat=submat)
                else:
                    submat, trace, weight = get_trace(mat_adj=sig_list, seg_idx=seg_idx)
                submats['sig_list'].append([submat, label_mask, weight])
                traces['sig_list'].append(trace)

                if return_name == 'all' or return_name == 'mean_bg':
                    y_bg = (submat * bg_mask).sum(-1).sum(-1) / bg_mask.sum()
                    submat = submat - y_bg[:, None, None]
                    submat, trace, weight = get_trace(submat=submat)
                    submats['mean_bg'].append([submat, label_mask, weight])
                    traces['mean_bg'].append(trace)
            
            if mat_list is not None and return_name == 'all' or return_name == 'y_adj':
                if label_idx == 1:
                    if input_aug is None:
                        input_aug = torch.linspace(-2, 2, len(mat_list[seg_idx]), device=device)
                    if beta_left is None:
                        x_train = torch.cat([input_aug[:train_size_left], input_aug[-train_size_right:]])
                        beta_left = linear_regression(x_train, order=linear_order)
                if seg_idx is None:
                    submat =  torch.cat([m[test_left:test_right] for m in mat_list], dim=0)[:, minr:maxr, minc:maxc]
                else:
                    submat = mat_list[seg_idx][:, minr:maxr, minc:maxc]
                y_true = submat.reshape(len(submat), -1)
                y_bg = (submat * bg_mask).reshape(len(submat), (maxr-minr)*(maxc-minc)).sum(1, keepdim=True) / bg_mask.sum()
                y_train = torch.cat([y_bg[:train_size_left], y_bg[-train_size_right:]])
                y_train_mean = y_train.mean()
                y_train_std = y_train.std()
                y_train = (y_train - y_train_mean) / y_train_std
                beta = torch.matmul(beta_left, y_train)
                y_pred = torch.matmul(power_series(input_aug, order=linear_order), beta)
                y_pred = y_pred * y_train_std + y_train_mean
                y_adj = y_true - y_pred
                submat = y_adj.reshape(len(y_adj), maxr-minr, maxc-minc)
                submat, trace, weight = get_trace(submat=submat)
                submats['y_adj'].append([submat, label_mask, weight])
                if compare:
                    traces['y_adj'].append(trace[test_left:test_right])
                else:
                    traces['y_adj'].append(trace)
    torch.cuda.empty_cache()
    torch.set_grad_enabled(is_grad_enabled)
    if return_name == 'all':
        return traces, submats
    else:
        return traces[return_name], submats[return_name]

    
def extract_one_label_data(submats, label_idx):
    submat, label_mask, weight = submats[label_idx]
    X = torch.log1p(submat - submat.min())
    x = X.unsqueeze(0).unsqueeze(0)
    y = x * label_mask
    return x, y, label_mask, weight


def prep_train_data(seg_idx, label_idx, label_image, regions, sig_list=None, mat_list=None, mat_adj=None, cor=None, 
                    return_name='mat_adj'):
    traces, submats = get_submat_traces(seg_idx=seg_idx, regions=regions, label_image=label_image, mat_adj=mat_adj, sig_list=sig_list, 
                                        mat_list=mat_list, cor=cor, weighted_denominator=True, return_name=return_name, compare=False)
    x, y, label_mask, weight = extract_one_label_data(submats, label_idx=label_idx)
    trace = traces[label_idx]
    return x, y, trace, label_mask, weight


def denoise_trace(trace, model=None, filepath='/home/jupyter/notebooks/checkpoints/denoise_trace.pt', return_detached=True, 
                  device=torch.device('cuda')):
    if model is None:
        model = UNet(in_channels=1, num_classes=1, out_channels=[8, 16, 32], num_conv=2, 
                     n_dim=1, kernel_size=3).to(device)
        model.load_state_dict(torch.load(filepath))
    with torch.no_grad():
        mean = trace.mean()
        std = trace.std()
        pred = model((trace-mean)/std)
        pred = model(pred)
        pred = pred * std + mean
    if return_detached:
        pred = pred.detach()
    for k in [k for k in locals().keys() if k!='pred']:
        del locals()[k]
    torch.cuda.empty_cache()
    return pred

def denoise_3d(mat, model=None, filepath='/home/jupyter/notebooks/checkpoints/3d_denoise.pt', return_detached=True, 
               batch_size=5000, device=torch.device('cuda')):
    if model is None:
        model = UNet(in_channels=1, num_classes=1, out_channels=[4, 8, 16], num_conv=2, n_dim=3, 
                     kernel_size=[3, 3, 3], same_shape=True).to(device)
        model.load_state_dict(torch.load(filepath))
    with torch.no_grad():
        num_batches = (mat.size(0) + batch_size - 1)//batch_size
        mat = torch.cat([model(mat[batch_size*i:batch_size*(i+1)]) for i in range(num_batches)], dim=0)
    if return_detached:
        mat = mat.detach()
    for k in [k for k in locals().keys() if k!='mat']:
        del locals()[k]
    torch.cuda.empty_cache()
    return mat

def attention_map(mat, model=None, filepath='/home/jupyter/notebooks/checkpoints/segmentation_count_hardmask.pt', 
                  batch_size=5000, return_detached=True, device=torch.device('cuda')):
    if model is None:
        model = UNet(in_channels=1, num_classes=1, out_channels=[4, 8, 16], num_conv=2, n_dim=3, 
                     kernel_size=[3, 3, 3], same_shape=True).to(device)
        model.load_state_dict(torch.load(filepath))
    nrow, ncol = mat.shape[1:]
    if batch_size*nrow*ncol > 1e7:
        batch_size = int(1e7 / (nrow*ncol))
    with torch.no_grad():
        num_batches = (mat.size(0) + batch_size - 1)//batch_size
        mat = torch.cat([model(mat[batch_size*i:batch_size*(i+1)]) for i in range(num_batches)], dim=0).mean(0)
    if return_detached:
        mat = mat.detach()
    for k in [k for k in locals().keys() if k!='mat']:
        del locals()[k]
    torch.cuda.empty_cache()
    return mat

def refine_one_label(submat, min_pixels=50, return_traces=False, percentile=50):
    soft_attention = attention_map(submat)
    label_image, regions = get_label_image(soft_attention, min_pixels=min_pixels)
    if return_traces:
        submats, traces = extract_traces(submat, softmask=soft_attention, label_image=label_image, regions=regions, 
                                         percentile=percentile)
        return submats, traces, soft_attention, label_image, regions
    else:
        return label_image
    
def refine_segmentation(submats, regions, label_image, min_pixels=50, min_pixels_super=900, connectivity=None):
    for label_idx in range(1, len(submats)+1):
        if (label_image==label_idx).sum() >= min_pixels_super:
            submat = submats[label_idx-1]
            minr, minc, maxr, maxc = regions[label_idx-1].bbox
            img = refine_one_label(submat, min_pixels=min_pixels)
            label_image[minr:maxr, minc:maxc] = img
    from skimage.measure import label, regionprops
    label_image = label(label_image>0, connectivity=connectivity)
    regions = regionprops(label_image)
    return label_image, regions


def basic_segmentation(mat, min_thresh=0.05, min_pixels=50, select_frames=True, show=True, median_detrend=False, 
                       fft=False, fft_max_freq=200):
    """Basic segmentation 
    Args:
        mat: torch.Tensor with shape (nframe, nrow, ncol) or (n_experiments, nframe, nrow, ncol)
        min_thresh: float, used by get_label_image
        min_pixels: int, used by get_label_image
        select_frames: default True, only used when mat.ndim==4, selecting only frames with average mean beyond an otsu threshold
        
    Returns:
        cor_map: torch.Tensor with shape (nrow, ncol)
        label_image: torch.Tensor with shape (nrow, ncol); 0 is background; label==i is mask for label i for i >= 1
        regions: object returned by regionprops
        
    """
    dtype = mat.dtype
    if dtype == torch.float16:
        mat = mat.float()
    if median_detrend:
        mat = mat - get_local_median(mat, window_size=50, dim=-3)
    if fft:
        if mat.ndim == 3:
            mat = torch.rfft(mat.transpose(0, 2), signal_ndim=1, normalized=True)[..., :fft_max_freq, :].reshape(
                mat.size(2), mat.size(1), -1).transpose(0, 2)
        elif mat.ndim == 4:
            mat = torch.rfft(mat.transpose(1, 3), signal_ndim=1, normalized=True)[..., :fft_max_freq, :].reshape(
                mat.size(0), mat.size(3), mat.size(2), -1).transpose(1, 3)
    if mat.ndim == 3:
        cor_map = get_cor_map(mat)
    elif mat.ndim == 4:
        cor_map = get_cor_map_4d(mat, select_frames=select_frames, top_cor_map_percentage=20, padding=2, topk=5, shift_times=[0, 1, 2], 
                                 return_all=False, plot=False)
    label_image, regions = get_label_image(cor_map, min_thresh=min_thresh, min_pixels=min_pixels)
    label_image = torch.from_numpy(label_image).to(mat.device)
    if show:
        imshow(cor_map)
        plot_image_label_overlay(cor_map, label_image=label_image, regions=regions)
    return cor_map, label_image, regions

def parse_meta_data(meta_file):
    """Parse the meta data json file generated by Trinh as of August 6th
    Change this function if the meta data json file is modified later
    
    Args:
        meta_file: local meta data json file path
    
    Returns:
        meta_data: a dictionary with at least the following keys:
            'numFramesRequested': int, number of frames
            'movSize': (int, int), (width, height)
            'blueFrameOnOff': a list of ints specifying when the blue light is turned on and off: [On, Off, On, Off, On, Off, ...]
            
    Notes:
        This is called in entire_pipeline; if the return of this function changes, make sure to change the corresponding part of entire_pipeline
    """
#     command = ['sed', '-i', 's/Null/null/g', meta_file] # substitute Null with null
#     response = subprocess.run(command, capture_output=True)
#     assert response.returncode == 0
#     command = ['sed', '-i', '/:/!d', meta_file] # remove lines starting without ':'
#     response = subprocess.run(command, capture_output=True)
#     assert response.returncode == 0
#     command = ['sed', '-i', '1i {', meta_file] # add first line '{' 
#     response = subprocess.run(command, capture_output=True)
#     assert response.returncode == 0
#     command = ['sed', '-i', '-e', '$a}', meta_file] # add last line '}' 
#     response = subprocess.run(command, capture_output=True)
#     assert response.returncode == 0
    
    try:
        with open(meta_file, 'r') as f:
                meta_data = json.load(f)
    except ValueError:
        # Trinh's script to generate metadata json file has bugs; these lines are to required to handle it
        with open(meta_file, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            lines = [re.sub('Null', 'null', line) for line in lines if line!=',']
            lines = functools.reduce(lambda x, y: x+y, lines)
            meta_data = json.loads(lines)
    assert 'numFramesRequested' in meta_data and 'movSize' in meta_data and 'blueFrameOnOff' in meta_data
    return meta_data

def prepare_data(bucket, bin_files=None, data_folder_prefix='.', result_folder='results', verbose=False):
    """Prepare meta data and create save folders
    
    Args:
        bucket: google bucket folder to be processed, e.g., gs://broad-opp-voltage/folder_name
        bin_files: default None, process all the .bin files in the bucket; otherwise only process file(s) specified by bin_files
        data_folder_prefix: default '.', i.e., current folder; default data folder name will be extracted from bucket
        result_folder: default 'results'
        
    Returns:
        bin_files: a list of .bin files to be processed
        meta_data: a dictionary containing meta data for each file
        data_folder: folder created to save temporary data and results
        
    """
    command = ['gsutil', 'ls', bucket]
    response = subprocess.run(command, capture_output=True)
    assert response.returncode == 0
    filepaths = response.stdout.decode().split()
    if bin_files is not None:
        if isinstance(bin_files, str):
            bin_files = [bin_files]
    else:
        bin_files = sorted([f.split('.')[0].split('/')[-1] for f in filepaths if re.search('.bin$', f)])

    data_folder = bucket.split('/')[-1]
    data_folder = f'{data_folder_prefix}/{data_folder}'
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    if not os.path.exists(f'{data_folder}/{result_folder}'):
        if verbose:
            print(f'Create folder {data_folder}/{result_folder}')
        os.makedirs(f'{data_folder}/{result_folder}')
    if not os.path.exists(f'{data_folder}/json'):
        if verbose:
            print(f'Create folder {data_folder}/json')
        os.makedirs(f'{data_folder}/json')
        command = ['gsutil', '-m', 'cp', f'{bucket}/*.json', f'{data_folder}/json']
        response = subprocess.run(command, capture_output=True)
        assert response.returncode == 0

    meta_data = {}
    for file in bin_files:
        meta_file = f'{data_folder}/json/{file}_metadata.json'
        if not os.path.exists(meta_file):
            command = ['gsutil', 'cp', f'{bucket}/{file}_metadata.json', meta_file]
            response = subprocess.run(command, capture_output=True)
            if response.returncode != 0:
                print(f'{meta_file} does not exist!')
                continue
        meta_data[file] = parse_meta_data(meta_file)
    bin_files = [k for k in bin_files if k in meta_data]
    return bin_files, meta_data, data_folder


def entire_pipeline(bucket, result_folder='results', bin_files=None, delete_local_data=True, 
                    apply_spectral_clustering=False, spectral_soft_threshold=True, spectral_cor_threshold=None,
                    denoise=False, denoise_model_config=None, denoise_loss_threshold=0, denoise_num_epochs=12, denoise_num_iters=600, 
                    denoise_batch_size=2, denoise_batch_size_eval=4,
                    display=False, verbose=False, half_precision=False, device=torch.device('cuda')):
    """Entire pipeline to process OPP voltage imaging data
    Args:
        bucket: google bucket folder containing .bin files and metadata .json files
        result_folder: default 'results'
        bin_files: default None, process all the .bin files with metadata in the bucket; 
            otherwise only process those specified in bin_files; bin_files can be a list of .bin files or a single .bin file
            note .bin suffix should not be included as filename
        delete_local_data: if True, will delete all intermediate data on the local disk
        apply_spectral_clustering: if True, apply spectral clustering to get fine-grained segmentation
        spectral_soft_threshold: if True, automatically determine when to split and when to stop
        spectral_cor_threshold: hard threshold between 0 and 1 used to determine when to split a cluster; 
            only used when spectral_soft_threshold is False
        denoise: if True, run denoise pipeline; currently, often encounter GPU memory errors when running on multiple .bin files
        denoise_model_config: default None; if not None, provide a json file path
        denoise_loss_threshold: default 0, if bigger than 0, then training will automatically stop after loss is below this threshold
        denoise_num_epochs: num of epochs to train the denoising model
        denosie_num_iters: num of iterations every epoch
        denoise_batch_size: default 2, can be increased to a larger integer if GPU memory is sufficient
        denoise_batch_size_eval: default 4
    
    Returns:
        After the call, the results will be uploaded to the cloud automatically
    """
    def save_results(mat, softmask, label_image, regions, save_folder, file, segmentation_name='basic_segmentation', display=False, verbose=False):
        submats, traces = extract_traces(mat, softmask=softmask, label_image=label_image, regions=regions, median_detrend=False)
        plot_image_label_overlay(softmask, label_image=label_image, regions=regions, save_file=f'{save_folder}/label_image__{segmentation_name}.png', 
                                 title=f'{file}: {label_image.max()} neurons detected', display=display)
        if len(traces) > 0:
            if verbose:
                print(f'{len(traces)} neuron detected in {file}.bin')
            np.save(f'{save_folder}/label_image__{segmentation_name}.npy', label_image.cpu().numpy())
            np.save(f'{save_folder}/traces__{segmentation_name}.npy', traces.cpu().numpy())
    
    def save_figures(segmentation_name='basic_segmentation', figsize=(20, 20), bounding_box=True):
        for file in bin_files:
            save_folder = f'{data_folder}/{result_folder}/{file}'
            if os.path.exists(f'{save_folder}/label_image__{segmentation_name}.npy'):
                cor_map = np.load(f'{save_folder}/cor_map.npy')
                label_image = np.load(f'{save_folder}/label_image__{segmentation_name}.npy')
                traces = np.load(f'{save_folder}/traces__{segmentation_name}.npy')
                image_label_overlay = label2rgb(label_image, image=cor_map)
                regions = regionprops(label_image)
                fig_folder = f'{save_folder}/figs/{segmentation_name}'
                if not os.path.exists(fig_folder):
                    os.makedirs(fig_folder)
                for sel_idx in range(label_image.max()):
                    fig, ax = plt.subplots(2, figsize=figsize)
                    ax[0].imshow(image_label_overlay)
                    if bounding_box:
                        region = regions[sel_idx]
                        minr, minc, maxr, maxc = region.bbox
                        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                                  fill=False, edgecolor='red', linewidth=2)
                        ax[0].add_patch(rect)
                        ax[0].text(minc-3, minr-1, sel_idx+1, color='r')
                    ax[0].set_axis_off()
                    ax[0].set_title(f'Neuron {sel_idx+1} segmentation')
                    ax[1].plot(traces[sel_idx])
                    ax[1].set_title(f'Neuron {sel_idx+1} trace')
                    plt.tight_layout()
                    plt.savefig(f'{fig_folder}/{sel_idx+1}.png')
                    plt.close()
                imgs = [f'{fig_folder}/{i+1}.png' for i in range(label_image.max())]
                save_gif_file(imgs, save_path=f'{fig_folder}/{label_image.max()}_neurons.gif')
                fig, ax = plt.subplots(figsize=figsize)
                for i in range(len(traces)):
                    ax.plot(traces[i], label=i+1, c=good_colors[i%len(good_colors)])
                    ax.text(-len(traces[i])*0.02, traces[i, :10].mean(), i+1, c=good_colors[i%len(good_colors)])
                plt.legend()
                plt.savefig(f'{fig_folder}/{label_image.max()}_traces.png')
                plt.close()
    
    bin_files, meta_data, data_folder = prepare_data(bucket=bucket, bin_files=bin_files, result_folder=result_folder, 
                                                     data_folder_prefix='.', verbose=verbose)
    
    good_colors = get_good_colors()
    
    if denoise:
        if denoise_model_config is not None and os.path.exists(denoise_model_config):
            with open(denoise_model_config, 'r') as f:
                denoise_model_config = json.load(f)
        else:
            denoise_model_config = {}
        denoise_features = denoise_model_config.pop('features', True)
        denoise_optimizer_fn_args = denoise_model_config.pop('optimizer_fn_args', {'lr': 1e-3, 'weight_decay': 1e-2})
        denoise_lr_scheduler = denoise_model_config.pop('denoise_lr_scheduler', None)
        #Todo: put all arguments into denoise_model_config
        denoise_model = None # will be updated for each file
        
    for file in bin_files:
        print(f'Process {file}')
        start_time = time.time()
        # download .bin file if not exists
        if not os.path.exists(f'{data_folder}/{file}.bin'):
            command = ['gsutil', '-m', 'cp', f'{bucket}/{file}.bin', data_folder]
            response = subprocess.run(command, capture_output=True)
            assert response.returncode == 0
        # create save folder if not exists
        save_folder = f'{data_folder}/{result_folder}/{file}'
        if not os.path.exists(save_folder):
            if verbose:
                print(f'Create folder {save_folder}')
            os.makedirs(save_folder)
        # load mat and meta data
        nframe = meta_data[file]['numFramesRequested']
        ncol, nrow = meta_data[file]['movSize']
        blue_light_on_off = np.array(meta_data[file]['blueFrameOnOff']).reshape(-1, 2)
        torch.cuda.empty_cache()
        mat = load_file(f'{data_folder}/{file}.bin', size=(nframe, nrow, ncol), astype='float16' if half_precision else 'float32', device=device)
        # without detrending, only use selected frames to calculate correlation
        submat = torch.stack([mat[i-1:j-1] for i, j in blue_light_on_off], dim=0)
        cor_map, label_image, regions = basic_segmentation(submat, min_pixels=20, select_frames=True, median_detrend=False, fft=False, show=False)
        # save correlation map
        imshow(cor_map, save_file=f'{save_folder}/cor_map.png', title=f'{file}: min_cor={cor_map.min():.2f}, max_cor={cor_map.max():.2f}', 
               display=display)
        np.save(f'{save_folder}/cor_map.npy', cor_map.cpu().numpy())
        # save basic segmentation results
        save_results(mat, segmentation_name='basic_segmentation', softmask=cor_map, label_image=label_image, regions=regions, save_folder=save_folder,
                     file=file, display=display, verbose=verbose)
        if denoise:
            # noise2self
            if verbose:
                print('Denoising')
                now = time.time()
            denoise_loss_history = []
            denoise_save_folder = denoise_model_config.pop('denoise_save_folder', f'{save_folder}/denoise')
            denoised_mat, denoise_model = get_denoised_mat(mat, features=denoise_features, model=denoise_model, save_folder=denoise_save_folder, 
                                                           loss_threshold=denoise_loss_threshold, loss_history=denoise_loss_history, verbose=verbose, 
                                                           optimizer_fn_args=denoise_optimizer_fn_args, lr_scheduler=denoise_lr_scheduler, 
                                                           out_channels=[64, 64, 128], kernel_size_unet=3, ndim=2, frame_depth=4,
                                                           last_out_channels=100, normalize=True, 
                                                           num_epochs=denoise_num_epochs, num_iters=denoise_num_iters, print_every=300, 
                                                           batch_size=denoise_batch_size, batch_size_eval=denoise_batch_size_eval, 
                                                           mask_prob=0.05, frame_weight=None, 
                                                           save_intermediate_results=False, movie_start_idx=250, movie_end_idx=750, fps=60,
                                                           loss_reg_fn=nn.MSELoss(), optimizer_fn=torch.optim.AdamW, 
                                                           window_size_row=None, window_size_col=None, weight=None, return_model=True, device=device)
            save_results(denoised_mat, segmentation_name='basic_segmentation_denoise', softmask=cor_map, label_image=label_image, regions=regions, 
                         save_folder=save_folder, file=file, display=display, verbose=verbose)
            if verbose:
                print(f'Denoising time: {time.time() - now:.2f} s')
        if apply_spectral_clustering:
            if verbose:
                print('Apply spectral clustering')
                now = time.time()
            sel_label_idx = 1
            while sel_label_idx <= label_image.max():
                if verbose:
                    print(sel_label_idx)
                split_clusters(sel_label_idx, mat, label_image, cor_threshold=spectral_cor_threshold, soft_threshold=spectral_soft_threshold, 
                               min_num_pixels=50, max_dist=2, median_detrend=True, apply_fft=True, fft_max_freq=200, verbose=verbose)
                sel_label_idx += 1
            regions = regionprops(label_image.cpu().numpy())
            save_results(mat, segmentation_name='spectral_clustering', softmask=cor_map, label_image=label_image, regions=regions, 
                         save_folder=save_folder, file=file, display=display, verbose=verbose)
            if verbose:
                print(f'Spectral clustering time: {time.time() - now:.2f} s')

        if verbose:
            print('Generating figures and uploading results to google bucket')
        save_figures(segmentation_name='basic_segmentation')
        save_figures(segmentation_name='basic_segmentation_denoise')
        save_figures(segmentation_name='spectral_clustering')
        save_figures(segmentation_name='denoise_spectral_clustering')
        command = ['gsutil', '-m', 'cp', '-r', 
                   save_folder, 
                   f'{bucket}/{result_folder}/{file}']
        response = subprocess.run(command, capture_output=True)
        assert response.returncode == 0
    
        if delete_local_data:
            os.remove(f'{data_folder}/{file}.bin')
            shutil.rmtree(f'{data_folder}/{result_folder}/{file}')
        empty_cache(lambda k, v: isinstance(v, torch.Tensor))
        torch.cuda.empty_cache()
        end_time = time.time()
        print(f'Time spent: {end_time - start_time}')
        
    if delete_local_data:
        shutil.rmtree(data_folder)