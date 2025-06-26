from scipy.signal import resample, correlate
import numpy as np
import torch

INCLUDED_CHANNELS=['A1-T3',
 'C3-CZ',
 'C3-P3',
 'C4-P4',
 'C4-T4',
 'CZ-C4',
 'F3-C3',
 'F4-C4',
 'F7-T3',
 'F8-T4',
 'FP1-F3',
 'FP1-F7',
 'FP2-F4',
 'FP2-F8',
 'P3-O1',
 'P4-O2',
 'T3-C3',
 'T3-T5',
 'T4-A2',
 'T4-T6',
 'T5-O1',
 'T6-O2']

def comp_xcorr(x, y, mode="valid", normalize=True):
    """
    Compute cross-correlation between 2 1D signals x, y
    Args:
        x: 1D array or tensor
        y: 1D array or tensor
        mode: 'valid', 'full' or 'same'
        normalize: If True, will normalize cross-correlation
    Returns:
        xcorr: cross-correlation of x and y
    """

    # Convert PyTorch tensors to NumPy arrays
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    # Compute cross-correlation
    xcorr = correlate(x, y, mode=mode)

    if normalize:
        cxx0 = np.sum(np.abs(x) ** 2)
        cyy0 = np.sum(np.abs(y) ** 2)
        if cxx0 != 0 and cyy0 != 0:
            scale = np.sqrt(cxx0 * cyy0)
            xcorr /= scale

    return xcorr

######## Graph related data utils ########
def keep_topk(adj_mat, top_k=3, directed=True):
    """ "
    Helper function to sparsen the adjacency matrix by keeping top-k neighbors
    for each node.
    Args:
        adj_mat: adjacency matrix, shape (num_nodes, num_nodes)
        top_k: int
        directed: whether or not a directed graph
    Returns:
        adj_mat: sparse adjacency matrix, directed graph
    """
    # Set values that are not of top-k neighbors to 0:
    adj_mat_noSelfEdge = adj_mat.copy()
    for i in range(adj_mat_noSelfEdge.shape[0]):
        adj_mat_noSelfEdge[i, i] = 0

    top_k_idx = (-adj_mat_noSelfEdge).argsort(axis=-1)[:, :top_k]

    mask = np.eye(adj_mat.shape[0], dtype=bool)
    for i in range(0, top_k_idx.shape[0]):
        for j in range(0, top_k_idx.shape[1]):
            mask[i, top_k_idx[i, j]] = 1
            if not directed:
                mask[top_k_idx[i, j], i] = 1  # symmetric

    adj_mat = mask * adj_mat
    return adj_mat