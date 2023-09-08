import numpy as np
import pandas as pd
import time as ttime
from iss_workflows.xray import *
import numexpr as ne

def get_transition_grid(dE_start, dE_end, E_range, round_up=True):
    if round_up:
        n = np.ceil(2 * E_range / (dE_start + dE_end))
    else:
        n = np.floor(2 * E_range / (dE_start + dE_end))
    delta = (E_range * 2 / n - 2 * dE_start) / (n - 1)
    steps = dE_start + np.arange(n) * delta
    # if not ascend:
    #     steps = steps[::-1]
    return np.cumsum(steps)


def xas_energy_grid(energy_range, e0, edge_start, edge_end, preedge_spacing, xanes_spacing, exafs_k_spacing,
                    E_range_before=15, E_range_after=20, n_before=10, n_after=20):
    # energy_range_lo= np.min(energy_range)
    # energy_range_hi = np.max(energy_range)
    energy_range_lo = np.min([e0 - 300, np.min(energy_range)])
    energy_range_hi = np.max([e0 + 2500, np.max(energy_range)])

    # preedge = np.arange(energy_range_lo, e0 + edge_start-1, preedge_spacing)
    preedge = np.arange(energy_range_lo, e0 + edge_start, preedge_spacing)

    # before_edge = np.arange(e0+edge_start,e0 + edge_start+7, 1)
    before_edge = preedge[-1] + get_transition_grid(preedge_spacing, xanes_spacing, E_range_before, round_up=False)

    edge = np.arange(before_edge[-1], e0 + edge_end - E_range_after, xanes_spacing)

    # after_edge = np.arange(e0 + edge_end - 7, e0 + edge_end, 0.7)

    eenergy = k2e(e2k(e0 + edge_end, e0), e0)
    post_edge = np.array([])

    while (eenergy < energy_range_hi):
        kenergy = e2k(eenergy, e0)
        kenergy += exafs_k_spacing
        eenergy = k2e(kenergy, e0)
        post_edge = np.append(post_edge, eenergy)

    after_edge = edge[-1] + get_transition_grid(xanes_spacing, post_edge[1] - post_edge[0], post_edge[0] - edge[-1],
                                                round_up=True)
    energy_grid = np.unique(np.concatenate((preedge, before_edge, edge, after_edge, post_edge)))
    energy_grid = energy_grid[(energy_grid >= np.min(energy_range)) & (energy_grid <= np.max(energy_range))]
    return energy_grid


def _generate_convolution_bin_matrix(sample_points, data_x):
    fwhm = _compute_window_width(sample_points)
    delta_en = _compute_window_width(data_x)

    mat = _generate_sampled_gauss_window(data_x.reshape(1, -1),
                                         fwhm.reshape(-1, 1),
                                         sample_points.reshape(-1, 1))
    mat *= delta_en.reshape(1, -1)
    mat /= np.sum(mat, axis=1)[:, None]
    return mat


_GAUSS_SIGMA_FACTOR = 1 / (2 * (2 * np.log(2)) ** .5)


def _generate_sampled_gauss_window(x, fwhm, x0):
    sigma = fwhm * _GAUSS_SIGMA_FACTOR
    a = 1 / (sigma * (2 * np.pi) ** .5)
    data_y = ne.evaluate('a * exp(-.5 * ((x - x0) / sigma) ** 2)')
    # data_y = np.exp(-.5 * ((x - x0) / sigma) ** 2)
    # data_y /= np.sum(data_y)
    return data_y


def _compute_window_width(sample_points):
    '''Given smaple points compute windows via approx 1D voronoi

    Parameters
    ----------
    sample_points : array
        Assumed to be monotonic

    Returns
    -------
    windows : array
        Average of distances to neighbors
    '''
    d = np.diff(sample_points)
    fw = (d[1:] + d[:-1]) / 2
    return np.concatenate((fw[0:1], fw, fw[-1:]))


def rebin(interpolated_dataset, e0, edge_start=-30, edge_end=50, preedge_spacing=5,
          xanes_spacing=-1, exafs_k_spacing=0.04, skip_binning=False):
    if skip_binning:
        binned_df = interpolated_dataset
        col = binned_df.pop("energy")
        n = len(binned_df.columns)
        binned_df.insert(n, col.name, col)
        binned_df = binned_df.sort_values('energy')
        convo_mat = None
    else:
        print(f'({ttime.ctime()}) Binning the data: BEGIN')
        if xanes_spacing == -1:
            if e0 < 14000:
                xanes_spacing = 0.2
            elif e0 >= 14000 and e0 < 21000:
                xanes_spacing = 0.3
            elif e0 >= 21000:
                xanes_spacing = 0.4
            else:
                xanes_spacing = 0.3

        interpolated_energy_grid = interpolated_dataset['energy'].values
        binned_energy_grid = xas_energy_grid(interpolated_energy_grid, e0, edge_start, edge_end,
                                             preedge_spacing, xanes_spacing, exafs_k_spacing)

        convo_mat = _generate_convolution_bin_matrix(binned_energy_grid, interpolated_energy_grid)
        ret = {'energy': binned_energy_grid}
        for k, v in interpolated_dataset.items():
            if k != 'energy':
                data_array = v.values
                if len(data_array[0].shape) == 0:
                    ret[k] = convo_mat @ data_array
                else:
                    data_ndarray = np.array([i for i in data_array], dtype=np.float64)
                    data_conv = np.tensordot(convo_mat, data_ndarray, axes=(1, 0))
                    ret[k] = [i for i in data_conv]

        binned_df = pd.DataFrame(ret)
    # binned_df = binned_df.drop('timestamp', axis=1)
    print(f'({ttime.ctime()}) Binning the data: DONE')
    return binned_df, convo_mat