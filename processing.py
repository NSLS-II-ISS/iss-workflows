from tiled.client import from_profile

import prefect
from prefect import task, Flow, Parameter

import time as ttime
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import xraydb
import numexpr as ne

tiled_client = from_profile("nsls2", username=None)["iss"]
tiled_client_iss = tiled_client["raw"]
tiled_client_sandbox = tiled_client["sandbox"]


# Used for Prefect logging.
# logger = prefect.context.get("logger")


def load_dataset_from_tiled(run, stream_name):
    t = run[stream_name]['data'][stream_name].read()
    arr = np.array(t.tolist()).squeeze()
    columns = list(t.dtype.fields.keys())
    return pd.DataFrame(arr, columns=columns)


def load_apb_dataset_from_tiled(run):
    apb_dataset = load_dataset_from_tiled(run, 'apb_stream')
    apb_dataset = apb_dataset[apb_dataset['timestamp'] > 1]
    ch_offsets = get_ch_properties(run.metadata['start'], 'ch',
                                   '_offset') * 1e3  # offsets are ib mV but the readings are in uV
    ch_gains = get_ch_properties(run.metadata['start'], 'ch', '_amp_gain')

    apb_dataset.iloc[:, 1:] -= ch_offsets
    apb_dataset.iloc[:, 1:] /= 1e6
    apb_dataset.iloc[:, 1:] /= (10 ** ch_gains)

    energy_dataset = load_dataset_from_tiled(run, 'pb9_enc1')
    energy_dataset = energy_dataset[energy_dataset['ts_s'] > 0]
    angle_offset = -float(run.metadata['start']['angle_offset'])

    return apb_dataset, energy_dataset, angle_offset


def get_ch_properties(hdr_start, start, end):
    ch_keys = [key for key in hdr_start.keys() if key.startswith(start) and key.endswith(end)]
    return np.array([hdr_start[key] for key in ch_keys])


def translate_apb_dataset(apb_dataset, energy_dataset, angle_offset, ):
    data_dict = {}
    for column in apb_dataset.columns:
        if column != 'timestamp':
            adc = pd.DataFrame()
            adc['timestamp'] = apb_dataset['timestamp']
            adc['adc'] = apb_dataset[column]

            data_dict[column] = adc

    energy = pd.DataFrame()
    energy['timestamp'] = energy_dataset['ts_s'] + 1e-9 * energy_dataset['ts_ns']
    enc = energy_dataset['encoder'].apply(lambda x: int(x) if int(x) <= 0 else -(int(x) ^ 0xffffff - 1))

    energy['encoder'] = encoder2energy(enc, 360000, angle_offset)

    data_dict['energy'] = energy
    return data_dict


def find_e0(md):
    e0 = -1
    if 'e0' in md['start']:
        e0 = float(md['start']['e0'])
    return e0


def k2e(k, E0):
    """
    Convert from k-space to energy in eV

    Parameters
    ----------
    k : float
        k value
    E0 : float
        Edge energy in eV

    Returns
    -------
    out : float
        Energy value

    See Also
    --------
    :func:`isstools.conversions.xray.e2k`
    """
    return ((1000 / (16.2009 ** 2)) * (k ** 2)) + E0


def e2k(E, E0):
    """
    Convert from energy in eV to k-space

    Parameters
    ----------
    E : float
        Current energy in eV
    E0 : float
        Edge energy in eV

    Returns
    -------
    out : float
        k-space value

    See Also
    --------
    :func:`isstools.conversions.xray.k2e`
    """
    return 16.2009 * (((E - E0) / 1000) ** 0.5)


def encoder2energy(encoder, pulses_per_deg, offset=0):
    """
    Convert from encoder counts to energy in eV

    Parameters
    ----------
    encoder : float or np.array()
        Encoder counts to convert
    pulses_per_deg: float
        Number of pulses per degree of the encoder
    offset : float
        Offset in degrees to adjust the conversion

    Returns
    -------
    out : float or np.array()
        Energy value or array of values

    See Also
    --------
    :func:`isstools.conversions.xray.energy2encoder`
    """
    return -12398.42 / (2 * 3.1356 * np.sin(np.deg2rad((encoder / pulses_per_deg) - float(offset))))


def energy2encoder(energy, pulses_per_deg, offset=0):
    """
    Convert from energy in eV to encoder counts

    Parameters
    ----------
    energy : float or np.array()
        Energy in eV to convert
    offset : float
        Offset in degrees to adjust the conversion

    Returns
    -------
    out : float or np.array()
        Encoder counts value or array of values

    See Also
    --------
    :func:`isstools.conversions.xray.encoder2energy`

    # This is how it's defined in the IOC
    record(calcout, "XF:08IDA-OP{Mono:HHM-Ax:E}Mtr-SP") {
    field(INPB, "-1977.004107667")
    field(INPC, "3.141592653")
    field(INPE, "XF:08IDA-OP{Mono:HHM-Ax:E}Offset.VAL PP MS")

    "ASIN(B/A)*180/C - E")
    """
    return pulses_per_deg * (np.degrees(np.arcsin(-12398.42 / (2 * 3.1356 * energy))) - float(offset))


def energy2angle(energy, offset=0):
    return np.degrees(np.arcsin(-12398.42 / (2 * 3.1356 * energy))) - float(offset)


def interpolate(dataset, key_base=None, sort=True):
    interpolated_dataset = {}
    min_timestamp = max([dataset.get(key).iloc[0, 0] for key in dataset])
    max_timestamp = min([dataset.get(key).iloc[len(dataset.get(key)) - 1, 0] for key in
                         dataset if len(dataset.get(key).iloc[:, 0]) > 5])
    if key_base is None:
        all_keys = []
        time_step = []
        for key in dataset.keys():
            all_keys.append(key)
            time_step.append(np.mean(np.diff(dataset[key].timestamp)))
        key_base = all_keys[np.argmax(time_step)]
    timestamps = dataset[key_base].iloc[:, 0]

    condition = timestamps < min_timestamp
    timestamps = timestamps[np.sum(condition):]

    condition = timestamps > max_timestamp
    timestamps = timestamps[: (len(timestamps) - np.sum(condition) - 1)]

    interpolated_dataset['timestamp'] = timestamps.values

    for key in dataset.keys():
        time = dataset.get(key).iloc[:, 0].values
        val = dataset.get(key).iloc[:, 1].values
        if len(dataset.get(key).iloc[:, 0]) > 5 * len(timestamps):
            time = [time[0]] + [np.mean(array) for array in np.array_split(time[1:-1], len(timestamps))] + [time[-1]]
            val = [val[0]] + [np.mean(array) for array in np.array_split(val[1:-1], len(timestamps))] + [val[-1]]
            # interpolated_dataset[key] = np.array([timestamps, np.interp(timestamps, time, val)]).transpose()

        # interpolated_dataset[key] = np.array([timestamps, np.interp(timestamps, time, val)]).transpose()
        interpolator_func = interp1d(time, np.array([v for v in val]), axis=0)
        val_interp = interpolator_func(timestamps)
        if len(val_interp.shape) == 1:
            interpolated_dataset[key] = val_interp
        else:
            interpolated_dataset[key] = [v for v in val_interp]

    intepolated_dataframe = pd.DataFrame(interpolated_dataset)
    # intepolated_dataframe = pd.DataFrame(np.vstack((timestamps, np.array([interpolated_dataset[key][:, 1] for
    #                                                                         key in interpolated_dataset.keys()]))).transpose())
    # keys = ['timestamp']
    # keys.extend(interpolated_dataset.keys())
    # intepolated_dataframe.columns = keys

    # intepolated_dataframe['mu_t'] = np.log( intepolated_dataframe['i0'] / intepolated_dataframe['it'] )
    # intepolated_dataframe['mu_f'] = intepolated_dataframe['iff'] / intepolated_dataframe['i0']
    # intepolated_dataframe['mu_r'] = np.log( intepolated_dataframe['it'] / intepolated_dataframe['ir'] )

    if sort:
        return intepolated_dataframe.sort_values('energy')
    else:
        return intepolated_dataframe


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
    binned_df = binned_df.drop('timestamp', 1)
    print(f'({ttime.ctime()}) Binning the data: DONE')
    return binned_df


def get_processed_df_from_uid(run, save_interpolated_file=False):
    md = run.metadata

    experiment = md['start']['experiment']

    # comments = create_file_header(hdr)
    # path_to_file = hdr.start['interp_filename']
    # path_to_file = _shift_root(path_to_file)
    # validate_path_exists(path_to_file)
    # path_to_file = validate_file_exists(path_to_file, file_type='interp')
    e0 = find_e0(md)
    data_kind = 'default'
    # file_list = []

    if experiment == 'fly_scan':

        # path_to_file = validate_file_exists(path_to_file, file_type='interp')
        stream_names = list(run.keys())
        try:
            # default detectors
            apb_df, energy_df, energy_offset = load_apb_dataset_from_tiled(run)
            raw_dict = translate_apb_dataset(apb_df, energy_df, energy_offset)

            # for stream_name in stream_names:
            #     if stream_name == 'pil100k_stream':
            #         apb_trigger_pil100k_timestamps = load_apb_trig_dataset_from_db(db, uid, use_fall=True,
            #                                                                        stream_name='apb_trigger_pil100k')
            #         pil100k_dict = load_pil100k_dataset_from_db(db, uid, apb_trigger_pil100k_timestamps)
            #         raw_dict = {**raw_dict, **pil100k_dict}
            #
            #     elif stream_name == 'xs_stream':
            #         apb_trigger_xs_timestamps = load_apb_trig_dataset_from_db(db, uid, stream_name='apb_trigger_xs')
            #         xs3_dict = load_xs3_dataset_from_db(db, uid, apb_trigger_xs_timestamps)
            #         raw_dict = {**raw_dict, **xs3_dict}

            # logger.info(f'({ttime.ctime()}) Loading file successful for UID {uid}/{path_to_file}')
        except Exception as e:
            # logger.info(f'({ttime.ctime()}) Loading file failed for UID {uid}/{path_to_file}')
            raise e
        try:
            interpolated_df = interpolate(raw_dict)
            # logger.info(f'({ttime.ctime()}) Interpolation successful for {path_to_file}')
            # if save_interpolated_file:
            #     save_interpolated_df_as_file(path_to_file, interpolated_df, comments)
        except Exception as e:
            # logger.info(f'({ttime.ctime()}) Interpolation failed for {path_to_file}')
            raise e

        try:
            if e0 > 0:
                processed_df = rebin(interpolated_df, e0)
                # (path, extension) = os.path.splitext(path_to_file)
                # path_to_file = path + '.dat'
                # logger.info(f'({ttime.ctime()}) Binning successful for {path_to_file}')

                # if draw_func_interp is not None:
                #     draw_func_interp(interpolated_df)
                # if draw_func_bin is not None:
                #     draw_func_bin(processed_df, path_to_file)
            else:
                print(f'({ttime.ctime()}) Energy E0 is not defined')
        except Exception as e:
            # logger.info(f'({ttime.ctime()}) Binning failed for {path_to_file}')
            raise e

        # save_binned_df_as_file(path_to_file, processed_df, comments)

    #
    # elif (experiment == 'step_scan') or (experiment == 'collect_n_exposures'):
    #     # path_to_file = validate_file_exists(path_to_file, file_type='interp')
    #     df = stepscan_remove_offsets(md)
    #     df = stepscan_normalize_xs(df)
    #     processed_df = filter_df_by_valid_keys(df)
    #     # df_processed = combine_xspress3_channels(df)
    #
    # else:
    #     return
    #
    # processed_df = combine_xspress3_channels(processed_df)

    # primary_df, extended_data = split_df_data_into_primary_and_extended(processed_df)

    ### WIP
    # if 'spectrometer' in md['start'].keys():
    #     if md['start']['spectrometer'] == 'von_hamos':
    #         pass
    # extended_data, comments, file_paths = process_von_hamos_scan(primary_df, extended_data, comments, hdr, path_to_file, db=db)
    # data_kind = 'von_hamos'
    # file_list = file_paths
    # save_vh_scan_to_file(path_to_file, vh_scan, comments)
    # file_list.append(path_to_file)
    return md, processed_df,  # comments, path_to_file, file_list, data_kind


# @task
# def log_uid(ref):
#     run = tiled_client_iss[ref]
#     full_uid = run.start["uid"]
#     # logger.info(f"{full_uid = }")
#     print(f"{full_uid = }")

#     # Returns work like normal.
#     return full_uid


# @task
def process_run(ref):
    run = tiled_client_iss[ref]
    full_uid = run.start["uid"]
    # logger.info(
    #     f"Now we have the full uid: {full_uid}, we can do something with it"
    # )
    print(f"Now we have the full uid: {full_uid}, we can do something with it")
    get_processed_df_from_uid(run)
    # logger.info(
    #     f"processing math works!"
    # )
    print("processing math works!")


# @task
# def wait_for_all_tasks():
#     logger.info("All tasks completed")

def processing_flow(ref):
    process_run(ref)

# with Flow("processing") as flow:
#     # We use ref because we can pass in an index, a scan_id,
#     # or a uid in the UI.
#     ref = Parameter("ref")
#     full_uid = log_uid(ref)
#     process_run_task = process_run(full_uid)
#     wait_for_all_tasks(upstream_tasks=[process_run_task])
