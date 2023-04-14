import prefect
from prefect import task, Flow, Parameter

from tiled.client import from_profile

import time as ttime
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import xraydb
import numexpr as ne
import copy

tiled_client = from_profile("nsls2", username=None)["iss"]
tiled_client_iss = tiled_client["raw"]
tiled_client_sandbox = tiled_client["sandbox"]


# Used for Prefect logging.
# logger = prefect.context.get("logger")


# x-ray conversion stuff
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



# the main function

def get_processed_df_from_uid(run):
    # experiment = run.metadata['start']['experiment']
    md = get_processed_md(run.metadata)
    experiment = md['experiment']
    if experiment == 'fly_scan':
        processed_df, processed_md = process_fly_scan(run, md)

    elif (experiment == 'step_scan') or (experiment == 'collect_n_exposures'):
        pass

    # processed_df = combine_xspress3_channels(processed_df)

        # logger.info(f'({ttime.ctime()}) Loading file successful for UID {uid}/{path_to_file}')

        # try:
        #
        #     # logger.info(f'({ttime.ctime()}) Interpolation successful for {path_to_file}')
        #     # if save_interpolated_file:
        #     #     save_interpolated_df_as_file(path_to_file, interpolated_df, comments)
        # except Exception as e:
        #     # logger.info(f'({ttime.ctime()}) Interpolation failed for {path_to_file}')
        #     raise e
        #
        # try:
        #     if e0 > 0:
        #         processed_df = rebin(interpolated_df, e0)
        #         # (path, extension) = os.path.splitext(path_to_file)
        #         # path_to_file = path + '.dat'
        #         # logger.info(f'({ttime.ctime()}) Binning successful for {path_to_file}')
        #
        #         # if draw_func_interp is not None:
        #         #     draw_func_interp(interpolated_df)
        #         # if draw_func_bin is not None:
        #         #     draw_func_bin(processed_df, path_to_file)
        #     else:
        #         print(f'({ttime.ctime()}) Energy E0 is not defined')
        # except Exception as e:
        #     # logger.info(f'({ttime.ctime()}) Binning failed for {path_to_file}')
        #     raise e

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

def process_fly_scan(run, md):
    apb_df = load_apb_dataset_from_tiled(run)
    energy_df = load_hhm_encoder_dataset_from_tiled(run)

    apb_dict = translate_dataset(apb_df)
    energy_dict = translate_dataset(energy_df, columns=['energy'])

    raw_dict = {**apb_dict, **energy_dict}

    for stream_name in run.metadata['summary']['stream_names']:
        if stream_name == 'pil100k_stream':
            pil100k_df = load_pil100k_dataset_from_tiled(run)
            pil100k_dict = translate_dataset(pil100k_df)
            raw_dict = {**raw_dict, **pil100k_dict}
    #
    #     elif stream_name == 'xs_stream':
    #         apb_trigger_xs_timestamps = load_apb_trig_dataset_from_db(db, uid, stream_name='apb_trigger_xs')
    #         xs3_dict = load_xs3_dataset_from_db(db, uid, apb_trigger_xs_timestamps)
    #         raw_dict = {**raw_dict, **xs3_dict}

    interpolated_df = interpolate(raw_dict)
    interpolated_md = copy.deepcopy(md)
    interpolated_md['processing_step'] = 'interpolated'
    # upload interpolated data

    e0 = md['e0']
    rebinned_df = rebin(interpolated_df, e0)
    rebinned_md = copy.deepcopy(md)
    rebinned_md['processing_step'] = 'rebinned'
    return rebinned_df, rebinned_md


# metadata transformations

def get_processed_md(run_metadata):
    md = copy.deepcopy(run_metadata['start'])

    md['time_start'] = md.pop('time')
    md['time_stop'] = run_metadata['stop']['time']
    md['time_duration'] = md['time_stop'] - md['time_start']
    md['time'] = (md['time_stop'] + md['time_start']) / 2

    md['proposal'] = md.pop('PROPOSAL')
    md['exit_status'] = run_metadata['stop']['exit_status']

    if 'scan_kind' not in md.keys():
        md['scan_kind'] = infer_scan_kind(md)

    return md


def infer_scan_kind(md):
    experiment = md['experiment']
    spectrometer_is_used = ('spectrometer' in md.keys())
    if spectrometer_is_used:
        if ((md['experiment'] == 'step_scan') and
            (md['spectrometer'] == 'johann') and
            ('spectrometer_energy_steps' in md.keys()) and
            (len(md['spectrometer_energy_steps']) > 1)):
            mono_is_moving = False
        else:
            mono_is_moving = True
    else:
        mono_is_moving = (experiment != 'collect_n_exposures_plan')

    if spectrometer_is_used:
        spectrometer_is_vonhamos = (md['spectrometer'] == 'von_hamos')
        if (not spectrometer_is_vonhamos):
            try:
                spectrometer_is_moving = (md['spectrometer_config']['scan_type'] != 'constant energy')
            except KeyError:
                try:
                    spectrometer_is_moving = ('spectrometer_energy_steps' in md.keys()) & (len(md['spectrometer_energy_steps']) > 1)
                except KeyError:
                    spectrometer_is_moving = False
        else:
            spectrometer_is_moving = False
    else:  # redundant but why not
        spectrometer_is_moving = False
        spectrometer_is_vonhamos = False

    if mono_is_moving:
        if (not spectrometer_is_used):
            scan_kind = 'xas'
        else:
            if spectrometer_is_moving:  # only johann can move together with mono
                scan_kind = 'johann_rixs'
            else:
                if spectrometer_is_vonhamos:
                    scan_kind = 'von_hamos_rixs'
                else:
                    scan_kind = 'johann_herfd'
    else:
        if (not spectrometer_is_used):
            scan_kind = 'constant_e'
        else:
            if spectrometer_is_moving:
                scan_kind = 'johann_xes'
            else:
                if spectrometer_is_vonhamos:
                    scan_kind = 'von_hamos_xes'
                else:
                    scan_kind = 'constant_e_johann'
    return scan_kind


# tiled_io

def _load_dataset_from_tiled(run, stream_name, field_name=None):
    if field_name is None:
        t = run[stream_name]['data'][stream_name].read()
        columns = list(t.dtype.fields.keys())
    else:
        t = run[stream_name]['data'][field_name].read()
        columns = [field_name]
    arr = np.array(t.tolist()).squeeze()
    return arr, columns

def load_dataset_from_tiled(run, stream_name, field_name=None):
    arr, columns = _load_dataset_from_tiled(run, stream_name, field_name=field_name)
    return pd.DataFrame(arr, columns=columns)

def _fix_apb_dataset_from_tiled(run):
    arr, columns = _load_dataset_from_tiled(run, 'apb_stream')
    # filter by time
    t = arr[:, 0]
    t_min = t[0] - 1e8  # s
    t_max = 5e9  # Fri Jun 11 04:53:20 2128
    arr = arr[(t >= t_min) & (t <= t_max), :]

    # filter by amplitude
    i0_min = -4e6  # uV
    i0_max = 0.5e6  # uV
    i0 = arr[:, 1]
    arr = arr[(i0 >= i0_min) & (i0 <= i0_max), :]

    # filter by repeating values
    _, idx_ord = np.unique(arr[:, 0], return_index=True)
    arr = arr[idx_ord, :]
    return pd.DataFrame(arr, columns=columns)

def get_ch_properties(hdr_start, start, end):
    ch_keys = [key for key in hdr_start.keys() if key.startswith(start) and key.endswith(end)]
    return np.array([hdr_start[key] for key in ch_keys])

def load_apb_dataset_from_tiled(run):
    # apb_dataset = load_dataset_from_tiled(run, 'apb_stream')
    apb_dataset = _fix_apb_dataset_from_tiled(run) # bandaid to deal with analogue pizzabox tiled readout
    ch_offsets = get_ch_properties(run.metadata['start'], 'ch', '_offset') * 1e3  # offsets are ib mV but the readings are in uV
    ch_gains = get_ch_properties(run.metadata['start'], 'ch', '_amp_gain')

    apb_dataset.iloc[:, 1:] -= ch_offsets
    apb_dataset.iloc[:, 1:] /= 1e6
    apb_dataset.iloc[:, 1:] /= (10 ** ch_gains)

    apb_dataset['mutrans'] = -np.log(apb_dataset['it'] / apb_dataset['i0'])
    apb_dataset['murefer'] = -np.log(apb_dataset['ir'] / apb_dataset['it'])
    apb_dataset['mufluor'] = apb_dataset['iff'] / apb_dataset['i0']

    return apb_dataset

def load_hhm_encoder_dataset_from_tiled(run):
    hhm_dataset = load_dataset_from_tiled(run, 'pb9_enc1')
    hhm_dataset = hhm_dataset[hhm_dataset['ts_s'] > 0]
    angle_offset = -float(run.metadata['start']['angle_offset'])

    hhm_dataset['timestamp'] = hhm_dataset['ts_s'] + 1e-9 * hhm_dataset['ts_ns']
    hhm_dataset['encoder'] = hhm_dataset['encoder'].apply(lambda x: int(x) if int(x) <= 0 else -(int(x) ^ 0xffffff - 1))
    hhm_dataset['energy'] = encoder2energy(hhm_dataset['encoder'] , 360000, angle_offset)

    return hhm_dataset

def _load_apb_trig_dataset_from_tiled(run, stream_name='apb_trigger_xs'):
    df = load_dataset_from_tiled(run, stream_name)
    df = df[df['timestamp'] > 1e8]
    timestamps = df.timestamp.values
    transitions = df.transition.values

    if transitions[0] == 0:
        timestamps = timestamps[1:]
        transitions = transitions[1:]
    rises = timestamps[0::2]
    falls = timestamps[1::2]
    n_0 = np.sum(transitions == 0)
    n_1 = np.sum(transitions == 1)
    n_all = np.min([n_0, n_1])
    apb_trig_timestamps = (rises[:n_all] + falls[:n_all]) / 2
    return apb_trig_timestamps


def _load_pil100k_dataset_from_tiled(run):#, apb_trig_timestamps):
    field_names = ['pil100k_roi1', 'pil100k_roi2', 'pil100k_roi3', 'pil100k_roi4', 'pil100k_image']
    data = {}
    for field_name in field_names:
        arr, columns = _load_dataset_from_tiled(run, 'pil100k_stream', field_name)
        column = columns[0]
        data[column] = [v for v in arr]
    return pd.DataFrame(data)

def load_pil100k_dataset_from_tiled(run):
    timestamp = _load_apb_trig_dataset_from_tiled(run, stream_name='apb_trigger_pil100k')
    df = _load_pil100k_dataset_from_tiled(run)

    n_pulses = timestamp.size
    n_images = len(df)
    n = np.min([n_pulses, n_images])

    df = df.iloc[:n, :]
    timestamp = timestamp[:n]
    df['timestamp'] = timestamp

    return df




    # output = {}
    # # t = hdr.table(stream_name='pil100k_stream', fill=True)
    # field_list = ['pil100k_roi1', 'pil100k_roi2', 'pil100k_roi3', 'pil100k_roi4']#, 'pil100k_image']
    # _t = {field : list(hdr.data(stream_name='pil100k_stream', field=field))[0] for field in field_list}
    # if load_images:
    #     _t['pil100k_image'] = [i for i in list(hdr.data(stream_name='pil100k_stream', field='pil100k_image'))[0]]
    # t = pd.DataFrame(_t)
    # # n_images = t.shape[0]
    # n_images = min(t['pil100k_roi1'].size, apb_trig_timestamps.size)
    # pil100k_timestamps = apb_trig_timestamps[:n_images]
    # keys = [k for k in t.keys() if (k != 'time') ]#and (k != 'pil100k_image')]
    # t = t[:n_images]
    # for j, key in enumerate(keys):
    #     output[key] = pd.DataFrame(np.vstack((pil100k_timestamps, t[key])).T, columns=['timestamp', f'{key}'])
    # return output


def translate_dataset(df, columns=None):
    if columns is None:
        columns = [c for c in df.columns if c!='timestamp']
    data_dict = {}
    for column in columns:
        data_dict[column] = df[['timestamp', column]]
    return data_dict

def derive_common_timestamp_grid(dataset, key_base=None):
    min_timestamp = max([df['timestamp'].min() for key, df in dataset.items()])
    max_timestamp = min([df['timestamp'].max() for key, df in dataset.items()])

    if key_base is None:
        all_keys = []
        time_step = []
        for key, df in dataset.items():
            all_keys.append(key)
            time_step.append(np.median(np.diff(df.timestamp)))
        key_base = all_keys[np.argmax(time_step)]
    timestamps = dataset[key_base].timestamp.values

    return timestamps[(timestamps >= min_timestamp) & (timestamps <= max_timestamp)]


def interpolate(dataset, key_base = None, sort=True):
    timestamp = derive_common_timestamp_grid(dataset, key_base=key_base)

    interpolated_dataset = {'timestamp': timestamp}
    # interpolated_dataset['timestamp'] = timestamp

    for key, df in dataset.items():
        time = df['timestamp'].values
        val = df[key].values
        if time.size > 5 * len(timestamp):
            time = [time[0]] + [np.mean(array) for array in np.array_split(time[1:-1], len(timestamp))] + [time[-1]]
            val = [val[0]] + [np.mean(array) for array in np.array_split(val[1:-1], len(timestamp))] + [val[-1]]

        interpolator_func = interp1d(time, np.array([v for v in val]), axis=0)
        val_interp = interpolator_func(timestamp)
        if len(val_interp.shape) == 1:
            interpolated_dataset[key] = val_interp
        else:
            interpolated_dataset[key] = [v for v in val_interp]

    intepolated_dataframe = pd.DataFrame(interpolated_dataset)
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
    # binned_df = binned_df.drop('timestamp', axis=1)
    print(f'({ttime.ctime()}) Binning the data: DONE')
    return binned_df



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
    md, df = get_processed_df_from_uid(run)
    md = dict(md)
    md['TESTING'] = 'TESTING'
    md['summary'].pop('datetime')
    tiled_client_sandbox.write_dataframe(df, md)
    print("uploading works!")
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
