import numpy as np
import pandas as pd
from iss_workflows.quality import check_apb_quality, check_xs_quality
from iss_workflows.xray import *
from itertools import product
from tqdm import tqdm

_xs_ch_range = list(range(1, 5))
_xs_roi_range = list(range(1, 5))
_xs_ch_roi_keys = [f'xs_ch{ch_i:02d}_roi{roi_i:02d}' for ch_i, roi_i in product(_xs_ch_range, _xs_roi_range)]
_xs_roi_combine_dict = {f'xs_roi{roi_i:02d}': [f'xs_ch{ch_i:02d}_roi{roi_i:02d}' for ch_i in _xs_ch_range] for roi_i in _xs_roi_range}

_pil_roi_range = list(range(1, 5))
_pil100k_roi_keys = [f'pil100k_roi{roi_i:01d}' for roi_i in _pil_roi_range]

_external_detector_keys = _xs_ch_roi_keys + list(_xs_roi_combine_dict.keys()) + _pil100k_roi_keys

# def read_tiled_array_in_chunks(arr_client):
#
#     for c in arr_client

def read_tiled_array_in_chunks(arr_client):
    output = []
    axis = [i for i, c in enumerate(arr_client.chunks) if len(c) > 1]
    if len(axis) == 0:
        axis = None
    elif len(axis) == 1:
        axis = axis[0]
    else:
        raise NotImplementedError('More general chunked-reading is not implemented yet ;-)')
    chunk_indices = list(product(*[list(range(len(c))) for c in arr_client.chunks]))
    if len(chunk_indices) == 1:
        return arr_client.read()
    else:
        for chunk_index in chunk_indices:
            output.append(arr_client.read_block(chunk_index))
        return np.concatenate(output, axis=axis)


def _load_dataset_from_tiled(run, stream_name, field_name=None):
    msg = f'[Processing] Reading stream {stream_name}'
    if field_name is None:
        arr_client = run[stream_name]['data'][stream_name]
        columns = list(arr_client.dtype.fields.keys())
    else:
        arr_client = run[stream_name]['data'][field_name]
        columns = [field_name]
        msg += f'/{field_name}'
    print(msg)
    t = read_tiled_array_in_chunks(arr_client)
    arr = np.array(t.tolist()).squeeze()
    return arr, columns

def load_dataset_from_tiled(run, stream_name, field_name=None):
    arr, columns = _load_dataset_from_tiled(run, stream_name, field_name=field_name)
    return pd.DataFrame(arr, columns=columns)

def _fix_apb_dataset_from_tiled(arr):
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
    return arr

def get_ch_properties(hdr_start, start, end):
    ch_keys = [key for key in hdr_start.keys() if key.startswith(start) and key.endswith(end)]
    return np.array([hdr_start[key] for key in ch_keys])

def load_apb_dataset_from_tiled(run, check_scan=True):
    arr, columns = _load_dataset_from_tiled(run, 'apb_stream')
    arr = _fix_apb_dataset_from_tiled(arr) # bandaid to deal with analogue pizzabox tiled readout
    apb_dataset = pd.DataFrame(arr, columns=columns)

    if check_scan:
        quality_dict = check_apb_quality(apb_dataset * 1e-3)
    else:
        quality_dict = None

    ch_offsets = get_ch_properties(run.metadata['start'], 'ch', '_offset') * 1e3  # offsets are ib mV but the readings are in uV
    ch_gains = get_ch_properties(run.metadata['start'], 'ch', '_amp_gain')

    apb_dataset.iloc[:, 1:] -= ch_offsets
    apb_dataset.iloc[:, 1:] /= 1e6
    apb_dataset.iloc[:, 1:] /= (10 ** ch_gains)

    apb_dataset['mutrans'] = -np.log(apb_dataset['it'] / apb_dataset['i0'])
    apb_dataset['murefer'] = -np.log(apb_dataset['ir'] / apb_dataset['it'])
    apb_dataset['mufluor'] = apb_dataset['iff'] / apb_dataset['i0']

    return apb_dataset, quality_dict

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
    apb_trig_durations = falls[:n_all] - rises[:n_all]
    apb_trig_timestamps = (rises[:n_all] + falls[:n_all]) / 2
    return apb_trig_timestamps, apb_trig_durations


def _load_pil100k_dataset_from_tiled(run, pil_name='pil100k', heavyweight_processing=True):
    field_names = [f'{pil_name}_roi1', f'{pil_name}_roi2', f'{pil_name}_roi3', f'{pil_name}_roi4']
    if heavyweight_processing:
        field_names.append(f'{pil_name}_image')
    data = {}
    for field_name in field_names:
        arr, columns = _load_dataset_from_tiled(run, f'{pil_name}_stream', field_name)
        column = columns[0]
        data[column] = [v for v in arr]
    return pd.DataFrame(data)

def _merge_trigger_and_detector_data(df, timestamp, exposure_time):
    n_pulses = timestamp.size
    n_images = len(df)
    n = np.min([n_pulses, n_images])
    df = df.iloc[:n, :]
    timestamp = timestamp[:n]
    exposure_time = exposure_time[:n]
    df['timestamp'] = timestamp
    df['exposure_time'] = exposure_time
    return df

def load_pil100k_dataset_from_tiled(run, pil_name='pil100k', heavyweight_processing=True):
    timestamp, exposure_time = _load_apb_trig_dataset_from_tiled(run, stream_name=f'apb_trigger_{pil_name}')
    df = _load_pil100k_dataset_from_tiled(run, pil_name=pil_name, heavyweight_processing=heavyweight_processing)
    return _merge_trigger_and_detector_data(df, timestamp, exposure_time)

def _load_xs_dataset_from_tiled(run, heavyweight_processing=True):
    field_names = [f'xs_ch{ch_i:02d}_roi{roi_i:02d}' for ch_i, roi_i in product([1, 2, 3, 4], [1, 2, 3, 4])]
    if heavyweight_processing:
        field_names = field_names + [f'xs_ch{ch_i:02d}_spectrum' for ch_i in range(1, 5)]
    data = {}
    for field_name in field_names:
        arr, columns = _load_dataset_from_tiled(run, 'xs_stream', field_name)
        column = columns[0]
        data[column] = [v for v in arr]
    return pd.DataFrame(data)

def _combine_xs_channels(df):
    for combined, list_to_be_combined in _xs_roi_combine_dict.items():
        df[combined] = df[list_to_be_combined].mean(axis=1)
    return df
def load_xs_dataset_from_tiled(run, check_scan=True, i0_quality=None, heavyweight_processing=True):
    timestamp, exposure_time = _load_apb_trig_dataset_from_tiled(run, stream_name='apb_trigger_xs')
    xs_dataset_raw = _load_xs_dataset_from_tiled(run, heavyweight_processing=heavyweight_processing)
    xs_dataset_raw = _combine_xs_channels(xs_dataset_raw)
    xs_dataset = _merge_trigger_and_detector_data(xs_dataset_raw, timestamp, exposure_time)
    if check_scan:
        quality_dict = check_xs_quality(xs_dataset, i0_quality=i0_quality)
    else:
        quality_dict = None
    return xs_dataset, quality_dict

def translate_dataset(df, columns=None):
    if columns is None:
        columns = [c for c in df.columns if c!='timestamp']
    data_dict = {}
    for column in columns:
        data_dict[column] = df[['timestamp', column]]
    return data_dict