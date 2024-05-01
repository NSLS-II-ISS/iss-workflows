from prefect import task, flow, get_run_logger

from tiled.client import from_profile
import time as ttime
import numpy as np
import pandas as pd
import copy

from tiled_io import (load_apb_dataset_from_tiled, load_hhm_encoder_dataset_from_tiled,\
                      load_pil100k_dataset_from_tiled, load_xs_dataset_from_tiled, translate_dataset)
from metadata import get_processed_md
from interpolate import interpolate
from rebin import rebin
from tiled_io import _xs_ch_roi_keys, _xs_roi_combine_dict, _pil100k_roi_keys
_external_detector_keys = _xs_ch_roi_keys + list(_xs_roi_combine_dict.keys()) + _pil100k_roi_keys

tiled_client = from_profile("nsls2")["iss"]
tiled_client_iss = tiled_client["raw"]
tiled_client_sandbox = tiled_client["sandbox"]





def get_processed_df_from_uid(run):
    # experiment = run.metadata['start']['experiment']
    md = get_processed_md(run.metadata)
    experiment = md['experiment']
    if experiment == 'fly_scan':
        processed_df, processed_md = process_fly_scan(run, md)

    elif (experiment == 'step_scan') or (experiment == 'collect_n_exposures'):
        pass
        # df = stepscan_remove_offsets(hdr)
        # df = stepscan_normalize_xs(df)
        # processed_df = filter_df_by_valid_keys(df)

    # processed_df = combine_xspress3_channels(processed_df)

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

def read_fly_scan_streams(run, md):
    apb_df, apb_quality_dict = load_apb_dataset_from_tiled(run)
    md['scan_quality'] = apb_quality_dict
    energy_df = load_hhm_encoder_dataset_from_tiled(run)

    apb_dict = translate_dataset(apb_df)
    energy_dict = translate_dataset(energy_df, columns=['energy'])

    raw_dict = {**apb_dict, **energy_dict}

    for stream_name in run.metadata['summary']['stream_names']:
        if stream_name == 'pil100k_stream':
            pil100k_df = load_pil100k_dataset_from_tiled(run)
            pil100k_dict = translate_dataset(pil100k_df)
            raw_dict = {**raw_dict, **pil100k_dict}

        if stream_name == 'xs_stream':
            xs_df, xs_quality_dict = load_xs_dataset_from_tiled(run, i0_quality=apb_quality_dict['i0'])
            xs_dict = translate_dataset(xs_df)
            raw_dict = {**raw_dict, **xs_dict}
            md['scan_quality'] = {**md['scan_quality'], **xs_quality_dict}

    return raw_dict, md

def interpolate_fly_scan_raw_dict(raw_dict, md):
    interpolated_df = interpolate(raw_dict)
    interpolated_df = normalize_external_detectors_by_i0(interpolated_df)
    interpolated_md = copy.deepcopy(md)
    interpolated_md['processing_step'] = 'interpolated'
    # upload interpolated data
    return interpolated_df, interpolated_md

def rebin_fly_scan_interpolated_df(interpolated_df, md, return_convo_mat=False):
    e0 = md['e0']
    rebinned_df, convo_mat = rebin(interpolated_df, e0)
    rebinned_md = copy.deepcopy(md)
    rebinned_md['processing_step'] = 'rebinned'
    if return_convo_mat:
        return rebinned_df, rebinned_md, convo_mat
    else:
        return rebinned_df, rebinned_md

def process_fly_scan(run, md):
    raw_dict, md = read_fly_scan_streams(run, md)
    interpolated_df, interpolated_md = interpolate_fly_scan_raw_dict(raw_dict, md)
    rebinned_df, rebinned_md = rebin_fly_scan_interpolated_df(interpolated_df, md)
    return rebinned_df, rebinned_md



def normalize_external_detectors_by_i0(df):
    i0 = df['i0'].values.copy()
    i0 /= np.median(i0)
    for key in _external_detector_keys:
        if key in df.keys():
            df[key] = df[key] / i0
    return df


# @task
# def log_uid(ref):
#     run = tiled_client_iss[ref]
#     full_uid = run.start["uid"]
#     # logger.info(f"{full_uid = }")
#     print(f"{full_uid = }")

#     # Returns work like normal.
#     return full_uid


@task
def process_run(ref):
    logger = get_run_logger()
    run = tiled_client_iss[ref]
    full_uid = run.start["uid"]
    logger.info(
        f"Now we have the full uid: {full_uid}, we can do something with it"
    )
    md, df = get_processed_df_from_uid(run)
    md['tag'] = 'prefect testing'

    tiled_client_sandbox.write_dataframe(df, metadata=md)
    logger.info(
        f"processing math works!"
    )



# @task
# def wait_for_all_tasks():
#     logger.info("All tasks completed")

@flow
def processing_flow(ref):
    process_run(ref)

# for uid in tiled_client_iss.search(Key('PROPOSAL') == '310173').search(Key('year') == '2023').search(Key('experiment') == 'fly_scan'):
#     print(uid)
#     try:
#         process_run(uid)
#     except:
#         print(f'{uid} could not be processed')

# process_run('9fca69cd-bf38-41f3-bc5b-e7a25ccb8ee8')

# with Flow("processing") as flow:
#     # We use ref because we can pass in an index, a scan_id,
#     # or a uid in the UI.
#     ref = Parameter("ref")
#     full_uid = log_uid(ref)
#     process_run_task = process_run(full_uid)
#     wait_for_all_tasks(upstream_tasks=[process_run_task])
