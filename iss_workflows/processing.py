

from tiled.client import from_profile
import time as ttime
import numpy as np
import pandas as pd
import copy

from iss_workflows.tiled_io import (load_apb_dataset_from_tiled, load_hhm_encoder_dataset_from_tiled,\
                      load_pil100k_dataset_from_tiled, load_xs_dataset_from_tiled, translate_dataset)
from iss_workflows.metadata import get_processed_md
from iss_workflows.interpolate import interpolate
from iss_workflows.rebin import rebin
from iss_workflows.tiled_io import _external_detector_keys
from iss_workflows.file_io import write_df_to_file

LOGGER = None
def get_prefect_logger():
    global LOGGER
    if LOGGER is None:
        try:
            from prefect import get_run_logger
            LOGGER = get_run_logger()
        except:
            pass

tiled_client_iss = None
tiled_client_sandbox = None
def get_tiled_nodes():
    global tiled_client_iss
    global tiled_client_sandbox
    if (tiled_client_iss is None) or (tiled_client_sandbox is None):
        tiled_client = from_profile("nsls2")["iss"] #["raw"]
        tiled_client_iss = tiled_client["raw"]
        tiled_client_sandbox = tiled_client["sandbox"]



def logger_info_decorator(function):
    def wrapper(*args, logger_msg='', **kwargs):
        if LOGGER is not None:
            LOGGER.info(f"{logger_msg}: beginning")
        try:
            result = function(*args, **kwargs)
            if LOGGER is not None:
                LOGGER.info(f"{logger_msg}: beginning")
            return result
        except Exception as e:
            LOGGER.info(f"{logger_msg}: FAILED. Reason: {e}")
            raise e
    return wrapper

@logger_info_decorator
def get_processed_df_from_run(run,
                              heavyweight_processing=True,
                              processing_kwargs=None,
                              draw_func_interp=None):
    md = get_processed_md(run.metadata)
    experiment = md['experiment']

    if experiment == 'fly_scan':
        processed_df, processed_md = process_fly_scan(run, md, draw_func_interp=draw_func_interp)

    elif (experiment == 'step_scan') or (experiment == 'collect_n_exposures'):
        process_step_scan(run, md)

    elif experiment == 'epics_fly_scan':
        process_epics_fly_scan(run, md)



    ### WIP
    # if 'spectrometer' in md['start'].keys():
    #     if md['start']['spectrometer'] == 'von_hamos':
    #         pass
    # extended_data, comments, file_paths = process_von_hamos_scan(primary_df, extended_data, comments, hdr, path_to_file, db=db)
    # data_kind = 'von_hamos'
    # file_list = file_paths
    # save_vh_scan_to_file(path_to_file, vh_scan, comments)
    # file_list.append(path_to_file)
    return processed_df, md  # comments, path_to_file, file_list, data_kind

@logger_info_decorator
def read_fly_scan_streams(run, md):
    raw_dict = {}

    # make sure that apb_stream is the first stream to be processed
    stream_names = [s for s in run.metadata['summary']['stream_names']] # paranoia
    if 'apb_stream' in stream_names:
        stream_names.remove('apb_stream')
        stream_names = ['apb_stream'] + stream_names

    for stream_name in stream_names:
        if stream_name == 'apb_stream':
            apb_df, apb_quality_dict = load_apb_dataset_from_tiled(run)
            apb_dict = translate_dataset(apb_df)
            raw_dict = {**raw_dict, **apb_dict}
            md['scan_quality'] = apb_quality_dict

        elif stream_name == 'pb9_enc1':
            energy_df = load_hhm_encoder_dataset_from_tiled(run)
            energy_dict = translate_dataset(energy_df, columns=['energy'])
            raw_dict = {**raw_dict, **energy_dict}

        elif stream_name.startswith('pil100k'):
            pil_name = stream_name.split('_')[0]
            pil100k_df = load_pil100k_dataset_from_tiled(run, pil_name=pil_name)
            pil100k_dict = translate_dataset(pil100k_df)
            raw_dict = {**raw_dict, **pil100k_dict}

        elif stream_name == 'xs_stream':
            xs_df, xs_quality_dict = load_xs_dataset_from_tiled(run, i0_quality=apb_quality_dict['i0'])
            xs_dict = translate_dataset(xs_df)
            raw_dict = {**raw_dict, **xs_dict}
            md['scan_quality'] = {**md['scan_quality'], **xs_quality_dict}

    return raw_dict, md

@logger_info_decorator
def interpolate_fly_scan_raw_dict(raw_dict, md):
    interpolated_df = interpolate(raw_dict)
    # interpolated_df = normalize_external_detectors_by_i0(interpolated_df)
    interpolated_md = copy.deepcopy(md)
    interpolated_md['processing_step'] = 'interpolated'
    # upload interpolated data
    return interpolated_df, interpolated_md

@logger_info_decorator
def rebin_fly_scan_interpolated_df(interpolated_df, md, return_convo_mat=False):
    e0 = md['e0']
    rebinned_df, convo_mat = rebin(interpolated_df, e0)
    rebinned_md = copy.deepcopy(md)
    rebinned_md['processing_step'] = 'rebinned'
    if return_convo_mat:
        return rebinned_df, rebinned_md, convo_mat
    else:
        return rebinned_df, rebinned_md

def process_fly_scan(run, md, draw_func_interp=None):
    raw_dict, md = read_fly_scan_streams(run, md,
                                         logger_msg='\t Loading fly data streams')
    interpolated_df, interpolated_md = interpolate_fly_scan_raw_dict(raw_dict, md,
                                         logger_msg='\t Interpolating fly data onto a common time grid')
    if draw_func_interp is not None:
        draw_func_interp(interpolated_df)
    rebinned_df, rebinned_md = rebin_fly_scan_interpolated_df(interpolated_df, md,
                                                              logger_msg='\t Rebinning fly data')
    return rebinned_df, rebinned_md

def process_step_scan(run, md):
    pass
    # df = stepscan_remove_offsets(hdr)
    # df = stepscan_normalize_xs(df)
    # processed_df = filter_df_by_valid_keys(df)
    # raise Exception('step_scan and count processing is not implemented yet!')

def process_epics_fly_scan(run, md):
    pass

def normalize_external_detectors_by_i0(df):
    i0 = df['i0'].values.copy()
    i0 /= np.median(i0)
    for key in _external_detector_keys:
        if key in df.keys():
            df[key] = df[key] / i0
    return df


@logger_info_decorator
def upload_data_to_sandbox(df, md):
    tiled_client_sandbox.write_dataframe(df, metadata=md)

@logger_info_decorator
def save_data_to_file(df, md, dump_to_tiff=False):
    path_to_file = write_df_to_file(df, md)
    if dump_to_tiff:
        raise Exception('Saving images to tiff files is not implemented yet')
    return [path_to_file]

@logger_info_decorator
def dispatch_file_to_cloud(file, cloud_dispatcher):
    pass

@logger_info_decorator
def dispatch_data_to_cloud(files, cloud_dispatcher):
    for file in files:
        dispatch_file_to_cloud(file, cloud_dispatcher,
                               logger_msg=f'\tDispatching {file} to the cloud')

# @task
def process_run(uid,
                send_to_sandbox=False,
                save_to_file=True,
                draw_func_interp=None,
                cloud_dispatcher=None,
                dump_to_tiff=False,
                heavyweight_processing=True,
                processing_kwargs=None):

    get_tiled_nodes()

    run = tiled_client_iss[uid]
    full_uid = run.start["uid"]

    df, md = get_processed_df_from_run(run,
                                       heavyweight_processing=heavyweight_processing,
                                       processing_kwargs=processing_kwargs,
                                       draw_func_interp=draw_func_interp,
                                       logger_msg=f'Processing data for {full_uid}')

    if save_to_file:
        files = save_data_to_file(df, md, dump_to_tiff=dump_to_tiff,
                                  logger_msg=f'Saving data to file for {full_uid}')

        if cloud_dispatcher is not None:
            dispatch_data_to_cloud(files, cloud_dispatcher,
                                   logger_msg=f'Dispatching files to cloud for {full_uid}')

    if send_to_sandbox:
        upload_data_to_sandbox(df, md,
                               logger_msg=f'Uploading data to sandbox for {full_uid}')



    # md['tag'] = 'prefect testing'
    # print(df)
    # return df



def process_run_task(*args, **kwargs):
    get_prefect_logger()
    from prefect import task
    @task
    def _process_run(*args, **kwargs):
        return process_run(*args, **kwargs)

    _process_run(*args, **kwargs)



# @flow
def processing_flow(ref):
    from prefect import flow
    @flow
    def _processing_flow(ref):
        process_run_task(ref)

    _processing_flow(ref)
