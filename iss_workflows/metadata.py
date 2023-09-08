import copy

def get_processed_md(run_metadata):
    md = copy.deepcopy(run_metadata['start'])

    # md['time_start'] = md.pop('time')
    md['time_stop'] = run_metadata['stop']['time']
    md['time_duration'] = md['time_stop'] - md['time']
    # md['time'] = (md['time_stop'] + md['time_start']) / 2
    if 'PROPOSAL' in md.keys():
        md['proposal'] = md.pop('PROPOSAL')
    else:
        md['proposal'] = md.pop('proposal')
    md['exit_status'] = run_metadata['stop']['exit_status']
#
    if 'scan_kind' not in md.keys():
        md['scan_kind'] = infer_scan_kind(md)

    if 'scan_name' not in md.keys():
        md['scan_name'] = f"{md['element']}-{md['edge']}"

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