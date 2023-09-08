import pandas as pd
import numpy as np

def check_saturation(df: pd.DataFrame, keys=None, threshold=3200):
    """
    Check currents (in mV) are not saturated relative to threshold.

    Returns `dict` with current channel keys (default `["i0", "it", "ir", "iff"]`) and `bool`
    values. `True` corresponds to unsaturated/good current. `False` corresponds to saturated current.
    """
    if keys is None:
        keys = ["i0", "it", "ir", "iff"]
    sat_dict = dict()
    for k in keys:
        if np.any(np.abs(df[k]) > threshold):
            sat_dict[k] = False
        else:
            sat_dict[k] = True
    return sat_dict


def check_amplitude(df: pd.DataFrame, keys=None, threshold=20):
    """
    Check currents (in mV) have sufficient non-zero amplitude relative to threshold.

    Returns `dict` with current channel keys (default `["i0", "it", "ir", "iff"]`) and `bool`
    values. `True` corresponds to good amplitude current. `False` corresponds to poor amplitude.
    """
    low_amp_dict = dict()
    if keys is None:
        keys = ["i0", "it", "ir", "iff"]
    for k in keys:
        if np.all(np.abs(df[k]) < threshold):
            low_amp_dict[k] = False
        else:
            low_amp_dict[k] = True
    return low_amp_dict


def check_mu_values(df: pd.DataFrame, good_amp_currents: dict):
    """
    Calculate mu values for each channel (transmission, reference, fluorescence) and
    check all mu values are valid (no `NaN`, `inf`, etc.).

    Returns `dict` with mu channel keys (`"mut"`, `"mur"`, `"muf"`) and `bool` values.
    `True` corresponds to all valid values. `False` indicates at least one value is invalid
    for that channel.
    """

    valid_values = {"mut": False, "mur": False, "muf": False}
    if good_amp_currents['i0'] and good_amp_currents['it']:
        mut = -np.log(df["it"] / df["i0"])
        valid_values['mut'] = np.all(np.isfinite(mut))
    if good_amp_currents['it'] and good_amp_currents['ir']:
        mur = -np.log(df["ir"] / df["it"])
        valid_values['mur'] = np.all(np.isfinite(mur))
    if good_amp_currents['i0'] and good_amp_currents['iff']:
        muf = df["iff"] / df["i0"]
        valid_values['muf'] = np.all(np.isfinite(muf))
    return valid_values


def check_apb_quality(df_mV: pd.DataFrame):# , md: dict):
    """
    Combine `degain`, `check_saturation`, `check_amplitude`, and `check_mu_values`
    to determine data (mu) quality for each channel (transmission, fluorescence, reference).

    Returns `dict` with mu channel keys (`"mut"`, `"mur"`, `"muf"`) and `bool` values.
    `True` indicates good data quality. `False` indicates poor data quality.
    """

    # df_mV = degain(df, md)
    unsaturated_currents = check_saturation(df_mV)
    good_amp_currents = check_amplitude(df_mV)
    valid_mu = check_mu_values(df_mV, good_amp_currents)
    mu_good = {mu: 'good' for mu in ["mut", "muf", "mur"]}  # default all good

    for k in unsaturated_currents.keys():
        mu_good[k] = 'good'
        if not unsaturated_currents[k]:
            mu_good[k] = 'saturated'
        if not good_amp_currents[k]:
            mu_good[k] = 'low_amplitude'

    if good_amp_currents["i0"] and unsaturated_currents["i0"]:
        if unsaturated_currents["it"]:
            mu_good["mut"] = 'good' if valid_mu["mut"] else 'invalid_values'
        else:
            mu_good["mut"] = 'saturated'
        if unsaturated_currents["iff"]:
            mu_good["muf"] = 'good' if valid_mu["muf"] else 'invalid_values'
        else:
            mu_good["muf"] = 'saturated'
    else:
        mu_good["mut"] = 'low_amplitude' if not good_amp_currents["i0"] else 'saturated'
        mu_good["muf"] = 'low_amplitude' if not good_amp_currents["i0"] else 'saturated'

    if (
        good_amp_currents["it"]
        and unsaturated_currents["it"]
        and unsaturated_currents["ir"]
    ):
        mu_good["mur"] = 'good' if valid_mu["mur"] else 'invalid_values'
    else:
        if not good_amp_currents["it"]:
            mu_good["mur"] = 'low_amplitude'
        if (not unsaturated_currents["it"]) or (not unsaturated_currents["ir"]):
            mu_good["mur"] = 'saturated'
    return mu_good

def check_xs_quality(df, total_roi_idx=4, threshold=450e3, i0_quality=None):
    channels = [f'xs_ch{ch_i:02d}_roi{total_roi_idx:02d}' for ch_i in [1, 2, 3, 4]]
    mu_good = {mu: 'good' for mu in [f'xs_ch{ch_i:02d}' for ch_i in [1, 2, 3, 4]]}  # default all False
    for mu_channel, data_key in zip(mu_good.keys(), channels):
        intensities = df[data_key].values / df['exposure_time']
        mu_good[mu_channel] = 'good' if np.all(intensities <= threshold) else 'saturated'
        if i0_quality is not None:
            if (i0_quality != 'good') and (mu_good[mu_channel] == 'good'):
                mu_good[mu_channel] = i0_quality
    return mu_good
