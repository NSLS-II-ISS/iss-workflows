from scipy.interpolate import interp1d
import numpy as np
import pandas as pd

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
