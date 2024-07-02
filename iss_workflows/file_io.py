from xas.metadata import key_match as md_key_match
import os
from subprocess import call
from datetime import datetime
import numpy as np

def create_file_header(md):
    output = ''
    for key, hr_key in md_key_match.items():
        if key in md.keys():
            value = md[key]
        elif key == 'stop_time':
            value = md['time_stop']
        else:
            value = 'None'
        if (key == 'time') or (key == 'stop_time'):
            value = datetime.fromtimestamp(value).strftime('%m/%d/%Y  %H:%M:%S.%f')
        output += f'# {hr_key}: {value}\n'
    return output

def validate_file_exists(path_to_file, file_type = 'interp'):
    """The function checks if the file exists or not. If exists, it adds an index to the file name.
    """
    if file_type == 'interp':
        prefix = 'r'
    elif file_type == 'bin':
        prefix = 'b'
    if os.path.isfile(path_to_file):
        (path, extension) = os.path.splitext(path_to_file)
        iterator = 2

        while True:

            new_filename = '{}-{}{:04d}{}'.format(path, prefix, iterator, extension)
            if not os.path.isfile(new_filename):
                return new_filename
            iterator += 1
    else:
        return path_to_file


def write_df_to_file(df, md):
    path_to_file, _ = os.path.splitext(md['interp_filename'])
    path_to_file = path_to_file + '.dat'

    path_to_file = validate_file_exists(path_to_file)
    comments = create_file_header(md)

    df = df[[c for c in df.columns if df[c].dtype == np.float64]]

    cols = df.columns.tolist()

    fmt = '%12.6f ' + (' '.join(['%12.6e' for i in range(len(cols) - 1)]))
    header = '  '.join(cols)

    # df = df[cols]
    np.savetxt(path_to_file,
               df.values,
               fmt=fmt,
               delimiter=" ",
               header=f'# {header}',
               comments=comments)
    call(['chmod', '774', path_to_file])
    return path_to_file