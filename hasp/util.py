import numpy as np


def combine_classes(target_list, label_arr, binary=False):
    '''
    target_list: the list of intergers of desired target class e.g. [1, 8]
    pred_arr: np.array, an array of the values from prediction
    '''
    if binary:
        return np.where(np.isin(label_arr, target_list), 1, 0)
    else:
        new_num = max(target_list)-1
        new_arr = np.where(np.isin(label_arr, target_list), label_arr, new_num)
        return new_arr


def np_pad_wrapper(arrays, max_len=64000):
    '''
    Pad array or list of arrays a list of arrays into a regular 2d array padded with -2
    Arrays longer than max_len are truncated to max_len
    arrays: list of arrays or single array
    max_len: number of columns after padding
    '''
    if len(arrays) > 1:
        return np.vstack(
            tuple(np.pad(arr[:max_len],
                         pad_width=(0, max(max_len - len(arr), 0)),
                         constant_values=-2) for arr in arrays)
        )
    else:
        arr = arrays
        return np.pad(arr, pad_width=(0, max_len - len(arr)), constant_values=-2)
