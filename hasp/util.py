import numpy as np


def combine_classes(target_list, pred_arr):
    '''
    target_list: the list of intergers of desired target class e.g. [1, 8]
    pred_arr: np.array, an array of the values from prediction
    '''
    new_num = max(target_list)-1
    new_arr = np.where(np.isin(pred_arr, target_list), pred_arr, new_num)
    return new_arr


def np_pad_wrapper(arrays, max_len=64000):
    '''
    Pad array or list of arrays a list of arrays into a regular 2d array padded with -2
    arrays: list of arrays or single array
    max_len: number of columns after padding
    '''
    if len(array) > 1:
        return np.vstack(
            tuple(np.pad(arr,
                         pad_width=(0, max_len - len(arr)),
                         constant_values=-2) for arr in arrays)
        )
    else:
        arr = arrays
        return np.pad(arr, pad_width=(0, max_len - len(arr)), constant_values=-2)
