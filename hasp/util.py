import numpy as np

def combine_classes(target_list, pred_arr):
    '''
    target_list: the list of intergers of desired target class e.g. [1, 8]
    pred_arr: np.array, an array of the values from prediction
    '''
    new_arr = np.where(np.isin(pred_arr, target_list), pred_arr, 10)
    return new_arr
