import numpy as np

def combine_classes(target_list, pred_arr):
    '''
    target_list: the list of intergers of desired target class e.g. [1, 8]
    pred_arr: np.array, an array of the values from prediction
    '''
    new_num = max(target_list)-1
    new_arr = np.where(np.isin(pred_arr, target_list), pred_arr, new_num)
    return new_arr
