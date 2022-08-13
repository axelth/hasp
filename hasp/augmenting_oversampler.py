from numbers import Real
from collections.abc import Mapping

import numpy as np
from scipy import sparse
from sklearn.utils import check_array, check_random_state
from sklearn.utils import _safe_indexing
from sklearn.utils.sparsefuncs import mean_variance_axis
from imblearn.over_sampling import RandomOverSampler


class AugmentingRandomOversampler(RandomOverSampler):

    def __init__(self, random_state=None, augment_method=None):
        super().__init__(self, sampling_strategy='minority',
                         random_state=random_state)
        self.augment_method = augment_method

    def _fit_resample(self, X, y):
        random_state = check_random_state(self.random_state)

        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        sample_indices = range(X.shape[0])
        for class_sample, num_samples in self.sampling_strategy_.items():
            target_class_indices = np.flatnonzero(y == class_sample)
            bootstrap_indices = random_state.choice(
                target_class_indices,
                size=num_samples,
                replace=True,
            )
            sample_indices = np.append(sample_indices, bootstrap_indices)
            
            # generate a bootstrap
            ## TODO augment
            # for arr in _safe_indexing(X, bootstrap_indices):
            #     X_resampled.append(
            #         self._augment_sample(arr)
            #     )
            
            X_resampled.append(_safe_indexing(X, bootstrap_indices))
            y_resampled.append(_safe_indexing(y, bootstrap_indices))

        self.sample_indices_ = np.array(sample_indices)

        X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)

        return X_resampled, y_resampled

    def _augment_sample(self, samples):
        mask_padding = samples > -2

        return samples[mask_padding]
        
