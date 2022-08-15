
import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils import _safe_indexing
from imblearn.over_sampling import RandomOverSampler

from hasp.augment_audio import augment_with_method
from hasp.util import np_pad_wrapper


class AugmentingRandomOversampler(RandomOverSampler):

    def __init__(self, random_state=None, augment_method=None, **kwargs):
        super().__init__(sampling_strategy='minority',
                         random_state=random_state)
        self.augment_method = augment_method
        self.kwargs = kwargs

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

            # generate a bootstrap augment
            X_resampled.append(
                np_pad_wrapper(
                     [
                         self._augment_sample(arr)
                         for arr in _safe_indexing(X, bootstrap_indices)
                     ]
                )
            )

            y_resampled.append(_safe_indexing(y, bootstrap_indices))

        self.sample_indices_ = np.array(sample_indices)

        X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)

        return X_resampled, y_resampled

    def _augment_sample(self, samples):
        mask_padding = samples > -2

        return augment_with_method(samples[mask_padding], 16000,
                                   self.augment_method, **self.kwargs)

