from audiomentations import Compose, AddGaussianNoise, TimeStretch, Shift,\
                            PitchShift, AirAbsorption
import numpy as np

def augment_with_method(samples: np.array, sample_rate: int, method: str, **kwargs):

    method_dict = {
        "AddGaussianNoise": AddGaussianNoise,
        "AirAbsorption": AirAbsorption,
        "TimeStretch": TimeStretch,
        "PitchShift": PitchShift,
        "Shift": Shift
    }

    if method in method_dict:
        augment = Compose([
            method_dict[method](**kwargs)
        ])
    else:
        raise ValueError(f"The augmentation method {method} is not defined \
                         in argument_with_method().")

    return augment(samples=samples, sample_rate=sample_rate)
