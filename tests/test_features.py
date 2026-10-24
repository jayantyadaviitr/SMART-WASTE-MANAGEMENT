import numpy as np

from m1_audio.features import FEATURE_NAMES, SR, extract_features
from m1_audio.synth_samples import CLASSES, _one_clip


def test_feature_vector_length():
    rng = np.random.default_rng(0)
    y = _one_clip(CLASSES["metal"], rng)
    vec = extract_features(y, sr=SR)
    assert vec.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(vec).all()


def test_metal_peak_higher_than_wet():
    rng = np.random.default_rng(1)
    metal = extract_features(_one_clip(CLASSES["metal"], rng))
    wet = extract_features(_one_clip(CLASSES["wet"], rng))
    assert metal[0] > wet[0]
