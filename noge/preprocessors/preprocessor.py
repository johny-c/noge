import numpy as np

from .input_meas import InputMeasPreprocessor
from .target_preprocessor import TargetPreprocessor


class Preprocessor:
    """ Preprocess measurement and target vectors before feeding to neural net. """

    def __init__(self, input_meas_type, output_meas_type, feature_range, meas_transform, target_transform,
                 temporal_offsets, max_nodes, device):
        assert len(feature_range) == 2
        assert feature_range[0] < feature_range[1]

        self.meas_transform = meas_transform
        self.target_transform = target_transform
        self.temporal_offsets = np.asarray(temporal_offsets, dtype=np.float32)
        self.max_nodes = max_nodes
        self.device = device

        self.input_meas_pp = InputMeasPreprocessor(input_meas_type, meas_transform, feature_range, max_nodes, device)
        self.target_pp = TargetPreprocessor(output_meas_type, temporal_offsets, target_transform, max_nodes, device)

    @property
    def dim_input_meas(self):
        return self.input_meas_pp.dim_input_meas

    @property
    def dim_output_meas(self):
        return self.target_pp.dim_output_meas

    def fit(self, measurements):
        self.input_meas_pp.fit(measurements)
        self.target_pp.fit(measurements)

    def transform_meas(self, meas: np.ndarray):
        return self.input_meas_pp.transform(meas)      # [B, D1]

    def transform_target(self, target: np.ndarray, target_mask: np.ndarray):
        """ Transform targets sampled from the replay buffer"""
        return self.target_pp.transform(target, target_mask)

    def transform_prediction(self, prediction):
        return self.target_pp.inverse_transform(prediction)
