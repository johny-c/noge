import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer, PowerTransformer
from enum import IntEnum


class MeasurementSpace:
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.mid = (low + high) / 2
        self.range = high - low
        self.dim = len(low)


class Meas(IntEnum):
    TIMESTEP = 0
    PATH_LENGTH = 1
    NUM_NODES = 2
    NUM_EDGES = 3
    EXPLORATION_RATE = 4        # R
    NODE_COVERAGE = 5          # N
    EDGE_COVERAGE = 6          # E
    SPARSITY = 7                # S
    PATH_LENGTH_OVER_TIME = 8   # L


MEAS_FLAGS = {'R': Meas.EXPLORATION_RATE,
              'N': Meas.NODE_COVERAGE,
              'E': Meas.EDGE_COVERAGE,
              'S': Meas.SPARSITY,
              'L': Meas.PATH_LENGTH_OVER_TIME
              }


class InputMeasPreprocessor:

    def __init__(self, meas_type: str, meas_transform: str, feature_range, max_nodes, device):
        assert all([m in MEAS_FLAGS for m in meas_type])
        assert meas_transform in ('normal', 'lognormal', 'minmax', 'data_minmax', 'identity')

        self.meas_type = meas_type
        self.meas_transform = meas_transform
        self.max_nodes = max_nodes
        self.device = device

        name = self.__class__.__name__
        self._meas_idx = [MEAS_FLAGS[key].value for key in meas_type]
        self.dim_input_meas = len(self._meas_idx)
        print(f"[{name}]: Using input measurements {meas_type} --> indices: {self._meas_idx}")

        if meas_transform == 'data_minmax':
            self.meas_scaler = MinMaxScaler(feature_range=feature_range)
        elif meas_transform == 'minmax':
            d = self.dim_input_meas

            if meas_type == 'R':
                lows = np.zeros(shape=(d,), dtype=np.float32)
                highs = np.ones(shape=(d,), dtype=np.float32)
            elif meas_type == 'L':
                lows = np.ones(shape=(d,), dtype=np.float32)
                highs = np.ones(shape=(d,), dtype=np.float32) * (max_nodes - 1)
            else:
                raise ValueError(f"Supported are only {MEAS_FLAGS.keys()} meas diffs.")

            data = np.row_stack((lows, highs))
            self.meas_scaler = MinMaxScaler(feature_range=feature_range)
            self.meas_scaler.fit(data)
        elif meas_transform == 'normal':
            self.meas_scaler = StandardScaler()
        elif meas_transform == 'lognormal':
            self.meas_scaler = PowerTransformer(method='box-cox')
        elif meas_transform == 'identity':
            self.meas_scaler = FunctionTransformer(func=None, inverse_func=None)

    def fit(self, data):

        if self.meas_transform == 'minmax':
            # minmax scaler is fit on [[0, 0, 0], [1, 1, 1]]
            return

        t = data[:, Meas.TIMESTEP]
        valid_data = data[t > 0]
        meas = valid_data[:, self._meas_idx]
        self.meas_scaler.fit(meas)

    def transform(self, meas: np.ndarray):
        """ meas = [er, er_diff, sign_er_diff, path_length, time_step]

        :param meas: np.ndarray of shape [B, D_in]
        :return: torch.Tensor of shape [B, D_out]
        """
        if meas.ndim == 1:  # online: [1, d]
            meas = meas.reshape(1, len(meas))

        meas_t = meas[:, self._meas_idx]

        # transform
        meas_t = self.meas_scaler.transform(meas_t)

        # torchify
        return torch.from_numpy(meas_t).to(self.device)
