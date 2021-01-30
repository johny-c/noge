import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, FunctionTransformer, PowerTransformer

from noge.preprocessors.input_meas import Meas, MeasurementSpace, MEAS_FLAGS


class TargetPreprocessor:

    def __init__(self, target_type: str, temporal_offsets: np.ndarray, target_transform: str, max_nodes: int, device):
        assert all([m in MEAS_FLAGS for m in target_type])
        assert target_transform in ('normal', 'lognormal', 'identity', 'minmax')

        self.target_type = target_type
        self.temporal_offsets = np.asarray(temporal_offsets, dtype=np.float32)
        self.max_nodes = max_nodes
        self.target_transform = target_transform

        self._meas_idx = [MEAS_FLAGS[key].value for key in target_type]

        name = self.__class__.__name__
        print(f"[{name}]: Using target measurements {target_type} --> indices: {self._meas_idx}")
        d = len(self._meas_idx)
        self.dim_output_meas = d

        if target_transform == 'normal':
            self.target_scaler = StandardScaler(with_mean=False)
        elif target_transform == 'lognormal':
            self.target_scaler = PowerTransformer(method='box-cox', standardize=False)
        elif target_transform == 'identity':
            self.target_scaler = FunctionTransformer()
        elif target_transform == 'minmax':
            lows = -np.ones(shape=(d,), dtype=np.float32)
            highs = np.ones(shape=(d,), dtype=np.float32)
            target = MeasurementSpace(lows, highs)
            func = lambda x: (x - target.mid) / target.range
            inverse_func = lambda x: x * target.range + target.mid
            self.target_scaler = FunctionTransformer(func=func, inverse_func=inverse_func)

        self.device = device

    def fit(self, data):

        t = data[:, Meas.TIMESTEP]
        mask = t > 0
        data = data[mask]

        name = self.__class__.__name__
        print(f"[{name}]: Measurements\nmin={np.min(data, 0)}\nmax={np.max(data, 0)}\nstd={np.std(data, 0)}")

        meas = data[:, self._meas_idx]  # [N, D]
        meas_all = np.tile(meas, len(self.temporal_offsets))  # [N, T*D]
        self.target_scaler.fit(meas_all)

    def transform(self, target: np.ndarray, target_mask: np.ndarray):
        """ Transform targets sampled from the replay buffer

        :param target: array, shape [B, T, D], measurement differences
        :param target_mask: array, shape [B, T, D], boolean array indicating valid targets
        :param to_torch: bool, whether to transform to torch or not
        :return: tuple of arrays
        """
        B, T, D = target.shape

        # [B, T, d]
        x = target[:, :, self._meas_idx]
        m = target_mask[:, :, self._meas_idx]

        # replace invalid values with a valid value that can be target_scaled without raising errors
        x[~m] = 1

        # reshape x from [B, T, d] to [B, T*d]
        d = len(self._meas_idx)
        x = x.reshape(B, T*d)
        xt = self.target_scaler.transform(x)

        # [B, T, D=1]
        xt = xt.reshape(B, T, d)
        mt = m.reshape(B, T, d)

        target = torch.from_numpy(xt).to(self.device)
        target_mask = torch.from_numpy(mt).to(self.device)

        return target, target_mask

    def inverse_transform(self, prediction: torch.FloatTensor):
        """ Transform predictions of neural net to real range values

        :param prediction: tensor, shape [B, T*D], float values
        :return: tensor, shape [B, T*D], float values
        """

        p = prediction.cpu().numpy()
        y = self.target_scaler.inverse_transform(p)
        y = torch.from_numpy(y).to(prediction.device)

        return y
