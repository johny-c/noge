import torch
from collections import deque

from noge.data_types import GraphObservation, NeuralGraphObservation
from xlog.utils import collapse_dicts


class Trainer:
    def __init__(self, replay_buffer, batch_size, network, preprocessor, criterion,
                 optimizer, scheduler=None, device='cpu'):
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.network = network
        self.preprocessor = preprocessor
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)

        self.train_metrics = deque(maxlen=10)
        self.num_train_steps = 0

    def graph_obs_fn(self, graph_obs: GraphObservation) -> NeuralGraphObservation:
        torch_graph_obs = {k: torch.from_numpy(v).to(self.device) for k, v in graph_obs.items()}
        return NeuralGraphObservation(**torch_graph_obs)

    def preprocess_batch(self, batch):
        raise NotImplementedError

    def step(self, batch_size=None):
        raise NotImplementedError

    @property
    def counters(self):
        return dict(train_steps=self.num_train_steps)

    def gather_metrics(self):
        return collapse_dicts(self.train_metrics)
