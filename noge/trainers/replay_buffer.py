import pprint
import numpy as np
import gym

from noge.data_types import Transition, ReplayBatch
from noge.trainers.graphs_buffer import GraphsBuffer


class Replay:

    def __init__(self, capacity, ob_space, graph_mem_config, min_horizon=1):
        assert isinstance(ob_space, gym.spaces.Dict)
        assert 'meas' in ob_space.spaces

        self.capacity = capacity
        self.ob_space = ob_space
        self.min_horizon = min_horizon

        # make normal numpy arrays for measurements and goals
        meas_space = ob_space.spaces['meas']
        self._measurements = np.empty(shape=(capacity, *meas_space.shape), dtype=meas_space.dtype)

        # observations have a complex dynamic structure. Contain views into numpy arrays created in the environment.
        # make a list of dicts for the other obs spaces
        # self._obs = [None for _ in range(capacity)]
        self._graphs_buffer = GraphsBuffer(capacity, graph_mem_config)

        print(f"Replay buffer:\nOb spaces:\n")
        pprint.pprint(ob_space.spaces)
        print()

        self._actions = np.empty(capacity, dtype=np.int64)
        self._rewards = np.empty(capacity, dtype=np.float32)
        self._terminals = np.empty(capacity, dtype=np.float32)
        self._timesteps = np.empty(capacity, dtype=np.int64)
        self._episode_idx = np.empty(capacity, dtype=np.int64)

        # book keeping
        self._load = 0
        self._pointer = 0
        self._current_episode = 0

    def push(self, transition: Transition):

        obs = transition.obs
        self._graphs_buffer.push(self._current_episode, obs)
        self._measurements[self._pointer] = obs['meas']
        self._actions[self._pointer] = transition.action
        self._rewards[self._pointer] = transition.reward
        self._terminals[self._pointer] = 0
        self._timesteps[self._pointer] = obs['t']
        self._episode_idx[self._pointer] = self._current_episode

        # NOTE: If episodes are terminated early (before done is True),
        # training data will be incorrect, namely, future time steps
        # from a different episode will be seen as from the same episode.
        # Episodes end when either
        #   A) the frontier is empty (the graph is fully explored)
        #   B) a time limit has been reached
        if transition.terminal:
            # we must add the final observation in the next slot
            self._pointer = (self._pointer + 1) % self.capacity
            self._load = min(self._load + 1, self.capacity)

            last_obs = transition.next_obs
            self._graphs_buffer.push(self._current_episode, last_obs)
            self._measurements[self._pointer] = last_obs['meas']
            # there are no actions or rewards at the last time step
            self._terminals[self._pointer] = 1
            self._timesteps[self._pointer] = last_obs['t']
            self._episode_idx[self._pointer] = self._current_episode

            # increment episode id
            self._current_episode += 1

        self._pointer = (self._pointer + 1) % self.capacity
        self._load = min(self._load + 1, self.capacity)

    def sample(self, batch_size: int) -> ReplayBatch:

        valid_idx = self._sample_valid_indices(batch_size)
        next_idx = (valid_idx + 1) % self.capacity

        # retrieve / copy from storage
        # obses = [self._obs[i] for i in valid_idx]     # list of dicts
        graph_obses = self._graphs_buffer.get(valid_idx)
        measurements = self._measurements[valid_idx]
        rewards = self._rewards[valid_idx]
        actions = self._actions[valid_idx]

        # next timestep
        next_graph_obses = self._graphs_buffer.get(next_idx)
        next_measurements = self._measurements[next_idx]
        terminals = self._terminals[next_idx]

        # make replay batch according to data protocol
        sample_batch = ReplayBatch(graph_obses=graph_obses,
                                   measurements=measurements,
                                   actions=actions,
                                   rewards=rewards,
                                   next_graph_obses=next_graph_obses,
                                   next_measurements=next_measurements,
                                   mask=1-terminals
                                   )
        return sample_batch

    def _sample_valid_indices(self, batch_size: int) -> np.ndarray:
        """ Sample transitions with enough future time steps """

        valid_idx = []
        num_valid_idx = 0
        while num_valid_idx < batch_size:

            # sample random indices and get their episode ids
            idx = np.random.randint(self._load, size=batch_size)
            episode_idx = self._episode_idx[idx]

            # consider the indices at t + min_horizon
            min_horizon_idx = (idx + self.min_horizon) % self.capacity
            min_horizon_episode_idx = self._episode_idx[min_horizon_idx]

            # check 1: future meas must be in the same episode
            valid_samples_mask = episode_idx == min_horizon_episode_idx

            # check 2: episode time step must be > 0 (in the first time step is there is no useful bias)
            timestep_mask = self._timesteps[idx] > 0
            valid_samples_mask &= timestep_mask

            valid_samples = idx[valid_samples_mask]

            valid_idx.append(valid_samples)
            num_valid_idx += len(valid_samples)

        valid_idx = np.concatenate(valid_idx)

        return valid_idx[:batch_size]

    def compute_moments(self):
        measurements = self._measurements
        meas_mu = np.mean(measurements, 0)
        meas_std = np.std(measurements, 0)
        meas_std[meas_std == 0] = 1  # avoid divisions by zero
        return meas_mu, meas_std

    def __len__(self) -> int:
        return self._load

    def is_full(self) -> bool:
        return self._load == self.capacity
