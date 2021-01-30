import numpy as np
from typing import List

from noge.data_types import GraphObservation, GoalPartialObservation
from noge.factory import make_memory


class GraphsBuffer:

    def __init__(self, capacity, graph_mem_config):

        self.capacity = capacity
        self.graph_mem_config = graph_mem_config

        self._current_graph = make_memory(**graph_mem_config)

        # the store will be a list of **references** to DynamicGraphMemory objects
        # NOTE: deque has O(N) runtime for random access, list has O(1)
        self._store = [None for _ in range(capacity)]
        self._store[0] = self._current_graph

        # book keeping
        self._index_to_timestep = np.empty(capacity, dtype=np.int64)
        self._current_episode_id = 0
        self._current_index = 0

    def push(self, episode_id: int, partial_graph_obs: GoalPartialObservation):

        i = self._current_index
        e = self._current_episode_id
        t = partial_graph_obs['t']

        if episode_id == e:
            # update existing graph memory
            self._current_graph.update(partial_graph_obs)
            self._index_to_timestep[i] = t
        else:
            # reset: create a new graph memory
            self._current_episode_id = episode_id
            self._current_graph = make_memory(**self.graph_mem_config)
            self._current_graph.update(partial_graph_obs)
            self._index_to_timestep[i] = 0

        self._store[i] = self._current_graph  # this is just a reference (no copy)
        self._current_index = (self._current_index + 1) % self.capacity

    def get(self, indices) -> List[GraphObservation]:
        """ We need to have a mapping from indices [0, capacity) to samples
            We can have an indirect mapping:
                From indices to (episodes, timesteps) pairs.
                indices to episode idx exists (DFP needs it). We only need an indices to timesteps mapping.

        :param indices: array of integers
        :return:
        """
        timesteps = self._index_to_timestep[indices]
        samples = [self._store[i].sample(t) for i, t in zip(indices, timesteps)]

        return samples
