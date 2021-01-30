import numpy as np
from gym.spaces import Dict, Box, Discrete

from noge.data_types import PartialGraphObservation
from .base_graph_env import BaseGraphEnv, StepEvents


class OnlineGraphEnv(BaseGraphEnv):

    def __init__(self, max_episode_steps, reward_type='path_length', data_generator=None,
                 max_nodes=None, max_edges=None, nn_feat=False):
        super().__init__(max_episode_steps=max_episode_steps, reward_type=reward_type, data_generator=data_generator)

        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.nn_feat = nn_feat

        # online state (for more efficient feeding to torch geometric)
        self._idx_to_node = np.empty(shape=(max_nodes,), dtype=np.int64)  # map online nodes to their original names
        self._node_to_idx = None  # map discovered nodes to online indices {0, 1, ..., N-1}
        self._online_frontier = None  # the frontier nodes (online ids)

        # initialize these arrays only once, then overwrite in each episode
        max_visits = min(max_episode_steps + 1, max_nodes)
        self._node_counts = np.empty(shape=(max_visits,), dtype=np.int64)
        self._edge_counts = np.empty(shape=(max_visits,), dtype=np.int64)
        self._online_visit_sequence = np.empty(shape=(max_visits,), dtype=np.int64)
        self._online_discovery_sequence = np.empty(shape=(max_nodes,), dtype=np.int64)
        self._known_edge_index = np.empty(shape=(2, max_edges), dtype=np.int64)

        # observations
        num_output_meas = 9
        self.observation_space = Dict(dict(
            t=Discrete(n=max_episode_steps),
            new_nodes=Box(low=0, high=max_nodes, shape=(max_nodes,), dtype=np.int64),
            new_edges=Box(low=0, high=max_edges, shape=(2, max_edges), dtype=np.int64),
            visited_node=Discrete(n=max_nodes),
            nn=Discrete(n=max_nodes),
            bf=Discrete(n=max_nodes),
            df=Discrete(n=max_nodes),
            meas=Box(low=-np.inf, high=np.inf, shape=(num_output_meas,), dtype=np.float32)
        ))

        # actions
        self.action_space = Discrete(n=max_nodes)

    def reset(self, **kwargs):

        events: StepEvents = super().reset(**kwargs)

        m = len(events['new_edges'])

        # track the discovered data renaming nodes as {0,1,...,N}
        self._idx_to_node[0] = self._source
        self._idx_to_node[1:m+1] = events['new_nodes']
        self._node_to_idx = {x: i for i, x in enumerate(self._idx_to_node[:m+1])}

        online_frontier_list = list(range(1, m+1))
        self._online_frontier = online_frontier_list

        # track the explored nodes as visited_seq of shape [N]
        online_source = 0
        self._online_visit_sequence[0] = online_source

        # track discovery sequence (of sets)
        self._online_discovery_sequence[0] = online_source
        self._online_discovery_sequence[1:1+m] = online_frontier_list

        # track the discovered edges as edge_index of shape [2, E]
        self._known_edge_index[0, :m] = online_source
        self._known_edge_index[1, :m] = online_frontier_list
        self._known_edge_index[0, m:2*m] = online_frontier_list
        self._known_edge_index[1, m:2*m] = online_source

        # reset counts
        self._node_counts[0] = m + 1
        self._edge_counts[0] = 2*m

        obs = self._get_partial_obs(events)
        return obs

    def _get_partial_obs(self, events: StepEvents) -> PartialGraphObservation:

        t = self._time_step

        if t > 0:
            n_old = self._node_counts[t-1]
            m_old = self._edge_counts[t-1]
        else:
            n_old = 0
            m_old = 0

        # new nodes
        n_new = self._node_counts[t]
        new_nodes = self._online_discovery_sequence[n_old:n_new].copy()

        # new edges
        m_new = self._edge_counts[t]
        new_edges = self._known_edge_index[:, m_old:m_new].copy()

        # visited
        visited_node = self._online_visit_sequence[t]

        if len(self._online_frontier):
            if self.nn_feat:
                nn = self.nearest_neighbor_policy()
            else:
                nn = -1

            bf = self._online_frontier[0]
            df = self._online_frontier[-1]
        else:
            nn = bf = df = -1

        path_nodes = np.array([self._node_to_idx[v] for v in events['path']], dtype=np.int64)
        delta_obs = PartialGraphObservation(t=t,
                                            new_nodes=new_nodes,
                                            new_edges=new_edges,
                                            visited_node=visited_node,
                                            path_nodes=path_nodes,
                                            path_cost=events['cost'],
                                            nn=nn,
                                            bf=bf,
                                            df=df,
                                            meas=self.get_current_measurements()
                                            )

        return delta_obs

    def nearest_neighbor_policy(self):
        v = super().nearest_neighbor_policy()
        return self._node_to_idx[v]

    def bfs_policy(self):
        return self._online_frontier[0]

    def dfs_policy(self):
        return self._online_frontier[-1]

    def random_policy(self):
        i = self.rng.randint(len(self._online_frontier))
        return self._online_frontier[i]

    def step(self, action):
        offline_subgoal = self._idx_to_node[action]
        events, reward, done, info = super().step(offline_subgoal)

        t = self._time_step  # time step > 0 (has been incremented in BaseGraphEnv)

        online_subgoal = action  # action = online node index = integer in [0, ..., N-1]
        self._online_frontier.remove(online_subgoal)
        self._online_visit_sequence[t] = online_subgoal

        # track discovered edges online indices
        num_discovered_edges = len(events['new_edges'])
        self._edge_counts[t] = self._edge_counts[t-1] + num_discovered_edges * 2

        new_nodes = events['new_nodes']
        num_discovered_nodes = len(new_nodes)
        self._node_counts[t] = self._node_counts[t-1] + num_discovered_nodes

        if num_discovered_edges > 0:
            # NOTE: Remember that events.discovered_frontier_neighbors is a set, not a list
            discovered_frontier_nodes_oidx = [self._node_to_idx[v] for v in events['found_frontier_nodes']]

            if num_discovered_nodes > 0:
                # track discovered nodes online indices
                n_prev = self._node_counts[t-1]
                n_curr = self._node_counts[t]
                new_nodes_oidx = list(range(n_prev, n_curr))
                self._node_to_idx.update(dict(zip(new_nodes, new_nodes_oidx)))
                self._idx_to_node[n_prev:n_curr] = new_nodes

                # update online node discovery sequence
                self._online_discovery_sequence[n_prev:n_curr] = new_nodes_oidx
                discovered_ends_oidx = new_nodes_oidx + discovered_frontier_nodes_oidx

                # update online frontier
                self._online_frontier.extend(new_nodes_oidx)

            else:
                discovered_ends_oidx = discovered_frontier_nodes_oidx

            # add both directions for undirected graph (symmetric adjacency)
            a = self._edge_counts[t-1]
            b = a + num_discovered_edges
            c = b + num_discovered_edges
            self._known_edge_index[0, a:b] = online_subgoal
            self._known_edge_index[1, a:b] = discovered_ends_oidx
            self._known_edge_index[0, b:c] = discovered_ends_oidx
            self._known_edge_index[1, b:c] = online_subgoal

        obs = self._get_partial_obs(events)

        return obs, reward, done, info

    def get_current_measurements(self):
        """Always return the cumulative path length.
           Conversion to other measurements will happen at some later stage of the algorithm"""

        t = self._time_step
        C = t + 1
        L = self._total_path_cost
        V = self._node_counts[t]
        E = self._edge_counts[t]

        exp_rate = self._exploration_rates[t]  # t / L_t
        plot = 0 if t == 0 else L / t
        node_discovery = C / V
        edge_discovery = C / E
        sparsity = (V - 1) / E

        meas = np.array([t, L, V, E, exp_rate, node_discovery, edge_discovery, sparsity, plot], dtype=np.float32)

        return meas
