import numpy as np

from noge.data_types import GraphObservation, PartialGraphObservation


class OnlineGraphMemory:
    extra_keys = ['bf', 'df', 'nn']

    def __init__(self, max_nodes, max_edges, max_episode_steps, pos_label=1, neg_label=0):
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.dim_node = None

        self.num_nodes = 0
        self.num_edges = 0
        self.time = 0

        # data
        max_timesteps = min(max_nodes, max_episode_steps + 1)
        self.max_timesteps = max_timesteps
        self.visited_seq = np.empty(shape=(max_timesteps,), dtype=np.int64)
        self.edge_index = np.empty(shape=(2, max_edges), dtype=np.int64)
        self.frontier = []
        self.x = None
        self.cum_path_lengths = np.empty(shape=(max_timesteps,), dtype=np.float32)
        self.exploration_rates = np.empty(shape=(max_timesteps,), dtype=np.float32)

        # extra data
        self.store = {key: np.full(shape=(max_timesteps,), fill_value=-1, dtype=np.int64) for key in self.extra_keys}

    def clear(self):
        self.frontier.clear()
        self.x.fill(self.neg_label)
        self.num_nodes = 0
        self.num_edges = 0

    def update(self, partial_graph_obs: PartialGraphObservation):
        # extract time step
        t = partial_graph_obs['t']
        new_edges = partial_graph_obs['new_edges']  # array of shape [2, ΔΜ]
        new_nodes = partial_graph_obs['new_nodes']
        visited_node = partial_graph_obs['visited_node']
        cost = partial_graph_obs['path_cost']

        # reset and update frontier
        if t == 0:
            self.clear()
            self.frontier.extend(new_nodes)
            self.frontier.remove(visited_node)
            self.cum_path_lengths[t] = cost  # 0
            # self.exploration_rates[t] = 0
        else:
            self.frontier.remove(visited_node)
            self.frontier.extend(new_nodes)
            self.cum_path_lengths[t] = cost + self.cum_path_lengths[t-1]
            # self.exploration_rates[t] = t / self.cum_path_lengths[t]

        # update node and edge counts
        self.num_nodes += len(new_nodes)

        # update edges
        m_old = self.num_edges
        m_new = m_old + new_edges.shape[1]
        self.edge_index[:, m_old:m_new] = new_edges
        self.num_edges = m_new

        # update visited sequence
        self.visited_seq[t] = visited_node

        # update nn, bf, df
        for key in self.extra_keys:
            self.store[key][t] = partial_graph_obs[key]

        # update time
        self.time = t

    def get(self) -> GraphObservation:
        # retrieve graph state at time t
        t = self.time
        n = self.num_nodes
        m = self.num_edges

        obs = GraphObservation(x=self.x[:n],
                               edge_index=self.edge_index[:, :m],
                               frontier=np.array(self.frontier),
                               visited_seq=self.visited_seq[:t+1]
                               )
        return obs

    def get_frontier(self):
        return self.frontier


class OfflineGraphMemory:
    extra_keys = ['bf', 'df', 'nn']

    def __init__(self, max_nodes, max_edges, max_episode_steps, pos_label=1, neg_label=0):
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.dim_node = None

        # data
        max_timesteps = min(max_nodes, max_episode_steps + 1)
        self.max_timesteps = max_timesteps
        self.visited_seq = np.empty(shape=(max_timesteps,), dtype=np.int64)
        self.edge_index = np.empty(shape=(2, max_edges), dtype=np.int64)

        # time -> counts for reconstructing history
        self.node_counts = np.empty(shape=(max_timesteps,), dtype=np.int64)
        self.edge_counts = np.empty(shape=(max_timesteps,), dtype=np.int64)
        self.cum_path_lengths = np.empty(shape=(max_timesteps,), dtype=np.float32)
        self.exploration_rates = np.empty(shape=(max_timesteps,), dtype=np.float32)

        # node -> time step for reconstructing history (each node is discovered once and visited once)
        self.discovery_times = np.full(shape=(max_nodes,), fill_value=max_timesteps + 1, dtype=np.int64)
        self.visit_times = np.full(shape=(max_nodes,), fill_value=max_timesteps + 1, dtype=np.int64)

        # extra data
        self.store = {key: np.full(shape=(max_timesteps,), fill_value=-1, dtype=np.int64) for key in self.extra_keys}

    def update(self, partial_graph_obs: PartialGraphObservation):
        # extract time step
        t = partial_graph_obs['t']
        new_edges = partial_graph_obs['new_edges']  # array of shape [2, ΔΜ]
        discovered_nodes = partial_graph_obs['new_nodes']
        visited_node = partial_graph_obs['visited_node']
        cost = partial_graph_obs['path_cost']

        if t == 0:
            n_old = 0
            m_old = 0
            self.cum_path_lengths[t] = cost
            # self.exploration_rates[t] = 0
        else:
            n_old = self.node_counts[t-1]
            m_old = self.edge_counts[t-1]
            self.cum_path_lengths[t] = cost + self.cum_path_lengths[t-1]
            # self.exploration_rates[t] = t / self.cum_path_lengths[t]

        n_new = n_old + len(discovered_nodes)
        m_new = m_old + new_edges.shape[1]

        # update counts history
        self.edge_counts[t] = m_new
        self.node_counts[t] = n_new

        # update edges
        self.edge_index[:, m_old:m_new] = new_edges

        # update visited sequence
        self.visited_seq[t] = visited_node

        # update nn, bf, df
        for key in self.extra_keys:
            self.store[key][t] = partial_graph_obs[key]

        # update times
        self.discovery_times[discovered_nodes] = t  # time step the nodes were discovered
        self.visit_times[visited_node] = t  # time step the nodes were visited

    def sample(self, t: int) -> GraphObservation:
        raise NotImplementedError
