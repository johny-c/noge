import numpy as np

from noge.data_types import GoalPartialObservation, GraphObservation
from .base_memory import OnlineGraphMemory, OfflineGraphMemory


class CategoricalOnlineMemory(OnlineGraphMemory):
    """ Maintain a node features array that is incrementally updated (no copy) """

    def __init__(self, max_nodes, max_edges, max_episode_steps, history, features, pos_label, neg_label):
        super().__init__(max_nodes, max_edges, max_episode_steps, pos_label, neg_label)

        self.history = history
        self.cat_features = features

        # features 'CYF' + cat_features = {B, D, N}
        self.feat_flags = {key: flag in features for flag, key in zip("BDN", self.extra_keys)}
        self.num_features = 3 + sum(self.feat_flags.values())

        self._last_path_nodes = None
        self.dim_node = history * self.num_features
        self.x = np.full(shape=(max_nodes, self.dim_node), fill_value=self.neg_label, dtype=np.float32)
        self.offsets = list(range(0, self.dim_node, history))

    def update(self, partial_graph_obs: GoalPartialObservation):
        n_old = self.num_nodes

        super().update(partial_graph_obs)

        n_new = self.num_nodes

        visited_node = partial_graph_obs['visited_node']

        # update node features in place
        pos = self.pos_label
        neg = self.neg_label
        x = self.x
        t = self.time
        d = self.num_features
        D = self.dim_node  # D = d * h

        # for the previously known nodes, shift all features 1 position to the left
        dims_past = D - d
        x[:n_old, :dims_past] = x[:n_old, d:]

        # in the present, the changes are:
        # C Y F B D N

        # C: visited node is now in the visited set
        col = dims_past
        if t > 0:
            previous_node = self.visited_seq[t - 1]
            x[previous_node, col] = pos

        if 'X' not in self.cat_features:
            x[visited_node, col] = pos

        # Y: visited node is current, previous node is not current
        col += 1
        if t > 0:
            previous_node = self.visited_seq[t-1]
            x[previous_node, col] = neg
        x[visited_node, col] = pos

        # F: visited node is no longer in the frontier, new nodes are
        col += 1
        x[n_old:n_new, col] = pos
        x[visited_node, col] = neg

        # BDN
        for key in self.extra_keys:
            if self.feat_flags[key]:
                arr = self.store[key]
                col += 1
                if t > 0:
                    previous_node = arr[t-1]
                    x[previous_node, col] = neg

                current_node = arr[t]
                if current_node >= 0:  # -1 means None
                    x[current_node, col] = pos


class CategoricalOfflineMemory(OfflineGraphMemory):
    """Maintain events history, so that graph state of any time step can be reconstructed"""

    def __init__(self, max_nodes, max_edges, max_episode_steps, history, features, pos_label, neg_label):

        super().__init__(max_nodes, max_edges, max_episode_steps, pos_label, neg_label)

        self.history = history
        self.cat_features = features

        # features 'CYF' + cat_features = {B, D, N}
        self.feat_flags = {key: flag in features for flag, key in zip("BDN", self.extra_keys)}
        self.num_features = 3 + sum(self.feat_flags.values())
        self.list_of_path_nodes = [None] * (max_episode_steps + 1)
        self.dim_node = history * self.num_features

    def update(self, partial_graph_obs: GoalPartialObservation):
        super().update(partial_graph_obs)

    def sample(self, t: int) -> GraphObservation:
        # retrieve graph state at time t
        m = self.edge_counts[t]
        n = self.node_counts[t]

        discovery_times = self.discovery_times[:n]
        visit_times = self.visit_times[:n]

        # get nodes discovered time step
        discovery_times = discovery_times.reshape(n, 1)
        # get nodes visited time step
        visitation_times = visit_times.reshape(n, 1)

        # update node features in place
        pos = self.pos_label
        neg = self.neg_label
        h = self.history
        d = self.num_features
        D = self.dim_node  # D = d * h

        # time steps
        timesteps = np.arange(t-h+1, t+1)  # [t-3, t-2, t-1, t] for h=4

        # discovered feature (v in U)
        discovered_mask = discovery_times <= timesteps  # [N, H]
        # visited feature (v in C)
        visited_mask = visitation_times <= timesteps  # [N, H]
        # frontier feature (v in F) = discovered but not (yet) visited
        frontier_mask = discovered_mask & ~visited_mask  # [N, H]
        # current feature (v == v_t)
        # current_mask = visitation_times == timesteps  # [N, H]

        # F C Y B D N
        x = np.full(shape=(n, D), fill_value=neg, dtype=np.float32)

        for j in range(h):
            timestep = t-h+1+j

            if timestep >= 0:

                # C: visited set
                col = j * d  # [0, d, 2d]
                M = visited_mask[:, j]
                x[M, col] = pos

                # Y: current node
                col += 1
                # M = current_mask[:, j]
                v = self.visited_seq[timestep]
                x[v, col] = pos

                if 'X' in self.cat_features:
                    x[v, col-1] = neg

                # F: frontier set
                col += 1
                M = frontier_mask[:, j]
                x[M, col] = pos

                # BDN
                for key in self.extra_keys:
                    if self.feat_flags[key]:
                        arr = self.store[key]
                        col += 1
                        current_node = arr[timestep]
                        if current_node >= 0:  # -1 means None
                            x[current_node, col] = pos

        # frontier
        frontier = np.where(frontier_mask[:, -1])[0]

        obs = GraphObservation(x=x,
                               edge_index=self.edge_index[:, :m],
                               frontier=frontier,
                               visited_seq=self.visited_seq[:t+1]
                               )
        return obs
