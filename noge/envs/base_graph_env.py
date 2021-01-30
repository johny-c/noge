import gym
import time
import numpy as np
import networkx as nx
from itertools import chain
from typing import TypedDict, List, Set, Union


class StepEvents(TypedDict):
    new_nodes: List
    new_edges: List
    found_frontier_nodes: Union[Set, type(None)]
    previous_node: Union[int, type(None)]
    current_node: int
    path: Union[List, type(None)]
    cost: float


class BaseGraphEnv(gym.Env):

    def __init__(self, max_episode_steps, reward_type='path_length', data_generator=None):

        assert reward_type in ('path_length', 'er_diff', 'sign_er_diff', 'edges-path', 'nodes-path')

        self.max_episode_steps = max_episode_steps
        self.reward_type = reward_type
        self.data_generator = data_generator
        self.rng = None

        # tracking of agent - graph state
        self._graph = None
        self._source = None
        self._maze = None
        self._known_graph = None
        self._current_node = None
        self._previous_node = None
        self._time_step = 0
        self._frontier = None
        self._frontier_set = None

        # track metrics
        self._episode_return = 0
        self._episode_start_time = 0
        self._num_total_nodes = 0
        self._num_total_edges = 0
        self._total_path_cost = 0
        self._total_path_edges = 0

        # initialize these arrays only once, then overwrite in each episode
        self._visit_sequence = np.empty(max_episode_steps + 1, dtype=np.int64)
        self._path_lengths = np.empty(max_episode_steps + 1, dtype=np.float32)
        self._cum_path_lengths = np.empty(max_episode_steps + 1, dtype=np.float32)
        self._exploration_rates = np.empty(max_episode_steps + 1, dtype=np.float32)
        self._node_coverages = np.empty(max_episode_steps + 1, dtype=np.float32)
        self._path_lengths_over_time = np.empty(max_episode_steps + 1, dtype=np.float32)

        self.action_space = gym.spaces.Discrete(n=1)  # n_t

    def seed(self, seed=None):
        self.rng = np.random.RandomState(seed)

    def reset(self, graph=None, source=None, maze=None):

        # initialize state
        if graph is None:
            sample = next(self.data_generator)
            graph = sample['graph']
            source = sample['source']
            maze = sample.get('maze')

        self._graph = graph
        self._source = source
        self._maze = maze
        self._known_graph = nx.Graph()
        self._current_node = source
        self._previous_node = None
        self._time_step = 0

        # populate frontier with starting node's neighbors
        neighbors = list(self._graph[source])
        m = len(neighbors)
        self.rng.shuffle(neighbors)  # avoid any pre-existing node ordering bias
        # self._frontier.clear()
        self._frontier = neighbors
        self._frontier_set = set(neighbors)
        self._num_total_nodes = m + 1
        self._num_total_edges = m
        self.action_space.n = m

        # populate node sequence
        self._visit_sequence[0] = source

        # populate measurements
        self._path_lengths[0] = 0
        self._cum_path_lengths[0] = 0
        self._exploration_rates[0] = 0
        self._path_lengths_over_time[0] = 0
        self._node_coverages[0] = 1 / (m + 1)

        # populate known graph with starting node and its edges
        discovered_edges = [(source, v) for v in neighbors]
        self._known_graph.add_edges_from(discovered_edges)

        # track metrics
        self._episode_return = 0
        self._episode_start_time = time.time()
        self._total_path_cost = 0
        self._total_path_edges = 0

        # return initial observation
        obs = StepEvents(
            new_nodes=neighbors,
            found_frontier_nodes=None,
            new_edges=discovered_edges,
            previous_node=None,
            current_node=source,
            path=[source],
            cost=0
        )

        return obs

    def _get_metrics(self):
        T = self._time_step
        C = T + 1
        L = self._total_path_cost  # true objective
        V = self._num_total_nodes

        metrics = dict(
            ep_return=self._episode_return,
            ep_length=T,
            ep_time=time.time() - self._episode_start_time,
            path_cost=L,
            nodes_discovered=V,
            er=T / L,
            avg_er=np.mean(self._exploration_rates[1:T + 1]),
            node_coverage=C / V,
            avg_node_coverage=np.mean(self._node_coverages[:T+1])
        )

        return metrics

    def nearest_neighbor_policy(self):
        """Search the known graph starting from the current node,
            until a node is found that is in the frontier (unexplored).
        """
        frontier_set = self._frontier_set

        for u, v in nx.bfs_edges(self._known_graph, self._current_node):
            if v in frontier_set:
                return v

        raise ValueError(f"Nearest Neighbor: Did not found frontier node in the BFS")

    def bfs_policy(self):
        return self._frontier[0]

    def dfs_policy(self):
        return self._frontier[-1]

    def random_policy(self):
        i = self.rng.randint(len(self._frontier))
        return self._frontier[i]

    def step(self, action: int):

        # interpret agent's action as subgoal
        # NOTE: We do not interpret the action as a pointer to a frontier node,
        # Since the action refers to the original node label,
        # it is easier to examine / reproduce the episode.
        node_to_visit = action

        # remove the selected node from the frontier
        self._frontier_set.remove(node_to_visit)
        self._frontier.remove(node_to_visit)
        self._visit_sequence[self._time_step + 1] = node_to_visit

        # find shortest path to subgoal in the known graph (assuming unweighted graphs)
        path_cost, path_nodes = nx.single_source_dijkstra(self._known_graph, self._current_node, target=node_to_visit)

        tp1 = self._time_step + 1

        # update path length measurements
        self._path_lengths[tp1] = path_cost
        self._cum_path_lengths[tp1] = self._cum_path_lengths[self._time_step] + path_cost

        # update rate measurements
        u = tp1 / self._cum_path_lengths[tp1]
        self._exploration_rates[tp1] = u
        self._path_lengths_over_time[tp1] = 1 / u

        # update path cost
        self._total_path_cost += path_cost
        self._total_path_edges += len(path_nodes) - 1

        # update known graph
        subgoal_neighbors = set(self._graph.adj[node_to_visit])
        known_subgoal_neighbors = self._known_graph.adj[node_to_visit]

        # nodes that were just revealed
        new_nodes = list(subgoal_neighbors.difference(self._known_graph))
        self.rng.shuffle(new_nodes)  # shuffle to avoid any ordering bias due to node labeling
        self._num_total_nodes += len(new_nodes)
        self._node_coverages[tp1] = (tp1 + 1) / self._num_total_nodes

        # we knew these nodes existed, we just didn't know they were connected to the subgoal (loop closure)
        found_frontier_nodes = subgoal_neighbors.intersection(self._known_graph).difference(known_subgoal_neighbors)

        # there are edges to the newly discovered nodes
        discovered_edges = [(node_to_visit, v) for v in chain(new_nodes, found_frontier_nodes)]

        self._known_graph.add_edges_from(discovered_edges)
        self._num_total_edges += len(discovered_edges)

        # update frontier
        self._frontier.extend(new_nodes)
        self._frontier_set.update(new_nodes)
        # self.action_space.n = len(self._frontier)

        # update current node
        self._previous_node = self._current_node
        self._current_node = node_to_visit
        self._time_step += 1

        # prepare return tuple (obs, reward, done, info)
        obs = StepEvents(
            new_nodes=new_nodes,
            new_edges=discovered_edges,
            found_frontier_nodes=found_frontier_nodes,
            previous_node=self._previous_node,
            current_node=node_to_visit,
            path=path_nodes,
            cost=path_cost
        )

        t = self._time_step

        if self.reward_type == 'path_length':
            reward = - path_cost
        elif self.reward_type == 'er_diff':
            reward = self._exploration_rates[t] - self._exploration_rates[t - 1]
        elif self.reward_type == 'sign_er_diff':
            reward = np.sign(self._exploration_rates[t] - self._exploration_rates[t - 1])
        elif self.reward_type == 'edges-path':
            reward = len(discovered_edges) - path_cost
        elif self.reward_type == 'nodes-path':
            reward = len(new_nodes) - path_cost
        elif self.reward_type == 'loop-path':
            reward = len(found_frontier_nodes) - path_cost
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")

        self._episode_return += reward

        done = (len(self._frontier) == 0) or (t >= self.max_episode_steps)
        if done:
            info = {
                'episode': self._get_metrics(),
                'path_lengths': self._path_lengths[:t + 1].copy(),
                'visit_sequence': self._visit_sequence[:t + 1].copy()
            }
        else:
            info = {}

        return obs, reward, done, info

    def render(self, mode='human'):
        pass
