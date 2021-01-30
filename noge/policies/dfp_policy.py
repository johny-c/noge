import torch
import numpy as np

from noge.data_types import GoalPartialObservation, NeuralGraphObservation, InferenceSample


class GraphDFPPolicy:

    def __init__(self, network, graph_memory, preprocessor, exploration_schedule=None, device='cpu'):
        self.network = network
        self.graph_memory = graph_memory
        self.preprocessor = preprocessor
        self.exploration_schedule = exploration_schedule
        self.device = device

        self._is_collecting = exploration_schedule is not None

    def __call__(self, partial_obs: GoalPartialObservation) -> int:

        # EnvObservation (PartialGraphObservation) -> GraphObservation
        self.graph_memory.update(partial_obs)
        frontier = self.graph_memory.get_frontier()
        frontier_size = len(frontier)

        # if there is only a single option, we don't need to forward pass
        if frontier_size == 1:
            return frontier[0]

        # if it is the first time step, all options should be equally likely
        t = partial_obs['t']
        if t == 0:
            i = np.random.randint(frontier_size)
            return frontier[i]

        # exploration: only for behavioral policy (collection phase)
        if self._is_collecting:
            u = np.random.rand()
            if u < self.exploration_schedule.current:
                i = np.random.randint(frontier_size)
                return frontier[i]

        # (epsilon < u) or this is an inference policy (evaluation phase)

        # graph
        graph_obs = self.graph_memory.get()
        torch_graph_obs = {k: torch.from_numpy(v).to(self.device) for k, v in graph_obs.items()}
        neural_graph_obs = NeuralGraphObservation(**torch_graph_obs)

        # measurement
        meas_t = self.preprocessor.transform_meas(partial_obs['meas'])

        # goal
        goal = torch.from_numpy(partial_obs['goal']).unsqueeze(0).to(self.device)

        # network input
        net_input = InferenceSample(graph_obs=neural_graph_obs, meas=meas_t, goal=goal)

        # forward pass
        with torch.no_grad():
            p = self.network(net_input)  # [A, D_goal]

        # p = self.preprocessor.transform_prediction(p)

        # DFP policy
        frontier_node_values = p @ goal.squeeze()   # [A]
        frontier_node_index = frontier_node_values.argmax().item()  # a

        # map index to frontier nodes
        frontier_obs = graph_obs['frontier']
        node = frontier_obs[frontier_node_index]

        return node
