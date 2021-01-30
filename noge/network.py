import torch
import torch.nn as nn
from typing import Union

from noge.neural import MLP, GCN_Model
from noge.data_types import DFPTrainingBatch, InferenceSample


class NOGENet(nn.Module):
    def __init__(self, gnn, meas_net, goal_net, joint_net, aggregation='mean', use_goal=True):
        super().__init__()
        assert aggregation == 'mean'

        self.gnn = gnn
        self.aggr_net = aggregation
        self.meas_net = meas_net
        self.goal_net = goal_net
        self.joint_net = joint_net
        self.use_goal = use_goal

    def forward(self, batch: Union[DFPTrainingBatch, InferenceSample]):

        if self.training:
            return self.forward_training_batch(batch)

        return self.forward_inference(batch)

    def _aggregate_nodes(self, z, visited_idx):
        Z_C = z[visited_idx]
        z_c = torch.mean(Z_C, 0).reshape(1, -1)  # [1, D_z]
        return z_c

    def _make_context(self, z_m, z_g, z_t, z_c):

        # goal vector g
        if self.use_goal:
            lst = [z_m, z_g, z_t, z_c]
        else:
            lst = [z_m, z_t, z_c]

        z_ctx = torch.cat(lst, 1)  # [N, 4D]

        return z_ctx

    def forward_inference(self, batch: InferenceSample):
        """ Predict for every node in the frontier.

        :param batch: dataclass instance with slots
                        graph_obs: NeuralGraphObservation
                        meas: torch.FloatTensor
                        goal: torch.FloatTensor

                        NeuralGraphObservation is a dataclass instance with slots:
                            key             value  (shape)
                            --------------------------------
                            x               tensor [N, D]
                            edge_index      tensor [2, M]
                            frontier        tensor [F]
                            visited_seq     tensor [C]

        :return: prediction tensor of shape [A, D_goal]
        """

        meas = batch.meas
        goal = batch.goal
        graph_obs = batch.graph_obs
        x = graph_obs.x
        edge_index = graph_obs.edge_index
        frontier = graph_obs.frontier
        visited_seq = graph_obs.visited_seq

        # print(f"\nFrontier [{len(frontier)}]:\n{frontier}")

        # encode nodes
        z = self.gnn(x, edge_index)

        # frontier nodes
        z_f = z[frontier]  # [A, D_z]

        # current node
        v_t = visited_seq[-1]
        z_t = z[v_t].unsqueeze(0)       # [1, D_z]

        # encode measurement
        z_m = self.meas_net(meas)  # [1, D_m]

        # encode goal
        z_g = self.goal_net(goal)  # [1, D_g]

        # aggregate visited nodes
        z_c = self._aggregate_nodes(z, graph_obs.visited_seq)  # [1, D_z]

        z_ctx = self._make_context(z_m, z_g, z_t, z_c)

        # joint prediction
        p = self.joint_net(z_f, z_ctx)  # [A, D_goal]

        return p

    def forward_training_batch(self, batch: DFPTrainingBatch):
        """

        :param batch: dataclass instance with slots
                        graph_obses: List[NeuralGraphObservation]
                        measurements: torch.FloatTensor
                        goals: torch.FloatTensor

                        NeuralGraphObservation is a dataclass instance with slots:
                            key             value  (shape)
                            --------------------------------
                            x               tensor [N, D]
                            edge_index      tensor [2, M]
                            frontier        tensor [F]
                            visited_seq     tensor [C]

        :return: prediction tensor of shape [B, D_goal]
        """

        meas = batch.measurements
        goals = batch.goals
        graph_obses = batch.graph_obses
        actions = batch.actions

        # node embeddings
        Z_list = [self.gnn(go.x, go.edge_index) for go in graph_obses]

        # frontier nodes
        z_f = [z[a].unsqueeze(0) for z, a in zip(Z_list, actions)]
        z_f = torch.cat(z_f)  # [B, D_z]

        # current nodes
        z_t = [z[go.visited_seq[-1]].unsqueeze(0) for z, go in zip(Z_list, graph_obses)]
        z_t = torch.cat(z_t)  # [B, D_z]

        # encode measurements
        z_m = self.meas_net(meas)  # [B, D_m]

        # encode goals
        z_g = self.goal_net(goals)  # [B, D_g]

        # aggregate visited nodes
        z_c = [self._aggregate_nodes(Z, go.visited_seq) for Z, go in zip(Z_list, graph_obses)]
        z_c = torch.cat(z_c)  # [B, D_z]

        # merge side information
        z_ctx = self._make_context(z_m, z_g, z_t, z_c)

        # concat context (state) with frontier node embeddings (action)
        p = self.joint_net(z_f, z_ctx)  # [B, D_goal]

        return p


class JointNet(nn.Module):

    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, output_activation, alpha, dropout):
        super().__init__()

        dims_hidden = [dim_hidden] * (num_layers - 1)
        dims = [dim_in] + dims_hidden + [dim_out]
        self.net = MLP(dims=dims, activate_last=False, alpha=alpha, dropout=dropout)

        if output_activation:
            self.f_out = nn.Tanh()
        else:
            self.f_out = nn.Identity()

    def forward(self, z_f, z_ctx):

        num_actions = z_f.shape[0]
        if z_ctx.shape[0] < num_actions:  # online [1, D_ctx]
            z_ctx = z_ctx.repeat(num_actions, 1)  # [N, D_ctx]

        phi = torch.cat((z_f, z_ctx), 1)  # [N, D_v + D_ctx]
        f_v = self.f_out(self.net(phi))

        return f_v


def make_network(dim_node,
                 dim_meas,
                 dim_goal,
                 dim_hidden,
                 nonlinearity,
                 alpha,
                 dropout,
                 num_gnn_layers,
                 num_meas_layers,
                 num_joint_layers,
                 use_goal,
                 output_activation,
                 max_edges):

    # Graph Encoder
    gnn_channels = [dim_node, dim_hidden // 2] + [dim_hidden] * (num_gnn_layers - 1)
    gnn = GCN_Model(gnn_channels, activate_last=True, max_edges=max_edges)

    # Measurement Encoder
    meas_dims = [dim_meas] + [dim_hidden] * num_meas_layers
    meas_net = MLP(meas_dims, nonlinearity=nonlinearity, alpha=alpha, activate_last=True, dropout=dropout)

    # Goal Encoder
    goal_dims = [dim_goal] + [dim_hidden] * num_meas_layers

    if use_goal:
        goal_net = MLP(goal_dims, nonlinearity=nonlinearity, alpha=alpha, activate_last=True, dropout=dropout)
    else:
        goal_net = nn.Identity()

    dim_node_emb = gnn_channels[-1]  # z_t, z_c
    dim_meas_emb = meas_dims[-1]     # z_m
    dim_goal_emb = goal_dims[-1]     # z_g


    # Join measurement encoding, goal encoding, current node embedding, visited aggregate encoding
    # z_ctx = [z_m || z_g || z_t || z_c ]
    dim_ctx = dim_meas_emb + dim_goal_emb * int(use_goal) + dim_node_emb * 2

    # Join frontier node embedding with context
    dim_joint_in = dim_ctx + dim_node_emb
    joint_net = JointNet(dim_in=dim_joint_in,
                         dim_hidden=2*dim_hidden,
                         dim_out=dim_goal,
                         num_layers=num_joint_layers,
                         output_activation=output_activation,
                         dropout=dropout,
                         alpha=alpha)

    network = NOGENet(gnn, meas_net, goal_net, joint_net, use_goal=use_goal)
    return network
