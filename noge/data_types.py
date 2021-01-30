import numpy as np
import torch
from dataclasses import dataclass
from typing import List, TypedDict


# Communication protocol between Environment , Collect Policy , Replay Buffer and Trainer


class PartialGraphObservation(TypedDict):
    t: int
    new_nodes: np.ndarray
    new_edges: np.ndarray
    visited_node: int
    path_nodes: np.ndarray
    path_cost: float
    nn: int
    bf: int
    df: int
    meas: np.ndarray


class GoalPartialObservation(TypedDict):
    t: int
    new_edges: np.ndarray
    new_nodes: np.ndarray
    visited_node: int
    path_nodes: np.ndarray
    path_cost: float
    nn: int
    bf: int
    df: int
    meas: np.ndarray
    goal: np.ndarray


# PartialObs + Memory = FullObs
class GraphObservation(TypedDict):
    x: np.ndarray
    edge_index: np.ndarray
    frontier: np.ndarray
    visited_seq: np.ndarray


class PolicyObservation(TypedDict):
    x: np.ndarray
    edge_index: np.ndarray
    frontier: np.ndarray
    visited_seq: np.ndarray
    input_meas: np.ndarray
    goal: np.ndarray


@dataclass
class NeuralGraphObservation:
    """ A single environment observation converted to a form feedable to a neural network.

        x:           array of shape (N_t, D), features of the observed graph nodes.
        edge_index:  array of shape (2, M_t), the observed graph edges.
        frontier:    array of shape (F_t,),   the indices of nodes that are in the frontier.
        visited_seq: array of shape (C_t,),   the indices of nodes that are visited in order of visitation.

    """
    __slots__ = ('x', 'edge_index', 'frontier', 'visited_seq')

    x: torch.FloatTensor
    edge_index: torch.LongTensor
    frontier: torch.LongTensor
    visited_seq: torch.LongTensor


@dataclass
class InferenceSample:
    """ A sample to be fed to a neural network for policy inference.
        Env (ObservationSpace) -> Collect Policy (NetworkSpace)

        graph_obses: a NeuralGraphObservation instance.
        measurements: tensor of shape (1, D_meas), a measurement vector.
        goals: tensor of shape (1, D_goal), a goal vector.

    """
    __slots__ = ('graph_obs', 'meas', 'goal')

    graph_obs: NeuralGraphObservation
    meas: torch.FloatTensor
    goal: torch.FloatTensor


@dataclass
class ReplayBatch:
    """ A batch of samples sampled from the replay buffer.

        graph_obses: list of B GraphObservation instances.
        measurements: array of shape (B, D_meas), a batch of measurement vectors.
        actions: array of shape (B,), a batch of actions.
        rewards: array of shape (B,), a batch of rewards.
        next_graph_obses: list of B GraphObservation instances.
        next_measurements: array of shape (B, D_meas), a batch of measurement vectors.
        mask: array of shape (B,), a batch of booleans indicating non-terminal timesteps.
    """
    __slots__ = ('graph_obses', 'measurements', 'actions', 'rewards', 'next_graph_obses', 'next_measurements', 'mask')

    graph_obses: List[GraphObservation]
    measurements: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_graph_obses: List[GraphObservation]
    next_measurements: np.ndarray
    mask: np.ndarray


@dataclass
class TrainingBatch:
    """ A batch of samples to be fed to a neural network for training.
        Replay Buffer (BufferSpace) -> Trainer (NetworkSpace)

        graph_obses: list of B NeuralGraphObservation instances.
        measurements: tensor of shape (B, D_meas), a batch of measurement vectors.
        goals: tensor of shape (B, D_goal), a batch of goal vectors.
        actions: tensor of shape (B,), a batch of actions.
        targets: tensor of shape (B, D_goal), a batch of target measurement vectors.
        targets_mask: tensor of shape (B, D_goal), the validity of each entry in the target measurements.

    """
    __slots__ = ('graph_obses', 'measurements', 'goals', 'actions', 'rewards',
                 'next_graph_obses', 'next_measurements', 'mask')

    graph_obses: List[NeuralGraphObservation]
    measurements: torch.FloatTensor
    goals: torch.FloatTensor
    actions: torch.LongTensor
    rewards: torch.FloatTensor
    next_graph_obses: List[NeuralGraphObservation]
    next_measurements: torch.FloatTensor
    mask: torch.FloatTensor


@dataclass
class DFPReplayBatch:
    """ A batch of samples sampled from the replay buffer.

        graph_obses: list of B GraphObservation instances.
        measurements: array of shape (B, D_meas), a batch of measurement vectors.
        goals: array of shape (B, D_goal), a batch of goal vectors.
        actions: array of shape (B,), a batch of actions.
        targets: array of shape (B, D_goal), a batch of target measurement vectors.
        targets_mask: array of shape (B, D_goal), the validity of each entry in the target measurements.

    """
    __slots__ = ('graph_obses', 'measurements', 'actions', 'goals', 'targets', 'targets_mask')

    graph_obses: List[GraphObservation]
    measurements: np.ndarray
    actions: np.ndarray
    goals: np.ndarray
    targets: np.ndarray
    targets_mask: np.ndarray


@dataclass
class DFPTrainingBatch:
    """ A batch of samples to be fed to a neural network for training.
        Replay Buffer (BufferSpace) -> Trainer (NetworkSpace)

        graph_obses: list of B NeuralGraphObservation instances.
        measurements: tensor of shape (B, D_meas), a batch of measurement vectors.
        goals: tensor of shape (B, D_goal), a batch of goal vectors.
        actions: tensor of shape (B,), a batch of actions.
        targets: tensor of shape (B, D_goal), a batch of target measurement vectors.
        targets_mask: tensor of shape (B, D_goal), the validity of each entry in the target measurements.

    """
    __slots__ = ('graph_obses', 'measurements', 'goals', 'actions', 'targets', 'targets_mask')

    graph_obses: List[NeuralGraphObservation]
    measurements: torch.FloatTensor
    goals: torch.FloatTensor
    actions: torch.LongTensor
    targets: torch.FloatTensor
    targets_mask: torch.BoolTensor


@dataclass
class Transition:
    """ An agent-environment interaction step.
        (ObservationSpace) -> Replay Buffer (BufferSpace)

        obs: An Observation instance observed by the agent at time t.
        action: The action the agent selected.
        reward: The reward the agent received.
        next_obs: An Observation instance observed by the agent at time t+1.
        terminal: Whether the episode ended at time t+1.

    """
    __slots__ = ('obs', 'action', 'reward', 'next_obs', 'terminal')

    obs: GoalPartialObservation
    action: int
    reward: float
    next_obs: GoalPartialObservation
    terminal: bool
