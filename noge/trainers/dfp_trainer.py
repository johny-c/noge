import torch
import numpy as np

from noge.data_types import DFPReplayBatch, DFPTrainingBatch, Transition
from noge.trainers.base_trainer import Trainer
from noge.trainers.replay_buffer import Replay


class DFPReplay(Replay):

    def __init__(self, capacity, ob_space, graph_mem_config, future_steps, min_horizon=4):
        super().__init__(capacity, ob_space, graph_mem_config, min_horizon)
        assert 'goal' in ob_space.spaces

        self.future_steps = future_steps

        # make normal numpy arrays for measurements and goals
        goal_space = ob_space.spaces['goal']
        self._goals = np.empty(shape=(capacity, *goal_space.shape), dtype=goal_space.dtype)

    def push(self, transition: Transition):
        obs = transition.obs
        # push goal before calling superclass to get the right 'pointer'
        self._goals[self._pointer] = obs['goal']

        if transition.terminal:
            next_pointer = (self._pointer + 1) % self.capacity
            self._goals[next_pointer] = transition.next_obs['goal']

        super().push(transition)

    def sample(self, batch_size: int) -> DFPReplayBatch:
        valid_idx = self._sample_valid_indices(batch_size)

        # retrieve / copy from storage
        graph_obses = self._graphs_buffer.get(valid_idx)
        measurements = self._measurements[valid_idx]  # [B, M, Dm]
        goals = self._goals[valid_idx]
        actions = self._actions[valid_idx]

        # make targets
        targets, targets_mask = self._make_targets(valid_idx)

        # make replay batch according to data protocol
        sample_batch = DFPReplayBatch(graph_obses=graph_obses,
                                      measurements=measurements,
                                      goals=goals,
                                      actions=actions,
                                      targets=targets,
                                      targets_mask=targets_mask)
        return sample_batch

    def _make_targets(self, indices: np.ndarray):
        # sample measurements
        measurements = self._measurements   # [N, Dm]
        meas_t = measurements[indices]        # [B, Dm]
        meas_t = np.expand_dims(meas_t, 1)  # [B, 1, Dm]

        # get future measurements at t + dt
        future_times = indices.reshape(-1, 1) + self.future_steps
        future_times = future_times % self.capacity     # [B, T]
        future_meas = measurements[future_times]        # [B, T, Dm]

        # mark future measurements as invalid if they are from a different episode
        episode_idx = self._episode_idx[indices].reshape(-1, 1)   # [B, 1]
        future_episode_idx = self._episode_idx[future_times]  # [B, T]
        valid_times_mask = future_episode_idx == episode_idx  # [B, T]

        # make targets: y = m_{t+dt} - m_t
        targets = future_meas - meas_t  # [B, T, Dm]

        # reshape targets mask [B, T] --> [B, T, Dm]
        B, T, D = targets.shape
        targets_mask = valid_times_mask.reshape(B, T, 1)  # [B, T, 1]
        targets_mask = np.tile(targets_mask, (1, 1, D))   # [B, T, Dm]

        return targets, targets_mask


class DFPTrainer(Trainer):

    def preprocess_batch(self, batch: DFPReplayBatch) -> DFPTrainingBatch:
        # preprocess list of dicts
        graph_obses = [self.graph_obs_fn(graph_obs) for graph_obs in batch.graph_obses]

        # most arrays can be directly moved to torch
        measurements = self.preprocessor.transform_meas(batch.measurements)
        goals = torch.from_numpy(batch.goals).to(self.device)
        actions = torch.from_numpy(batch.actions).to(self.device)

        # targets can be rescaled
        targets, targets_mask = self.preprocessor.transform_target(batch.targets, batch.targets_mask)

        # make training batch as defined in data protocol
        train_batch = DFPTrainingBatch(graph_obses=graph_obses,
                                       measurements=measurements,
                                       goals=goals,
                                       actions=actions,
                                       targets=targets,
                                       targets_mask=targets_mask)

        return train_batch

    def step(self, batch_size=None):

        B = batch_size or self.batch_size
        replay_batch: DFPReplayBatch = self.replay_buffer.sample(B)
        train_batch: DFPTrainingBatch = self.preprocess_batch(replay_batch)

        self.network.train()
        predictions = self.network(train_batch)  # [B, D_goal]

        targets = train_batch.targets            # [B, T=D_goal]
        target_masks = train_batch.targets_mask  # [B, T=D_goal]
        num_targets = targets.numel()
        num_valid_targets = target_masks.sum().item()

        # compute loss
        loss = self.criterion(predictions, targets, target_masks)

        # do gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        loss_sum = loss.item()
        full_loss = loss_sum / num_targets
        valid_loss = loss_sum / num_valid_targets

        # log stats
        d = dict(pred_loss=full_loss, valid_pred_loss=valid_loss)
        self.train_metrics.append(d)
        self.num_train_steps += 1
