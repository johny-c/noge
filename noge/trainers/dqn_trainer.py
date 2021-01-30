import torch
import copy

from noge.data_types import ReplayBatch, TrainingBatch, InferenceSample
from noge.trainers.base_trainer import Trainer


class DQNTrainer(Trainer):
    def __init__(self, gamma, target_update_freq, replay_buffer, batch_size, network, preprocessor,
                 criterion, optimizer, scheduler=None, device='cpu'):
        super().__init__(replay_buffer, batch_size, network, preprocessor, criterion, optimizer, scheduler, device)

        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.target_net = copy.deepcopy(network).to(self.device)
        self.target_net.eval()

    def preprocess_batch(self, batch: ReplayBatch) -> TrainingBatch:
        # preprocess list of dicts
        graph_obses = [self.graph_obs_fn(graph_obs) for graph_obs in batch.graph_obses]
        next_graph_obses = [self.graph_obs_fn(graph_obs) for graph_obs in batch.next_graph_obses]

        # most arrays can be directly moved to torch
        measurements = self.preprocessor.transform_meas(batch.measurements)
        next_measurements = self.preprocessor.transform_meas(batch.next_measurements)

        actions = torch.from_numpy(batch.actions).to(self.device)
        rewards = torch.from_numpy(batch.rewards).to(self.device)
        mask = torch.from_numpy(batch.mask).to(self.device)

        # make training batch as defined in data protocol
        train_batch = TrainingBatch(graph_obses=graph_obses,
                                    measurements=measurements,
                                    actions=actions,
                                    rewards=rewards,
                                    next_graph_obses=next_graph_obses,
                                    next_measurements=next_measurements,
                                    goals=None,
                                    mask=mask)

        return train_batch

    def step(self, batch_size=None):

        B = batch_size or self.batch_size
        replay_batch: ReplayBatch = self.replay_buffer.sample(B)
        train_batch: TrainingBatch = self.preprocess_batch(replay_batch)

        self.network.train()
        q_sa = self.network(train_batch)  # [B, 1]

        q_next = []
        with torch.no_grad():
            for go, meas in zip(train_batch.next_graph_obses, train_batch.next_measurements):
                sample = InferenceSample(graph_obs=go, meas=meas.unsqueeze(0), goal=None)
                q_next_i = self.target_net(sample)   # [N_i, 1]
                q_next.append(q_next_i.max(0)[0])

        q_next = torch.cat(q_next)  # [B,]
        q_next = train_batch.mask * q_next  # [B,]

        q_target = train_batch.rewards + self.gamma * q_next  # [B,]

        loss = self.criterion(q_sa, q_target.reshape(B, 1))

        # do gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        # log stats
        self.train_metrics.append(dict(q_loss=loss.item()))
        self.num_train_steps += 1

        if self.num_train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.network.state_dict())
