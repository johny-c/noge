import tqdm
import torch
import tempfile
from pathlib import Path
from collections import deque
from itertools import count
from sacred import Ingredient

from noge.data_types import Transition
from noge.evaluation import save_eval_results
from xlog.sum_writer import sum_writer, log_metrics, log_model
from xlog.utils import collapse_dicts


loop_ing = Ingredient('loop', ingredients=[sum_writer])


class Actor:
    def __init__(self, env, policy, replay_buffer):
        self.env = env
        self.policy = policy
        self.replay_buffer = replay_buffer

        # global vars / counters
        self.num_steps = 0
        self.num_episodes = 0

        # metrics
        self.collect_metrics = deque(maxlen=10)

        # initialize observation
        self.last_obs = env.reset()

    def reset_counters(self, metrics=False):
        self.num_steps = 0
        self.num_episodes = 0

        if metrics:
            self.collect_metrics.clear()

    def step(self, n=1, use_tqdm=False, run_episode=False):

        env = self.env

        if run_episode:
            step_iter = count()
        elif use_tqdm:
            step_iter = tqdm.trange(n)
        else:
            step_iter = range(n)

        with torch.no_grad():
            for _ in step_iter:
                # policy inference
                action = self.policy(self.last_obs)

                # simulation
                next_obs, reward, done, info = env.step(action)

                # storage
                transition = Transition(obs=self.last_obs, action=action, reward=reward, next_obs=next_obs, terminal=done)
                self.replay_buffer.push(transition)

                self.last_obs = next_obs

                # update counters
                self.num_steps += 1

                # handle end of episode
                if done:
                    # print(f"Episode done at step {self.num_steps}")
                    episode_metrics = info['episode']
                    self.collect_metrics.append(episode_metrics)

                    # update counters
                    self.num_episodes += 1

                    # have to reset
                    self.last_obs = env.reset()

                    if run_episode:
                        break

    @property
    def counters(self):
        return dict(env_steps=self.num_steps, episodes=self.num_episodes)

    def gather_metrics(self):
        # list of dicts to dict of (means of) lists
        return collapse_dicts(self.collect_metrics)


@loop_ing.capture
def main_loop(actor, trainer, evaluator, network, exploration_schedule,
              init_eval, n_eval_artifacts, n_train_steps, train_freq, log_freq, test_freq, save_model, _log):

    logger = _log

    # reset counters and metrics collected during filling the replay buffer
    actor.reset_counters(metrics=True)
    perf_metric = 'er'
    best_perf = 0

    # LOOP
    with tempfile.TemporaryDirectory() as temp_dir:  # make a temp dir to store models, metrics, etc.

        run_dir = Path(temp_dir)

        # evaluate once before training
        if init_eval:
            eval_results = evaluator.run(network)
            test_metrics = save_eval_results(eval_results, run_dir, epoch=0, n_artifacts=n_eval_artifacts)
            best_perf = test_metrics[perf_metric]
            global_vars = dict(epsilon=exploration_schedule.current, train_steps=0, **actor.counters)
            log_metrics(0, test_metrics, global_vars, mode='test')

        # loop
        epoch = 1
        for t in range(1, n_train_steps + 1):

            # TRAINING STEP
            trainer.step()
            epsilon = exploration_schedule.step()  # reduce epsilon (epsilon greedy policy)

            # COLLECT DATA
            network.eval()
            actor.step(n=train_freq)  # act in the environment

            # REPORT TRAINING METRICS
            if t % log_freq == 0:
                train_metrics = trainer.gather_metrics()
                train_metrics.update(actor.gather_metrics())
                global_vars = dict(epsilon=epsilon, train_steps=t, **actor.counters)
                log_metrics(t, train_metrics, global_vars, mode='train')

            # EVALUATION
            if t % test_freq == 0:
                eval_results = evaluator.run(network)
                test_metrics = save_eval_results(eval_results, run_dir, epoch=epoch, n_artifacts=n_eval_artifacts)
                epoch_perf = test_metrics[perf_metric]
                global_vars = dict(epsilon=epsilon, train_steps=t, **actor.counters)
                log_metrics(t, test_metrics, global_vars, mode='test')

                # epoch performance
                if epoch_perf > best_perf:
                    logger.info(f"{perf_metric} improved: {best_perf:.4f} --> {epoch_perf:.4f}\n")
                    best_perf = epoch_perf

                    if save_model:
                        model_path = run_dir / f"model_epoch_{epoch:03}.pth"
                        log_model(network, model_path, epoch=epoch, er=best_perf, train_step=t)

                # UPDATE MOMENTS
                logger.info(f"Finished epoch {epoch:3}, perf: {epoch_perf:.4f}  (best performance: {best_perf:.4f})")
                logger.info("-" * 80)
                epoch += 1

    return best_perf
