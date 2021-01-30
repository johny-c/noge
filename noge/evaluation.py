import torch
import time
import tqdm
import pickle
import pandas as pd
from sacred import Ingredient

from xlog.utils import get_logger
from xlog.sum_writer import sum_writer


eval_ing = Ingredient('evaluator', ingredients=[sum_writer])
logger = get_logger(__name__)


def run_episode(env, policy, env_reset_kwargs):

    obs = env.reset(**env_reset_kwargs)
    done = False
    episode_info = None
    while not done:
        action = policy(obs)
        next_obs, reward, done, episode_info = env.step(action)
        obs = next_obs

    return episode_info


@eval_ing.capture
def evaluate_policy(env, test_loader, policy, _run, _log):
    """Evaluate policy on test_data"""

    episode_infos = []

    _log.info(f"Evaluating ...")
    eval_tic = time.time()

    for e, sample in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
        # G, s = sample['graph'], sample['source']
        episode_info = run_episode(env, policy, sample)
        episode_infos.append(episode_info)

    eval_toc = time.time()
    _log.info(f"Evaluated in {eval_toc - eval_tic:.2f}s.")

    return episode_infos


class Evaluator:
    def __init__(self, data_loader, env, policy):
        self.data_loader = data_loader
        self.policy = policy
        self.env = env

    def run(self, network):
        n_test_graphs = len(self.data_loader)
        results = [None for _ in range(n_test_graphs)]

        logger.info(f"Evaluating . . .")
        tic = time.time()
        with torch.no_grad():
            for i, test_sample in enumerate(self.data_loader):
                results[i] = run_episode(self.env, self.policy, test_sample)

        toc = time.time()
        logger.info(f"Evaluation took {toc-tic:.2f}s.")

        return results

    def close(self):
        logger.info(f"Closing evaluation env...")
        ret = self.env.close()
        logger.info(f"Done.")
        return ret


@eval_ing.capture
def save_eval_results(results, save_dir, epoch, n_artifacts, _run):

    # split results into episode metrics and other artifacts
    metrics_list = [epinfo.pop('episode') for epinfo in results]  # list of dicts

    # save detailed metrics as .csv
    test_metrics_df = pd.DataFrame.from_records(metrics_list)
    test_metrics_path = save_dir / f"test_metrics_epoch_{epoch:03}.csv"
    test_metrics_df.to_csv(path_or_buf=test_metrics_path, index=False)
    _run.add_artifact(test_metrics_path)

    # save all path lengths as a separate artifact
    path_lengths_list = [epinfo.pop('path_lengths') for epinfo in results]  # list of numpy arrays (of varying size)
    save_path = save_dir / f"test_path_lengths_epoch_{epoch:03}.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(path_lengths_list, f)
    _run.add_artifact(save_path)

    # return summarized (average) metrics as dict
    mean_test_metrics = test_metrics_df.mean().to_dict()

    # save artifacts
    if n_artifacts > 0:
        n = len(results)

        if n <= n_artifacts:
            # if desired <= existing, save all artifacts
            artifacts_to_save = results
        else:
            # else save every existing / desired artifact
            save_freq = n // n_artifacts
            idx = [i for i in range(n) if i % save_freq == 0]
            idx = idx[:n_artifacts]
            artifacts_to_save = [results[i] for i in idx]

        save_path = save_dir / f"test_episodes_epoch_{epoch:03}.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(artifacts_to_save, f)
        _run.add_artifact(save_path)

    return mean_test_metrics
