import tempfile
from sacred import Experiment
from pathlib import Path

from noge.data_loaders import get_datasets, get_test_loader
from noge.evaluation import save_eval_results, eval_ing, evaluate_policy
from noge.envs import BaseGraphEnv
from noge.constants import REAL_DATASETS, SYNTHETIC_DATASETS
from noge.constants import BASELINES as POLICIES
from noge.constants import EVAL_DIR
from xlog.utils import get_logger
from xlog.mlflow_observer import MlflowObserver


ex = Experiment(name='NOGE_BASELINE', ingredients=[eval_ing])
ex.observers = [MlflowObserver(tracking_uri=str(EVAL_DIR))]
ex.logger = get_logger(__name__)


@ex.config
def cfg():
    # env
    dataset = 'maze'
    max_episode_steps = 500
    weighted = False

    # algorithm
    algo = 'dfs'

    # data
    test_size = 0.2
    n_test_episodes = 50
    n_eval_artifacts = 5

    # other
    seed = 1
    data_seed = 1


@ex.main
def evaluate(dataset, max_episode_steps, algo, test_size, n_test_episodes, n_eval_artifacts,
             seed, data_seed, weighted, _run):
    logger = ex.logger

    # load data
    train_set, test_set = get_datasets(dataset, seed=data_seed, test_size=test_size, weighted=weighted)
    test_loader = get_test_loader(test_set, seed=seed, num_samples=n_test_episodes)
    logger.info(f"Evaluating {algo.upper()} on {dataset} dataset [size={len(test_set)}].")

    # environment
    env = BaseGraphEnv(max_episode_steps=max_episode_steps, weighted=weighted)
    env.seed(seed)

    # policy
    if algo == 'nn':
        policy_fn = env.nearest_neighbor_policy
    elif algo == 'dfs':
        policy_fn = env.dfs_policy
    elif algo == 'bfs':
        policy_fn = env.bfs_policy
    elif algo == 'random':
        policy_fn = env.random_policy
    else:
        raise ValueError(f"Unrecognized algorithm name: {algo}")

    # all policies are greedy / blind, they don't depend on the **contents** of the frontier
    def policy(obs):
        return policy_fn()

    perf_metric = 'er'

    # make a temp dir to store models, metrics, etc.
    with tempfile.TemporaryDirectory() as temp_dir:

        run_dir = Path(temp_dir)
        logger.info(f"Saving temporary files under {run_dir}...\n")

        # evaluate
        eval_results = evaluate_policy(env, test_loader, policy)

        # store evaluation results and metrics
        test_metrics = save_eval_results(eval_results, run_dir, epoch=0, n_artifacts=n_eval_artifacts)
        epoch_perf = test_metrics[perf_metric]

    return epoch_perf


ex2 = Experiment(name='NOGE_MANY_BASELINES', ingredients=[ex])
ex2.logger = get_logger(__name__)


@ex2.config
def cfg():
    datasets = SYNTHETIC_DATASETS + REAL_DATASETS  # maze, grid, barabasi, caveman, tree, ladder, SFO, OXF, MUC
    policies = POLICIES
    max_episode_steps = 500
    n_eval_artifacts = 5
    seeds = [1, 2, 3, 4, 5]


@ex2.automain
def eval_all(datasets, policies, max_episode_steps, n_eval_artifacts, seeds):
    all_datasets = list(SYNTHETIC_DATASETS) + list(REAL_DATASETS)
    for dataset in datasets:
        assert dataset in all_datasets, f"Dataset {dataset} is unknown"

    for policy in policies:
        assert policy in POLICIES, f"Policy {policy} is unknown"

    for dataset in datasets:
        for algo in policies:
            for i, seed in enumerate(seeds, start=1):
                ex2.logger.info(f"RUN {i:3}/{len(seeds)} - {algo.upper()} on {dataset}")
                ex2.run(command_name='evaluate',
                       config_updates={
                           'dataset': dataset,
                           'algo': algo,
                           'max_episode_steps': max_episode_steps,
                           'n_eval_artifacts': n_eval_artifacts,
                           'seed': seed,
                       })
                ex2.logger.info("-" * 40)


# run with
# PYTHONPATH+=. python scripts/evaluate_baselines.py with datasets="['MUC','OXF']" policies="['dfs','nn']"
