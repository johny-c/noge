import pandas as pd
from tabulate import tabulate
from sacred import Ingredient

from xlog.utils import save_model


sum_writer = Ingredient('sum_writer')


@sum_writer.capture
def log_metrics(step, metrics, global_vars, mode, _run, _log):
    """

    :param step: int, global step (training step or env.step)
    :param metrics: dict, metrics recorded at the current step
    :param global_vars: dict, global variables at the current step
    :param mode: str, such as 'train', 'test' or 'val'
    :param _run: sacred.Run, the sacred run currently active
    :param _log: logging.Logger, the run's logger
    :return: dict, metrics report
    """

    # prefix with mode
    prefix = mode + '_'
    metrics_and_vars = {prefix + key: value for key, value in metrics.items()}
    metrics_and_vars.update(global_vars)

    # log to experiment storage
    for key, value in metrics_and_vars.items():
        _run.log_scalar(key, value, step)

    # show on screen
    # report = tabulate([metrics_and_vars], tablefmt='grid', floatfmt='.4f', headers='keys')  # one row
    report = tabulate(metrics_and_vars.items(), tablefmt='grid')  # one column, default floatfmt = 'g'
    _log.info(f"{mode.capitalize()} report:")
    _log.info(f"\n{report}")
    _log.info("-" * 40)

    return metrics_and_vars


@sum_writer.capture
def log_metrics_list(metrics_list, path, _run, **kwargs):
    """ Save a list of (dicts of) metrics to a csv file.

    :param metrics_list: list of dicts, each dict contains the same set of keys (metrics names).
    :param path: str or Path, where to save.
    :param _run: sacred.Run, the sacred run currently active
    :param kwargs: dict, any extra info to store in the DataFrame (e.g. algorithm name, dataset name)
    :return: pd.DataFrame
    """

    df = pd.DataFrame.from_records(metrics_list)
    for k, v in kwargs.items():
        df[k] = v

    df.to_csv(path_or_buf=path, index=False)
    _run.add_artifact(path)

    return df


@sum_writer.capture
def log_model(model, path, _run, _log, **info):
    # save model
    # model_path = run_dir / f"model_epoch_{epoch:03}.pth"
    _log.info(f"Saving model to {path}")
    # info = {'epoch': epoch}
    save_model(model, path, **info)
    _run.add_artifact(path)
