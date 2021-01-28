import time
import pickle
import torch
import pprint
import logging
import datetime
import numpy as np

LOG_FORMAT = '%(asctime)s - %(filename)-20s - %(levelname)-10s - %(message)s'


def get_logger(name=None, fmt=LOG_FORMAT, level=logging.INFO):
    logger = logging.getLogger(name)
    formatter = logging.Formatter(fmt=fmt)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if level:
        logger.setLevel(level)
    return logger


def get_timestamp(fmt='%b%d_%H-%M-%S'):
    return datetime.datetime.now().strftime(fmt)


def save_model(model, path, **kwargs):
    """
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        ...
    }, PATH)
    """
    torch.save({'model_state_dict': model.state_dict(), **kwargs}, path)


def load_model(model, path):
    """
        model = TheModelClass(*args, **kwargs)
        optimizer = TheOptimizerClass(*args, **kwargs)

        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        model.eval()
        # - or -
        model.train()
    """

    print(f"Loading model from {path}")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    pprint.pprint({k: v for k, v in checkpoint.items() if k != 'model_state_dict'})
    return model


def save_pickle(data, path):
    print(f"Storing data to {path}...")
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path):
    print(f"Loading data from {path}..")
    with open(path, "rb") as f:
        data = pickle.load(f)

    return data


def collapse_dicts(list_of_dicts, reduce_fn=np.mean):
    """List of dicts to dict of lists or dict of reduced lists"""
    dict_0 = list_of_dicts[0]
    keys = dict_0.keys()
    out = {k: reduce_fn([d[k] for d in list_of_dicts]) for k in keys}
    return out


class Timer:
    def __init__(self, logger=None):
        self.logger = logger or get_logger()
        self.start_time = None
        self.measurement = None

    def __enter__(self):
        self.logger.info(f"Timing . . .")
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        time_elapsed = end_time - self.start_time
        self.logger.info(f"Took {time_elapsed:.2f}s")
        self.measurement = time_elapsed
        return None

    def get(self):
        return self.measurement
