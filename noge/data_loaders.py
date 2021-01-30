import numpy as np
import torch.utils.data as tud
from sklearn.model_selection import train_test_split

from noge.constants import REAL_DATASETS, PLACES, DATA_DIR
from xlog.utils import load_pickle


class GraphDataset(tud.Dataset):

    def __init__(self, graphs, mazes=None):

        self.graphs = graphs
        self.mazes = mazes

        self.num_nodes_per_graph = np.array([G.number_of_nodes() for G in graphs], dtype=int)
        self.num_edges_per_graph = np.array([G.number_of_edges() for G in graphs], dtype=int)

        n_graphs = len(graphs)
        self._pairs = [(g, s) for g in range(n_graphs) for s in graphs[g].nodes]
        graph_idx, sources = zip(*self._pairs)
        self.samples_graph_idx = np.array(graph_idx)
        self._samples_sources = np.array(sources)

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, item):
        graph_index, source = self._pairs[item]
        graph = self.graphs[graph_index]

        sample = dict(graph=graph, source=source)

        if self.mazes is not None:
            sample.update(maze=self.mazes[graph_index])

        return sample

    @property
    def max_nodes(self):
        return max(self.num_nodes_per_graph)

    @property
    def max_edges(self):
        return max(self.num_edges_per_graph)

    @property
    def num_graphs(self):
        return len(self.graphs)


class SubsetSampler(tud.Sampler):

    def __init__(self, dataset, seed, num_samples=50):
        assert num_samples <= len(dataset)

        self.dataset = dataset
        self.seed = seed

        self.rng = np.random.RandomState(seed=seed)

        # for evaluation only choose pairs once (to be consistent across epochs)
        n_graphs = len(dataset.graphs)

        if n_graphs >= num_samples:
            # sample one source node per graph
            num_nodes_per_graph = self.dataset.num_nodes_per_graph
            indices = []
            offset = 0
            for num_nodes in num_nodes_per_graph:
                # num_nodes = num_nodes_per_graph[g]
                index = self.rng.randint(num_nodes)
                indices.append(offset + index)
                offset += num_nodes

                if len(indices) == num_samples:
                    break

            self._indices = indices

        else:

            # the number of graphs is less than the required num_samples
            n_total = len(dataset)

            if n_total <= num_samples:
                # if the total number of samples is less than or equal to required, use all samples
                self._indices = list(range(n_total))
            else:
                # if the total number of samples is larger than required, sub-sample
                self._indices = self.rng.choice(n_total, size=num_samples, replace=False).tolist()
                self._indices.sort()

    def __iter__(self):
        return iter(self._indices)

    def __len__(self):
        return len(self._indices)


def get_test_loader(dataset, seed, num_samples):
    sampler = SubsetSampler(dataset, seed=seed, num_samples=num_samples)

    # set batch_size = None to get each sample without a batch dimension
    # set collate_fn = identity to not trigger auto_collate which converts to torch types
    loader = tud.DataLoader(dataset=dataset, batch_size=None,
                            collate_fn=lambda x: x, sampler=sampler)
    return loader


class BalancedInfiniteRandomSampler:
    """ Sample each graph with equal probability (in the limit) """

    def __init__(self, dataset, seed, cycle_size=100_000, replace=True):
        self.dataset = dataset
        self.seed = seed
        self.rng = np.random.RandomState(seed=seed)

        # each node's weight should be proportional to 1 over the graph size of the node
        inverse_graph_sizes = 1. / dataset.num_nodes_per_graph
        self.p = inverse_graph_sizes[dataset.samples_graph_idx]
        self.p = self.p / self.p.sum()
        # self.weights = torch.as_tensor(self.p, dtype=torch.double)
        self.cycle_size = cycle_size
        self.replacement = replace

    def __iter__(self):
        while True:
            # sample once every `cycle_size` (rng.choice is slow)
            indices = self.rng.choice(len(self.dataset), self.cycle_size, self.replacement, p=self.p).tolist()
            # items = torch.multinomial(self.weights, self.cycle_size, self.replacement).tolist()
            for index in indices:
                yield index


def get_train_generator(dataset, seed):

    sampler = BalancedInfiniteRandomSampler(dataset, seed)
    sampler_iter = iter(sampler)

    while True:
        index = next(sampler_iter)
        sample = dataset[index]
        yield sample


def _get_real_graph(dataset):
    proc_dir = DATA_DIR / 'osm' / 'processed'

    # get place and name
    place = PLACES[dataset]
    name = place['city'].replace(' ', '')

    path_train = proc_dir / f"{name}_train.pkl"
    path_test = proc_dir / f"{name}_test.pkl"

    train_graph = load_pickle(path_train)
    test_graph = load_pickle(path_test)

    train_set = GraphDataset([train_graph])
    test_set = GraphDataset([test_graph])

    return train_set, test_set


def get_datasets(dataset, seed, test_size, val_size=0):
    if dataset in REAL_DATASETS:
        return _get_real_graph(dataset)

    path_graphs = DATA_DIR / f"{dataset}.pkl"
    graphs = load_pickle(path_graphs)
    graphs_train, graphs_test = train_test_split(graphs, test_size=test_size, random_state=seed)

    graphs_val = None
    if val_size > 0:
        graphs_train, graphs_val = train_test_split(graphs_train, test_size=val_size, random_state=seed)

    mazes_train = None
    mazes_test = None
    mazes_val = None
    if dataset in ('maze', 'hypermaze'):
        graphs_train, mazes_train = zip(*graphs_train)
        graphs_test, mazes_test = zip(*graphs_test)

        if graphs_val is not None:
            graphs_val, mazes_val = zip(*graphs_val)

    train_set = GraphDataset(graphs_train, mazes_train)
    test_set = GraphDataset(graphs_test, mazes_test)

    if graphs_val is not None:
        val_set = GraphDataset(graphs_val, mazes_val)
        return train_set, val_set, test_set

    return train_set, test_set
