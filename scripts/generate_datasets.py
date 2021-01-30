import math
import networkx as nx
from sacred import Experiment
from pathlib import Path

from noge.constants import SYNTHETIC_DATASETS
from noge.constants import DATA_DIR
from noge.envs import make_maze, maze_to_graph
from xlog.utils import save_pickle


def make_data(name, n_graphs, print_every=50):

    if name not in SYNTHETIC_DATASETS:
        raise ValueError(f"Unknown dataset name: {name}. Must be one of {SYNTHETIC_DATASETS}.")

    if name == 'barabasi':
        graphs = []
        nps = math.ceil(n_graphs / 100)
        for n in range(100, 200):
            for k in range(4, 5):  # each node is connected to 4 existing nodes
                for _ in range(nps):  # generate 5 graphs with n nodes
                    graphs.append(nx.barabasi_albert_graph(n, k))
                    if len(graphs) % print_every == 0:
                        print(f"Generated {len(graphs):4} graphs..")
    elif name == 'ladder':
        # nps = math.ceil(n_graphs / 100)
        graphs = [nx.ladder_graph(n) for n in range(100, 200)]
    elif name == 'grid':
        graphs = []
        # nps = math.ceil(n_graphs / 100)
        for n_rows in range(8, 18):
            for n_cols in range(8, 18):
                # for _ in range(nps):
                graphs.append(nx.grid_2d_graph(n_rows, n_cols))
                if len(graphs) % print_every == 0:
                    print(f"Generated {len(graphs):4} graphs..")

        # networkx labels grid nodes by (x, y) coordinates instead of integers, so relabel them
        for i in range(len(graphs)):
            G = graphs[i]
            node_dict = {yx: i for i, yx in enumerate(G.nodes)}
            edges = [(node_dict[u], node_dict[v]) for u, v in G.edges]
            G = nx.from_edgelist(edges)
            G.pos_yx = {i: yx for yx, i in node_dict.items()}
            graphs[i] = G

    elif name == 'tree':
        graphs = []
        # nps = math.ceil(n_graphs / 2)
        branch_factor_to_range = {3: (4, 6), 4: (4, 5), 5: (4, 4)}
        for branch_factor, (hmin, hmax) in branch_factor_to_range.items():
            for height in range(hmin, hmax+1):
                graphs.append(nx.balanced_tree(branch_factor, height))
                if len(graphs) % print_every == 0:
                    print(f"Generated {len(graphs):4} graphs..")
    elif name == 'caveman':
        # graphs = []
        # for i in range(5,10):
        #     for j in range(5,25):
        #         for k in range(5):
        #             graphs.append(nx.relaxed_caveman_graph(i, j, p=0.1))
        graphs = []
        # nps = math.ceil(n_graphs / (3 * 50))
        for n_cliques in range(2, 5):
            for clique_size in range(30, 80):
                # for _ in range(nps):
                graphs.append(nx.connected_caveman_graph(n_cliques, clique_size))
                if len(graphs) % print_every == 0:
                    print(f"Generated {len(graphs):4} graphs..")
    elif name == 'maze':
        graphs = []
        maze_sizes = list(range(15, 25, 2))
        nps = math.ceil(n_graphs / len(maze_sizes))
        for maze_size in maze_sizes:
            for _ in range(nps):
                maze = make_maze(maze_size, maze_size)
                G = maze_to_graph(maze)
                graphs.append((G, maze))
                if len(graphs) % print_every == 0:
                    print(f"Generated {len(graphs):4} graphs..")

        return graphs
    else:
        raise ValueError(f"Unknown type of graphs: {name}")

    return graphs


ex = Experiment('Synthetic graphs generation')


@ex.config
def cfg():
    name = 'maze'
    n_graphs = 500 if name in ('barabasi', 'maze') else None
    print_every = 50
    data_dir = DATA_DIR
    seed = 123


@ex.automain
def main(name, n_graphs, print_every, data_dir, _log):

    _log.info(f"Making paths...")
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    path = data_dir / f"{name}.pkl"

    _log.info(f"Generating {name} graphs...")
    data = make_data(name, n_graphs, print_every=print_every)

    _log.info(f"{len(data):5} graphs generated. Now storing...")
    save_pickle(data, path)
    _log.info(f"Data stored.")
