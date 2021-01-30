import time
import sacred
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox

from noge.constants import PLACES
from noge.constants import DATA_DIR
from xlog.utils import save_pickle, load_pickle

RAW_DIR = DATA_DIR / 'osm' / 'raw'
PROC_DIR = DATA_DIR / 'osm' / 'processed'

ex = sacred.Experiment('Real Network Figure')


@ex.config
def cfg():
    dpi = 300
    output_dir = None
    show = 0
    dataset = 'SFO'
    edge_width = 1
    W = 12
    H = 12
    split = False
    alpha = 0.5


def get_name_from_place(place):
    return place['city'].replace(' ', '')


def split_graph(G, name, xmin, xmax, ymin, ymax, W, H, dpi=300, output_dir=None, show=True, alpha=1.0):
    # separating line: y = a * x + b
    dx = xmax - xmin
    dy = ymax - ymin
    a = dy / dx
    b = ymax - a * xmax

    pos = {v: (G._node[v]['x'], G._node[v]['y']) for v in G._node}

    # subgraph G1
    nodes1 = {v for v in G._node if pos[v][1] > a * pos[v][0] + b}
    edges1 = [(u, v, {'weight': G[u][v][0]['length']}) for u, v in G.edges() if u in nodes1 and v in nodes1]
    G1 = nx.from_edgelist(edges1)

    # subgraph G2
    nodes2 = {v for v in G._node if v not in nodes1}
    edges2 = [(u, v, {'weight': G[u][v][0]['length']}) for u, v in G.edges() if u in nodes2 and v in nodes2]
    G2 = nx.from_edgelist(edges2)

    if len(edges1) > len(edges2):
        G_train, G_test = G1, G2
    else:
        G_train, G_test = G2, G1

    path_train = PROC_DIR / f"{name}_train.pkl"
    path_test = PROC_DIR / f"{name}_test.pkl"

    save_pickle(G_train, path_train)
    save_pickle(G_test, path_test)

    # color nodes for figure
    nc = ['red' if v in nodes1 else 'blue' for v in G]

    fig, ax = plt.subplots(figsize=(W, H))
    nx.draw_networkx_nodes(G, pos=pos, node_color=nc, node_size=1, ax=ax, alpha=alpha)
    nx.draw_networkx_edges(G, pos=pos, edge_color='gray', width=1, ax=ax)
    plt.plot([xmin, xmax], [ymin, ymax], 'go--')

    # save splitting figure
    if output_dir is not None:
        fig_path = output_dir / f"{name}_split.png"
        print(f"Saving figure in {fig_path}")
        fig.savefig(fig_path, dpi=dpi)

    if show:
        plt.show()


@ex.automain
def get_graph(dataset, output_dir, edge_width, dpi, show, W, H, split, alpha):
    # get place and name
    place = PLACES[dataset]
    name = get_name_from_place(place)

    # load or download raw graph
    RAW_DIR.mkdir(exist_ok=True, parents=True)
    PROC_DIR.mkdir(exist_ok=True, parents=True)
    path_raw = RAW_DIR / f"{name}.pkl"
    if path_raw.exists():
        # load downloaded graph
        G = load_pickle(path_raw)
    else:
        # download
        print(f"Downloading data for {place['city']}...")
        tic = time.time()
        G = ox.graph_from_place(place, network_type='drive')
        toc = time.time()
        print(f"Done in {toc-tic:.2f}s.")

        # save "raw" downloaded data
        save_pickle(G, path_raw)

    # load or filter processed graph
    path_proc = PROC_DIR / f"{name}.pkl"
    if path_proc.exists():
        G2 = load_pickle(path_proc)
    else:
        # remove directions and self-loops (and save)
        G2 = G.to_undirected()
        sle = list(nx.selfloop_edges(G2))
        G2.remove_edges_from(sle)
        save_pickle(G2, path_proc)

    fig, ax = ox.plot_graph(G2,
                            figsize=(W, H),
                            node_size=0,
                            edge_linewidth=edge_width,
                            edge_color='gray',
                            show=show,
                            dpi=dpi,
                            filepath=name,
                            save=output_dir is not None)

    # area = ox.gdf_from_place(place)
    # nodes, edges = ox.graph_to_gdfs(G)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # split graph in train and test subgraphs
    if split:
        split_graph(G2, name, xmin, xmax, ymin, ymax, alpha=alpha,
                    W=W, H=H, output_dir=output_dir, dpi=dpi, show=show)
