import numpy as np
import networkx as nx


def make_maze(width=81, height=51, complexity=.75, density=.75):
    r"""Generate a random maze array.

    It only contains two kind of objects, obstacle and free space. The numerical value for obstacle
    is ``1`` and for free space is ``0``.

    Code from https://en.wikipedia.org/wiki/Maze_generation_algorithm

    >>> make_maze(10, 10)
    array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
           [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
           [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
           [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
           [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
           [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
           [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
           [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
           [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=uint8)
    """
    # Only odd shapes
    shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
    # Adjust complexity and density relative to maze size
    complexity = int(complexity * (5 * (shape[0] + shape[1])))
    density = int(density * ((shape[0] // 2) * (shape[1] // 2)))
    # Build actual maze
    Z = np.zeros(shape, dtype=bool)
    # Fill borders
    Z[0, :] = Z[-1, :] = 1
    Z[:, 0] = Z[:, -1] = 1
    # Make aisles
    for i in range(density):
        x, y = np.random.randint(0, shape[1] // 2 + 1) * 2, np.random.randint(0, shape[0] // 2 + 1) * 2
        Z[y, x] = 1
        for j in range(complexity):
            neighbours = []
            if x > 1:             neighbours.append((y, x - 2))
            if x < shape[1] - 2:  neighbours.append((y, x + 2))
            if y > 1:             neighbours.append((y - 2, x))
            if y < shape[0] - 2:  neighbours.append((y + 2, x))
            if len(neighbours):
                y_, x_ = neighbours[np.random.randint(0, len(neighbours))]
                if Z[y_, x_] == 0:
                    Z[y_, x_] = 1
                    Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                    x, y = x_, y_

    return Z.astype(np.uint8)


def maze_to_graph(maze):
    """We use the convention that the maze cell at row y, col x
        has label(x, y) = y*num_cols + x.
        Equivalently, the node with label u corresponds to the
        maze cell at row y, col x = label // num_cols, label % num_cols

    :param maze: np.array of shape (num_rows, num_cols)
    :return: nx.Graph
    """

    n_rows, n_cols = maze.shape

    # Find free cells
    free_mask = maze == 0

    # Grid has implicit naming 0, ..., N-1 \\ N, N+2, ..., 2N-1 \\ 2N, ...
    # node_id = row * n_cols + col

    # horizontal neighbors (first N-1 cols to last N-1 cols)
    horizontal_edges_mask = np.logical_and(free_mask[:, :-1], free_mask[:, 1:])  # (N, M-1)

    # nodes at (y, x) are connected to (y, x+1)
    yy, xx = np.where(horizontal_edges_mask)
    node_id = yy * n_cols + xx
    edges = [(v, v + 1) for v in node_id]

    # vertical neighbors (first N-1 rows to last N-1 rows)
    vertical_edges_mask = np.logical_and(free_mask[:-1], free_mask[1:])  # (N-1, M)

    # nodes at (y, x) are connected to (y+1, x)
    yy, xx = np.where(vertical_edges_mask)
    node_id = yy * n_cols + xx
    edges.extend([(v, v + n_cols) for v in node_id])

    graph = nx.from_edgelist(edges)

    return graph


def maze_to_pos(maze):
    n_rows, n_cols = maze.shape
    yy, xx = np.where(maze == 0)  # yy = row, xx = col
    node_id = yy * n_cols + xx
    pos = {i: (x, y) for i, y, x in zip(node_id, yy, xx)}
    return pos
