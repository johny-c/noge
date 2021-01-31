# noge-mwe
Anonymized code for "Neural Online Graph Exploration".

## Steps to reproduce

In the top directory `noge-mwe`:

1. Install dependencies in a virtual environment:
   
   a) Create a virtual environment (need python 3.8):
   ```bash
      conda env create -n noge-mwe python==3.8 -y
    ```
   b) Activate the virtual environment:
   ```bash
      conda activate noge-mwe
   ```
   c) Install dependencies
   ```bash
      chmod +x install_dependencies.sh
      ./install_dependencies.sh
   ```

2. Make data
   
    Add the top directory to your `$PYTHONPATH`.

    a) To procedurally generate one of the synthetic datasets (`barabasi`, `caveman`, `grid`, `ladder`, `maze`, 
   `tree`), run:
   
    ```bash
       python scripts/generate_datasets.py with dataset=$DATASET
    ```

    b) To download and split one of the real city networks datasets (`MUC`, `OXF`, `SFO`), run:
    ```bash
       python scripts/fetch_real_datasets.py with dataset=$DATASET split=True
    ```
   
    Data will be stored in a `data/` directory as pickle files.
    The generated data sets take up ~20.5MB in total and the city networks ~28MB in total.


3. Train / Evaluate

   Make sure the top directory is in your `$PYTHONPATH`.
   
    a) To train `NOGE` on a generated dataset, e.g. `grid`, run
    ```bash
       python scripts/train_dfp.py with dataset=grid
    ```

    b) To train on a real city network, e.g. `MUC`, run
    ```bash
       python scripts/train_dfp.py with dataset=MUC model.dim_hidden=128
    ```

    c) To train `NOGE-NN`, a.k.a. add the nearest neighbor as an input feature:
    ```bash
       python scripts/train_dfp.py with dataset=MUC model.dim_hidden=128 cat_features=YN
    ```

    Runs configurations and metrics (train and test) will be stored in a `mlruns/` directory. You can inspect them 
   in your browser by running
   ```bash
      mlflow ui
   ```


## Baselines

1. To run all non-learning baselines, run
```bash
   python scripts/evaluate_baselines.py with datasets=$DATASETS
```
Make sure you have already generated/fetched the $DATASETS to be used.

For instance, to evaluate `BFS` and `RANDOM` on the `maze` and `grid` datasets, run
```bash
   python scripts/evaluate_baselines.py with datasets="['maze', 'grid']"  policies="['random','bfs']"
```

2. To train a `DQN`, run
```bash
   python scripts/train_dqn.py with dataset=$DATASET reward_type=$REWARD_TYPE
```
where $REWARD_TYPE can be `path_length` or `er_diff` (exploration rate difference).
