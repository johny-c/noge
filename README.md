# noge-mwe
Anonymized code for "Neural Online Graph Exploration".

## Steps to reproduce

In the root directory `noge-mwe`:

1. Install dependencies in a virtual environment. The script
   ```bash
    ./install_dependencies.sh
    ```
    creates a conda environment (named `noge-mwe`) and installs everything there.
   
2. Make data
   
    a) To procedurally generate one of the synthetic datasets (`barabasi`, `caveman`, `grid`, `ladder`, `maze`, 
   `tree`), run:
   
    ```bash
       python scripts/generate_datasets.py with dataset=\$DATASET
    ```

    b) To download and split one of the real city networks datasets (`MUC`, `OXF`, `SFO`), run:
    ```bash
       python scripts/fetch_real_datasets.py with dataset=$DATASET split=True
    ```
   
    Data will be stored in a `data/` directory as pickle files.
    The generated data sets take up ~20.5MB in total and the city networks ~28MB in total.

4. Train / Evaluate
   
    a) To train `NOGE` on a generated dataset, e.g. `grid`, run
    ```bash
       python scripts/train_dfp.py with dataset=grid
    ```

    b) To train on a real city network, e.g. `MUC`, run
    ```bash
       python scripts/train_dfp.py with dataset=MUC model.dim_hidden=128
    ```

    Runs configurations and metrics (train and test) will be stored in a `mlruns/` directory. You can browse through 
   them by running
   ```bash
      mlflow ui
   ```
