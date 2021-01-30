#!/bin/bash

#conda create -n noge-mwe python=3.8 -y
#conda activate noge-mwe

pip install osmnx
pip install dataclasses
pip install tqdm
pip install tabulate
pip install mlflow
pip install sacred

conda install pandas -y
conda install scikit-learn -y
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
pip install torch-geometric

pip install gym
