#!/bin/bash

# Follow instructions in README
#conda create -n noge python=3.8 -y
#conda activate noge

pip install osmnx==1.0.1
pip install dataclasses==0.6
pip install tqdm==4.56.0
pip install tabulate==0.8.7
pip install mlflow==1.13.1
pip install sacred==0.8.2
pip install scipy==1.6.0
pip install scikit-learn==0.24.1

conda install pytorch==1.7.1 torchvision torchaudio cpuonly -c pytorch -y

pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
pip install torch-geometric==1.6.3

pip install gym==0.18.0
