conda create --name noge-mwe python=3.6
conda activate noge-mwe
pip install osmnx
pip install dataclasses
pip install tqdm
pip install tabulate
pip install mlflow
pip install sacred
pip install gym
conda install pandas
conda install scikit-learn
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
pip install torch-geometric
