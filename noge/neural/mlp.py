import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dims, nonlinearity='relu', alpha=0.2, activate_last=False, dropout=0):
        super().__init__()

        assert len(dims) >= 2

        nonlinearity = nonlinearity or 'identity'
        activations = nn.ModuleDict({
            'relu': nn.ReLU(),
            'lrelu': nn.LeakyReLU(alpha),
            'identity': nn.Identity()
        })

        layers = []
        for d1, d2 in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d1, d2))
            layers.append(activations[nonlinearity])
            if dropout:
                layers.append(nn.Dropout(dropout))

        if not activate_last:
            num_extra = 2 if dropout else 1
            layers = layers[:-num_extra]

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
