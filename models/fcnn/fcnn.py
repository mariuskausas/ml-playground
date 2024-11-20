# Import libraries 
from torch import nn


class LLayerFCNN(nn.Module):
    """ 
    Implementation of a general L-layer fully connected neural network.

    Model architecture: FC -> FC (x N times) -> Sigmoid
    
    Parameters
    ----------
    network_params : list
        A list of lists containing L-layer FCNN parameters.

        # [in_f, out_f, activation, dropout_prob]
        network_params = [
            [2, 10, "tanh", 0.0],  # 1st FC layer
            [10, 10, "tanh", 0.0],  # 2nd FC layer
            [10, 1, "none", "none"],  # Last FC layer
        ]
    """

    def __init__(self, network_params):
        super().__init__()

        self.activations = nn.ModuleDict([
            ['sigmoid', nn.Sigmoid()],
            ['tanh', nn.Tanh()],
            ['relu', nn.ReLU()],
        ])

        # Initiate a set of linear blocks
        blocks = []
        for L_param in network_params[:-1]:
            blocks.append(self._linear_block(
                in_features=L_param[0], 
                out_features=L_param[1], 
                activation=L_param[2],
                dropout_prob=L_param[3],
            ))
        self.block = nn.Sequential(*blocks)
        
        # Initiate a last linear block
        self.linear = nn.Linear(
            in_features=network_params[-1][0], 
            out_features=network_params[-1][1]
        )

        # Sigmoid as a final layer
        self.sigmoid = nn.Sigmoid()

    def _linear_block(self, in_features, out_features, activation="relu", dropout_prob=0.0):
        """ Define a linear block."""
        return nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.Dropout(p=dropout_prob),
            self.activations[activation],
        )
        
    def forward(self, x):
        """ Forward pass."""
        x = self.block(x)
        x = self.linear(x)
        output = self.sigmoid(x)
        return output
