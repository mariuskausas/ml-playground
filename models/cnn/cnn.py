# Import libraries 
from torch import nn


class CNN(nn.Module):
    """ 
    Implementation of a general Convolutional Neural Network.

    Building blocks: N x CONV -> RELU; N x MAXPOOL; FLATTEN; 2 x FC; SOFTMAX

    Model inspiration is based on VGG: https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py#L308

    Parameters
    ----------
    network_params : dict
        A dictionary of CNN parameters.

        # Example configuration of a basic CNN
        configuration = {
            "input_size" : 28, # 28 * 28
            "initial_number_of_channels" : 1,
            "layers": [32, "maxpool", 32, "maxpool"],
            "size_of_fc" : 128,
            "num_of_classes" : 10,
            "batch_norm" : True,
        }
    """
    
    def __init__(self, configuration):
        
        super().__init__()

        self._input_size = configuration["input_size"]
        self._number_of_channels = configuration["initial_number_of_channels"]
        self._layers = configuration["layers"]
        self._num_of_maxpool_layers = self._layers.count("maxpool")
        self._size_of_fc = configuration["size_of_fc"]
        self._num_of_classes = configuration["num_of_classes"]
        self._batch_norm = configuration["batch_norm"]

        self._cnn_block = self._define_layers()
        self._flatten = nn.Flatten(1)
        self._classifier = self._define_classifier()

    def _define_layers(self):
        
        layers = []
        for l in self._layers:
            if l == "maxpool":
                layers.append(
                    nn.MaxPool2d(
                        kernel_size=(2, 2), 
                        stride=(2, 2), 
                    ))
            else:
                layers.append(
                    nn.Conv2d(
                        in_channels=self._number_of_channels,
                        out_channels=l,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding="same"
                    ))
                if self._batch_norm:
                    layers.append(nn.BatchNorm2d(num_features=l))
                layers.append(nn.ReLU())
                self._number_of_channels = l
                
        return nn.Sequential(*layers)

    def _define_classifier(self):
        """ Define a final set of FCNN + Softmax layer."""
        output_size = int(self._input_size / 2 ** self._num_of_maxpool_layers)
        classifier = nn.Sequential(
            nn.Linear(
                self._number_of_channels * output_size * output_size, self._size_of_fc),
            nn.Linear(self._size_of_fc, self._num_of_classes),
            nn.Softmax(dim=1),
        )
        return classifier

    def forward(self, x):
        """ Forward pass."""
        x = self._cnn_block(x)    
        x = self._flatten(x)
        x = self._classifier(x)
        return x
