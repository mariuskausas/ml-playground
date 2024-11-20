class Autoencoder(nn.Module):
    """ 
    Implementation of a basic Autoencoder.
    
    Parameters
    ----------
    input_size : int
        Size of the input dimensions, e.g., 28 * 28 
        for a flattened image of 28 x 28.
    """
    
    def __init__(self, input_size):
        super().__init__()
        self._input_size = input_size

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_size),
            nn.Sigmoid()
        )
  
    def forward(self, x):
        """ Forward pass."""
        encoded = self.encoder(x)    
        decoded = self.decoder(encoded)
        return decoded