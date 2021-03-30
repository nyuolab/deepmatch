import numpy as np
import torch
import torch.nn as nn

ACTIVATION_KEY = {
    'relu' : nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'tanh' : nn.Tanh,
    'swish' : (lambda x : x.mul_(nn.Sigmoid(x))),
    'leaky_relu' : nn.LeakyReLU,
}

def dense_reg_layer(input_features, output_features, dropout_rate=0.0, batch_norm=True):
    """Create a standard layer for encoder/decoder sequence.

    Construct a torch.nn.Sequential Linear layer with a LeakyReLU activation, BatchNom, and Dropout.
    
    """
    layers = [nn.Linear(input_features, output_features), nn.LeakyReLU(inplace=True)]
    
    if batch_norm:
        layers.append(nn.BatchNorm1d(output_features))

    layers.append(nn.Dropout(dropout_rate))
    return nn.Sequential(*layers)


class AutoEncoder(nn.Module):
    """ Autoencoder class

    <Describe the autoencoder>

    Attributes:
        something
        else

    Methods:
        something
        else

    """
    def __init__(self, build_params, **kwargs):
        """ Initialize the AutoEncoder.
        """
        super().__init__()

        # Store input variables.
        self.build_params = build_params

        # Create the model.
        self.create_model()

    def create_encoder(self, autoencoder):
        """ Create the encoder block in the model.
        """

        autoencoder = self.create_encoding(autoencoder)
        return autoencoder

    def create_encoding(self, autoencoder):
        """ Create the encoding layers of the autoencoder.
        """

        total_layers = self.build_params['encoding']['total_layers']

        if self.build_params['encoding'].get('scale') is not None:
            scale = self.build_params['encoding'].get('scale')
            latent_dims = self.build_params['latent']['dimensions']
            layer_dims = [latent_dims * (scale**(total_layers - layer_idx)) for layer_idx in range(total_layers)]
        else:
            dense_dim = self.build_params['encoding']['dimensions']
            layer_dims = [dense_dim for _ in range(self.build_params['encoding']['total_layers'])]
        
        # Create empty encoder for storing layers
        encoder = []

        # Construct linear + activation layers
        for layer_idx, dense_dim in enumerate(layer_dims):
            if layer_idx == 0:
                layer = dense_reg_layer(self.build_params['data']['input_features'], dense_dim)
            else:
                layer = dense_reg_layer(previous_dim, dense_dim)
            
            previous_dim = dense_dim
            encoder.append(layer)

        latent_dims = self.build_params['latent']['dimensions']
        latent_layer = dense_reg_layer(layer_dims[-1], latent_dims)
        encoder.append(latent_layer)

        # Populate the autoencoder.
        autoencoder.add_module('encoder', nn.Sequential(*encoder))
        
        return autoencoder

    def create_decoder(self, autoencoder):
        """ Create the decoder block in the model.
        """

        total_layers = self.build_params['encoding']['total_layers']

        if self.build_params['encoding'].get('scale') is not None:
            scale = self.build_params['encoding'].get('scale')
            latent_dims = self.build_params['latent']['dimensions']
            layer_dims = [latent_dims * (scale**(layer_idx + 1)) for layer_idx in range(total_layers)]
        else:
            dense_dim = self.build_params['encoding']['dimensions']
            layer_dims = [dense_dim for _ in range(self.build_params['encoding']['total_layers'])]
        
        # Create empty decoder for storage
        decoder = []

        # Construct linear + activation layers
        for layer_idx, dense_dim in enumerate(layer_dims):
            if layer_idx == 0:
                layer = dense_reg_layer(self.build_params['latent']['dimensions'], dense_dim)
            else:
                layer = dense_reg_layer(previous_dim, dense_dim)
            
            previous_dim = dense_dim
            decoder.append(layer)

        # Final Decoding Layer
        final_decode_layer = dense_reg_layer(layer_dims[-1], self.build_params['data']['input_features'])
        decoder.append(final_decode_layer)

        # Populate the autoencoder.
        autoencoder.add_module('decoder', nn.Sequential(*decoder))

        # Add to model
        self.model['autoencoder'] = autoencoder


    def create_model(self):
        """ Create the autoencoder architecture.
        """

        self.model = nn.ModuleDict()
        autoencoder = nn.Sequential()
        autoencoder = self.create_encoder(autoencoder)
        self.create_decoder(autoencoder)
        

    def forward(self, x, is_training=True):
        """Perform a forward prop of x."""

        # We'll only call this function when we're training.
        if is_training:
            self.train()
        
        return self.model.autoencoder(x)
    

    def predict(self, x, return_targets=True):
        """Evaluate the model."""
        
        # Set to evaluation mode.
        self.eval()

        with torch.no_grad():
            
            # Forward pass.
            recon = self.forward(x, is_training=False)

        return recon

    def latent_representation(self, x):
        """Represent the patient in the latent space."""
        
        # Set to evaluation
        self.eval()

        with torch.no_grad():
            return self.model.autoencoder.encoder(x)

    def load_state(self, path, device='cpu', prediction=False):
        """Load model parameters."""

        self.load_state_dict(torch.load(path, map_location=device))
        if prediction:
            self.eval()