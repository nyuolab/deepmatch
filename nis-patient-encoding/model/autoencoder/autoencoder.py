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

        Expected 'build_params' structure:
            1. 'embedding' :
                [i.] {  embedding_name : str,
                        num_classes : int,
                        dimensions : int,
                        feature_idx : [int, ]
                    }

            2. 'one-hots' : [int, ] (inds of those that need to be one-hot'd)

            3. 'encoding' :
                [i.] {  total_layers : int
                        dimensions : int
                        activation: str

                    }

            4. 'latent' : { 'dimensions' : int }

            4. 'decoding' :
                [i. ] { total_layers : int,
                        dimensions : [int, ],
                        activation: [str, ],
                        output_dims: int,
                    }

        """
        super().__init__()

        # Store input variables.
        self.build_params = build_params

        # Create the model.
        self.create_model()

    def create_encoder(self, autoencoder):
        """ Create the encoder block in the model.
        """

        self.create_embeddings()
        autoencoder = self.create_encoding(autoencoder)
        return autoencoder

    def create_embeddings(self):
        """ Create the embeddings for model inputs.
        """

        self.dense_encoding_input_dim = 0
        self.decode_output_dim = 0

        embeddings = nn.ModuleDict()

        for embedding_name, embedding_layer_info in self.build_params['features']['embedding'].items():
            embeddings[embedding_name] = nn.Embedding(embedding_layer_info['num_classes'], embedding_layer_info['dimensions'])
            self.dense_encoding_input_dim += embedding_layer_info['dimensions'] * len(embedding_layer_info['feature_idx'])

        self.model['embeddings'] = embeddings

        self.decode_output_dim = self.dense_encoding_input_dim

        for one_hot in self.build_params['features']['one_hots'].values():
            self.dense_encoding_input_dim += one_hot['num_classes']
            self.decode_output_dim += 1

        for cont in self.build_params['features']['continuous'].values():
            self.dense_encoding_input_dim += 1
            self.decode_output_dim += 1

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
                layer = dense_reg_layer(self.dense_encoding_input_dim, dense_dim)
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

        # Construct 'de-embedding' layer
        deembedding_layer = dense_reg_layer(layer_dims[-1], self.dense_encoding_input_dim)
        decoder.append(deembedding_layer)

        # Populate the autoencoder.
        autoencoder.add_module('decoder', nn.Sequential(*decoder))

        # Add to model
        self.model['autoencoder'] = autoencoder

        # Reconstruct each feature for loss calculation later.
        reconstruction = nn.ModuleDict()
        for feature_type, type_info in self.build_params['features'].items():
            for feature, feature_info in type_info.items():
                if feature_type in ['embedding', 'one_hots']:
                    dense_unit = nn.Linear(self.dense_encoding_input_dim, feature_info['num_classes'])
                    reconstruction[feature] = dense_unit

                elif feature_type == 'continuous':
                    dense_unit = nn.Linear(self.dense_encoding_input_dim, 1)
                    reconstruction[feature] = dense_unit

        # Add to model
        self.model['reconstruction'] = reconstruction


    def create_model(self):
        """ Create the autoencoder architecture.
        """

        self.model = nn.ModuleDict()
        autoencoder = nn.Sequential()
        autoencoder = self.create_encoder(autoencoder)
        self.create_decoder(autoencoder)

    def forward(self, x):
        """Perform a forward prop of x."""

        # We'll only call this function when we're training.
        self.train()

        embedded, ground_truth = self.embed(x)
        decoded = self.model.autoencoder(embedded)

        reconstruction = {}
        for recon_feature, recon_layer in self.model['reconstruction'].items():
            reconstruction[recon_feature] = recon_layer(decoded)

        return reconstruction, ground_truth

    def embed(self, x):
        """Modify the features as needed."""

        # Initalize stores
        pre_encoding = []
        ground_truth = {}

        for embedding, embedding_layer in self.build_params['features']['embedding'].items():
            features_to_use = embedding_layer['feature_idx']
            emb_flat_size = len(features_to_use) * embedding_layer['dimensions']
            batch_emb = x[:, features_to_use].to(torch.long)
            embedded = self.model['embeddings'][embedding](batch_emb).view(-1, emb_flat_size)
            
            ground_truth[embedding] = nn.functional.one_hot(batch_emb, num_classes=embedding_layer['num_classes'])
            ground_truth[embedding] = (torch.sum(ground_truth[embedding], axis=1) > 0).to(torch.float)
            pre_encoding.append(embedded)

        for one_hot_name, one_hot in self.build_params['features']['one_hots'].items():
            nc = one_hot['num_classes']
            one_hot_encoded = nn.functional.one_hot(x[:, one_hot['feature_idx']].to(torch.long), num_classes=nc)
            
            pre_encoding.append(one_hot_encoded.to(torch.float32))
            ground_truth[one_hot_name] = one_hot_encoded.to(torch.float)

        for cont_name, continuous_feature in self.build_params['features']['continuous'].items():
            cont = x[:, continuous_feature['feature_idx']].view(-1, 1)
            pre_encoding.append(cont)
            ground_truth[cont_name] = cont

        # Save for future exploratory purposes.
        self.pre_encoding = pre_encoding
        pre_encoding = torch.cat(self.pre_encoding, axis=1)

        return pre_encoding, ground_truth

    def predict(self, x, return_targets=True):
        """Evaluate the model."""
        
        # Set to evaluation mode.
        self.eval()

        with torch.no_grad():
            
            # Forward pass.
            recons, targets = self.forward(x)

            # Sigmoid activation for all relevant recon layers
            for feature, recon in recons.items():
                recons[feature] = nn.functional.sigmoid(recon)

        # Return
        if return_targets:
            return recons, targets
        else:
            return recons

    def latent_representation(self, x):
        """Represent the patient in the latent space."""
        
        # Set to evaluation
        self.eval()

        with torch.no_grad():
            # Create embedding
            embedded, _ = self.embed(x)
            return self.model.autoencoder.encoder(embedded)

    def load_state(self, path, device='cpu', prediction=False):
        """Load model parameters."""

        self.load_state_dict(torch.load(path, map_location=device))
        if prediction:
            self.eval()

    def format_ground_truth(self, x):
        """Modify the features as needed."""

        # Initalize stores
        pre_encoding = []
        ground_truth = {}

        for embedding, embedding_layer in self.build_params['features']['embedding'].items():
            features_to_use = embedding_layer['feature_idx']
            emb_flat_size = len(features_to_use) * embedding_layer['dimensions']
            batch_emb = x[:, features_to_use].to(torch.long)
            
            ground_truth[embedding] = nn.functional.one_hot(batch_emb, num_classes=embedding_layer['num_classes'])
            ground_truth[embedding] = (torch.sum(ground_truth[embedding], axis=1) > 0).to(torch.float)

        for one_hot_name, one_hot in self.build_params['features']['one_hots'].items():
            nc = one_hot['num_classes']
            one_hot_encoded = nn.functional.one_hot(x[:, one_hot['feature_idx']].to(torch.long), num_classes=nc)
            
            ground_truth[one_hot_name] = one_hot_encoded.to(torch.float)

        for cont_name, continuous_feature in self.build_params['features']['continuous'].items():
            cont = x[:, continuous_feature['feature_idx']].view(-1, 1)
            ground_truth[cont_name] = cont

        return ground_truth