import numpy as np
import torch

class SimilarPatientIdentifier(object):
    """Class that handles Patient Similarity finding."""
    def __init__(self, data, model):
        self.data_loader = data
        self.model = model

        self.create_index()

    def create_index(self):
        """Create a kdtree for patients."""
        
        self.model.eval()
        latent_dims = self.model.build_params['latent']['dimensions']
        
        # self.index = faiss.index_factory(latent_dims, "PCA64,IVF16384_HNSW32,Flat")
        self.index = faiss.IndexFlatL2(latent_dims)

        for batch in self.data_loader:
            batch_ls = self.model.latent_representation(batch_ls).detach()
            self.index.add(batch_ls)

    def find_similar_patients(self, patient, num_similar):
        """Find n similar patients to a given input patient."""
        
        patient_ls = self.model.latent_representation(patient.reshape(-1, 1)).detach()
        distances, similar_idxs = self.index.search(patient_ls, num_similar)
        
