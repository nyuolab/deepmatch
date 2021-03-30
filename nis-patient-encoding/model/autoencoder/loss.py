import torch
import torch.nn.functional as F
import torch.nn as nn

def loss_mse(output, target):
    """ Compute the MSE loss.
    """

    return F.mse_loss(output, target)

class CustomLoss(nn.modules.loss._Loss):
    """
    """

    def __init__(self):
        super().__init__()

        self.bcel = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(self, recons, targets):

        loss = 0
        for feature, recon in recons.items():
            target = targets[feature]

            if feature == 'AGE' or feature == 'age':
                loss += self.mse(recon, target)
            else:
                loss += self.bcel(recon, target)
        
        return loss