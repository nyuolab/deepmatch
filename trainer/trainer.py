import torch
from torch.utils.tensorboard import SummaryWriter
import time

class Trainer(object):
    """ Trains model on synthetic dataset.
    """

    def __init__(self, model, loss_function, learning_rate, data_loader, device):
        self.model = model.train()
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.data_loader = data_loader
        self.device = device
        self.step = None

        self.create_optimizer()

    def create_optimizer(self):
        """Create an optimizer with a LR scheduler."""

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        lr = self.learning_rate
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                                    mode='min',
                                                                    factor=0.1,
                                                                    patience=0,
                                                                    verbose=True,
                                                                    min_lr=1e-7,
                                                                    threshold=1e-3)


    def train(self, num_epochs, save_path, log_freq=100, save_freq=100000, load=False):
        """ Train the model.

        Arguments:
            num_epochs: integer of # of epochs to train the model.

        """

        writer = SummaryWriter(save_path)

        if load:
            self.model = self.model.load_state_dict(torch.load(save_path, map_location=self.device)).to(self.device)

        running_loss = 0
        step = 0

        # Start training
        for epoch in range(num_epochs):
            for batch_idx, batch in enumerate(self.data_loader):

                # Forward pass
                batch = batch.to(self.device).to(torch.float)
                batch_hat = self.model(batch).to(torch.float)

                # Compute reconstruction loss
                loss = self.loss_function(batch_hat, batch) # input, target

                # Backprop and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                step = epoch * len(self.data_loader) + batch_idx

                if (batch_idx % log_freq) == 1:
                    # Write loss
                    running_loss /= log_freq
                    writer.add_scalar('loss/training', running_loss, step)

                    print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}"
                            .format(epoch, num_epochs, batch_idx, len(self.data_loader), running_loss))

                    writer.flush()
                    save_loss_for_sched = running_loss
                    running_loss = 0

                if (epoch * len(self.data_loader) + batch_idx) % (save_freq) == 0:
                    save_name = (save_path + str(time.time()).split('.')[0] + '.pth')
                    torch.save(self.model.state_dict(), save_name)

            self.scheduler.step(save_loss_for_sched)
            