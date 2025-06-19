import os
from torch import optim, nn
import lightning as L
import torch
import torch.nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

class LitAutoEncoder(L.LightningModule):
    def __init__(self, num_inputs, comp_coeff=9, light_dimension=2, num_outputs=3, pixel_position=2):
        super().__init__()
        self.num_inputs = num_inputs  # number of input, for RTI will be most probably 3 times the number of source images
        self.comp_coeff = comp_coeff  # number of computed coefficients, per pixel, in the latent space
        self.light_dimension = light_dimension  # number of light dimensions for RTI: 2 -> (x,y) | 3 -> (x,y,z)
        self.num_outputs = num_outputs  # number of outputs from the decoder, for RTI is commonly 3 (RGB channels)
        self.encoder, self.decoder = self.autoencoder()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        loss = self.common_step(batch, batch_idx)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, prog_bar=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self.common_step(batch, batch_idx)
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss)
        return {"loss": loss}

    def predict_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("pred_loss", loss)
        return {"loss": loss}

    def common_step(self, batch, batch_idx):
        x, y, g = batch

        x = x.view(x.size(0), -1)

        z = self.encoder(x)
        zy = torch.cat((z, y), dim=1)
        x_hat = self.decoder(zy)
        loss = nn.functional.mse_loss(x_hat, g)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-2, amsgrad=True)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"  # The metric to monitor
             }
        }
        #return optimizer

    def on_train_epoch_end(self):
        epoch = self.current_epoch  # Get current epoch number
        print(f"\nTraining completed for epoch {epoch}")  # Print to console
    def on_train_end(self, ):
        print("Training is done.")

    def autoencoder(self):
        encoder = nn.Sequential(nn.Linear(self.num_inputs, 150),
                                nn.ELU(),
                                nn.Linear(150, 150),
                                nn.ELU(),
                                nn.Linear(150, 150),
                                nn.ELU(),
                                nn.Linear(150, self.comp_coeff))

        decoder = nn.Sequential(nn.Linear(self.comp_coeff + self.light_dimension,50),
                                nn.ELU(),
                                nn.Linear(50, 50),
                                nn.ELU(),
                                nn.Linear(50, 3))
        return encoder, decoder
