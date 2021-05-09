from typing import Tuple

import torch
from torch import Tensor, optim
import pytorch_lightning as pl

from .modules import Encoder, Decoder, ReconLoss, LatentLoss


class VAE(pl.LightningModule):
    def __init__(self, lr: float) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.lr = lr
        self.save_hyperparameters()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)
        self.recon_loss = ReconLoss()
        self.latent_loss = LatentLoss()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encoder(x)
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

    def configure_optimizers(self) -> optim.Adam:
        return optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        recon, mu, logvar = self.forward(batch)

        recon_loss = self.recon_loss(recon, batch)
        latent_loss = self.latent_loss(mu, logvar)
        loss = recon_loss + latent_loss

        self.log("recon_loss", recon_loss.item())
        self.log("latent_loss", latent_loss.item())
        self.log("training_loss", loss.item())

        return loss
