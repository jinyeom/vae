from typing import Tuple

import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F
import pytorch_lightning as pl


class Encoder(nn.Module):
    def __init__(self, z_dim: int) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.fc_mu = nn.Linear(1024, z_dim)
        self.fc_logvar = nn.Linear(1024, z_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, z_dim: int) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.fc = nn.Linear(z_dim, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc(x), inplace=True)
        x = x.view(x.size(0), x.size(1), 1, 1)
        x = F.relu(self.deconv1(x), inplace=True)
        x = F.relu(self.deconv2(x), inplace=True)
        x = F.relu(self.deconv3(x), inplace=True)
        x = torch.sigmoid(self.deconv4(x))
        return x


def softclip(tensor: Tensor, min_value: float) -> Tensor:
    return min_value + F.softplus(tensor - min_value)


def gaussian_nll(mu: Tensor, log_sigma: Tensor, x: Tensor) -> Tensor:
    log_a = log_sigma + 0.5 * np.log(2 * np.pi)
    sigma = log_sigma.exp()
    return log_a + 0.5 * torch.pow((x - mu) / sigma, 2)


class ReconLoss(nn.Module):
    def forward(self, pred: Tensor, targ: Tensor) -> Tensor:
        log_sigma = (targ - pred).pow(2).mean(keepdim=True).sqrt().log()
        log_sigma = softclip(log_sigma, -6)
        loss = gaussian_nll(pred, log_sigma, targ).sum()
        return loss


class LatentLoss(nn.Module):
    def forward(self, mu: Tensor, logvar: Tensor) -> Tensor:
        loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return loss


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
