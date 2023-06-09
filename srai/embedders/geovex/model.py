import torch
from torch import nn
import torch.nn.functional as F
from torch import sigmoid
import pytorch_lightning as pl
from typing import List, Tuple
from torchmetrics.functional import f1_score as f1


# This is based on https://openreview.net/forum?id=7bvWopYY1H


def build_mask_funcs(R):
    def w_dist(i, j):
        r = max(abs(i - R), abs(j - R), abs(i - j))
        return 1 / (1 + r) if r <= R else 0

    def w_num(i, j):
        r = max(abs(i - R), abs(j - R), abs(i - j))
        return 1 / (R * r) if r <= R and r > 0 else 1 if r == 0 else 0

    return w_dist, w_num


class GeoVeXLoss(nn.Module):
    def __init__(self, R):
        super(GeoVeXLoss, self).__init__()
        self.R = R
        self._w_dist, self._w_num = build_mask_funcs(self.R)

        M = 2 * self.R + 1
        self._w_dist_matrix = torch.tensor(
            [[self._w_dist(i, j) for j in range(M)] for i in range(M)],
            dtype=torch.float32,
        )
        self._w_num_matrix = torch.tensor(
            [[self._w_num(i, j) for j in range(M)] for i in range(M)],
            dtype=torch.float32,
        )

    def forward(self, pi, lambda_, y):
        # trim the padding from y
        y = y[:, :, :2 * self.R + 1,  :2 * self.R + 1]

        I0 = (y == 0).float()
        I_greater_0 = (y > 0).float()

        # torch.exp(-1 * lambda_) instead of torch.exp(lambda_). the paper has a typo, I think...
        log_likelihood_0 = I0 * torch.log(pi + (1 - pi) * torch.exp(-1 * lambda_))
        log_likelihood_greater_0 = I_greater_0 * (
            torch.log(1 - pi)
            - lambda_
            + y * torch.log(lambda_)
            - torch.lgamma(y + 1)  # this is the ln(factorial(y))
        )

        log_likelihood = log_likelihood_0 + log_likelihood_greater_0
        loss = -torch.sum(log_likelihood * self._w_dist_matrix * self._w_num_matrix) / (
            torch.sum(self._w_dist_matrix) * torch.sum(self._w_num_matrix)
        )

        return loss


class HexagonalConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=2, padding=0, bias=True, groups=1
    ):
        super(HexagonalConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=groups
        )
        self.register_buffer("hexagonal_mask", self.create_hexagonal_mask())
        # relu is applied after the transpose convolution

    def create_hexagonal_mask(self):
        mask = torch.tensor([[0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=torch.float32)
        return mask

    def forward(self, x):
        self.conv.weight = nn.Parameter(self.conv.weight * self.hexagonal_mask)
        out = self.conv(x)
        return out


class HexagonalConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        padding=0,
        output_padding=0,
        bias=True,
    ):
        super(HexagonalConvTranspose2d, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            bias=bias,
        )
        self.register_buffer("hexagonal_mask", self.create_hexagonal_mask())

    def create_hexagonal_mask(self):
        mask = torch.tensor([[0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=torch.float32)
        return mask

    def forward(self, x):
        self.conv_transpose.weight.data *= self.hexagonal_mask
        out = self.conv_transpose(x)
        return out


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class GeoVeXZIP(nn.Module):
    def __init__(self, in_dim, r, out_dim):
        super(GeoVeXZIP, self).__init__()

        # comes in as (batch_size, k_dim, R, R)
        self.in_dim = in_dim
        self.R = r
        self.out_dim = out_dim
        self.pi = nn.Linear(in_dim, out_dim)
        self.lambda_ = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        _x = x.view(-1, self.R, self.R, self.in_dim)
        pi = torch.sigmoid(self.pi(_x))
        # clamp pi to avoid nan's
        pi = pi.view(-1, self.out_dim, self.R, self.R,)
        lambda_ = torch.exp(self.lambda_(_x)).view(-1, self.out_dim, self.R, self.R,)

        # pad by 1 on the right and bottom
        # pi = F.pad(pi, (0, 1, 0, 1), mode="constant", value=1e-5)
        # lambda_ = F.pad(lambda_, (0, 1, 0, 1), mode="constant", value=0)
        pi = torch.clamp(pi, 1e-5, 1 - 1e-5)
        return pi, lambda_
    

class GeoVexModel(pl.LightningModule):
    def __init__(self, k_dim, R, emb_size=32, lr=1e-5, weight_decay=1e-5):
        super().__init__()

        self.k_dim = k_dim
        self.R = R
        self.lr = lr
        self.weight_decay = weight_decay
        self.emb_size = emb_size

        num_conv = 1
        lin_size = 4  # self.R // (2 ** num_conv)

        self.encoder = nn.Sequential(
            nn.BatchNorm2d(self.k_dim),
            nn.ReLU(),
            # nn.Conv1d(self.k_dim, 256, kernel_size=3, stride=2),
            # have to add padding to preserve the input size
            HexagonalConv2d(self.k_dim, 256, kernel_size=3, stride=2, padding=6),
            # # HexagonalConv2d(self.k_dim, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            HexagonalConv2d(256, 512, kernel_size=3, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # HexagonalConv2d(512, 1024, kernel_size=3, stride=2),
            # nn.BatchNorm2d(1024),
            # nn.ReLU(),
            nn.Flatten(),
            # # # TODO: make this a function of R
            nn.Linear(lin_size * lin_size * 512, self.emb_size),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.emb_size, lin_size * lin_size * 512),
            # maintain the batch size, but reshape the rest
            Reshape((-1, 512, lin_size, lin_size)),
            # HexagonalConvTranspose2d(1024, 512, kernel_size=3, stride=2),
            # # HexagonalConvTranspose2d(512, 256, kernel_size=3, stride=2),
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            HexagonalConvTranspose2d(512, 256, kernel_size=3, stride=2, ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # HexagonalConvTranspose2d(256, 256, kernel_size=3, stride=2),
            # nn.BatchNorm2d(256),
            # nn.ReLU(),
            # HexagonalConvTranspose2d(256, self.k_dim, kernel_size=3, stride=2),
            # nn.BatchNorm2d(self.k_dim),
            # nn.ReLU(),
            # nn.ReLU(),
            # ,
            # nn.BatchNorm2d(self.k_dim),
            # nn.ReLU(),
            GeoVeXZIP(256, self.R * 2 + 1, self.k_dim),
        )

        self._loss = GeoVeXLoss(self.R)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        x = batch
        pi, lambda_ = self(x)
        loss = self._loss.forward(pi, lambda_, x)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        pi, lambda_ = self(x)
        loss = self._loss.forward(pi, lambda_, x)
        self.log("validation_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            # weight_decay=self.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr * 10,
            anneal_strategy="cos",
            steps_per_epoch=5,
            epochs=25,
            verbose=True,
        )
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer,
        #     gamma=0.98,
        #     verbose=True,
        # )
        return [optimizer], [lr_scheduler]
        # return optimizer
        # return optim
