import torch
import numpy as np
from torch import nn
from torch.nn.init import xavier_normal_
from tucker_riemopt import SFTucker

from torch.optim import Optimizer
from tucker_riemopt import SFTuckerRiemannian


class SFTuckER:
    def __init__(self, d, d1, d2, **kwargs):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.rank = (d2, d1, d1)
        self.E = torch.rand((len(d.entities), d1), device=device)
        self.R = torch.rand((len(d.relations), d2), device=device)
        self.W = torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)), dtype=torch.float, device=device)
        self.criterion = torch.nn.BCELoss()
    def parameters(self):
        return nn.ParameterList([self.W, self.E, self.R])

    def init(self):
        xavier_normal_(self.E.data)
        xavier_normal_(self.R.data)
        # with torch.no_grad():
        #     self.E.weight.data = torch.linalg.qr(self.E.weight)[0]
        #     self.R.weight.data = torch.linalg.qr(self.R.weight)[0]

    def forward(self, e_idx, r_idx):
        relations = self.R[r_idx, :]
        subjects = self.E[e_idx, :]
        preds = torch.einsum("abc,da->dbc", self.W, relations)
        preds = torch.bmm(subjects.view(-1, 1, subjects.shape[1]), preds).view(-1, subjects.shape[1])
        preds = preds @ self.E.T
        return torch.sigmoid(preds)


class RGD(Optimizer):
    def __init__(self, model_parameters, rank, max_lr):
        self.rank = rank
        self.max_lr = max_lr
        self.lr = max_lr
        self.direction = None
        self.loss = None

        defaults = dict(rank=rank, max_lr=self.max_lr, lr=self.lr)
        params = model_parameters
        super().__init__(params, defaults)

    def fit(self, loss_fn, model, normalize_grad=False):
        x_k = SFTucker(model.W.data, [model.R.data], num_shared_factors=2, shared_factor=model.E.data)
        rgrad, self.loss = SFTuckerRiemannian.grad(loss_fn, x_k)
        rgrad_norm = rgrad.norm().detach()

        if normalize_grad:
            normalizer = normalize_grad / rgrad_norm
        else:
            normalizer = 1

        self.direction = normalizer * rgrad
        return rgrad_norm

    @torch.no_grad()
    def step(self):
        W, E, R = self.param_groups[0]["params"]

        x_k = self.direction.point
        x_k = (-self.param_groups[0]["lr"]) * self.direction + SFTuckerRiemannian.TangentVector(x_k)
        x_k = x_k.construct().round(self.rank)

        W.data.add_(x_k.core - W)
        R.data.add_(x_k.regular_factors[0] - R)
        E.data.add_(x_k.shared_factor - E)


class RSGDwithMomentum(RGD):
    def __init__(self, params, rank, max_lr, momentum_beta=0.9):
        super().__init__(params, rank, max_lr)
        self.momentum_beta = momentum_beta
        self.momentum = None

    def fit(self, loss_fn, x_k: SFTuckerRiemannian, normalize_grad=False):
        if self.direction is not None:
            self.momentum = SFTuckerRiemannian.project(x_k, self.direction)
        else:
            self.momentum = SFTuckerRiemannian.TangentVector(x_k, torch.zeros_like(x_k.core))
        rgrad, self.loss = SFTuckerRiemannian.grad(loss_fn, x_k)
        rgrad_norm = rgrad.norm().detach()

        if normalize_grad:
            normalizer = normalize_grad / rgrad_norm
        else:
            normalizer = 1

        self.direction = normalizer * rgrad + self.momentum_beta * self.momentum
        return rgrad_norm

    @torch.no_grad()
    def step(self):
        W, E, R = self.param_groups[0]["params"]

        x_k = self.direction.point
        x_k = (-self.param_groups[0]["lr"]) * self.direction + SFTuckerRiemannian.TangentVector(x_k)
        x_k = x_k.construct().round(self.rank)
        self.direction = self.direction.construct()

        W.data.add_(x_k.core - W)
        R.data.add_(x_k.regular_factors[0] - R)
        E.data.add_(x_k.shared_factor - E)
