import torch
import numpy as np
from torch import nn
from torch.nn.init import xavier_normal_
from tucker_riemopt import SFTucker

from torch.optim import Optimizer
from tucker_riemopt import SFTuckerRiemannian

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class SFTuckER:
    def __init__(self, d, d1, d2, **kwargs):
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

    def fit(self, loss_fn, model):
        x_k = SFTucker(model.W.data, [model.R.data], num_shared_factors=2, shared_factor=model.E.data)
        rgrad, self.loss = SFTuckerRiemannian.grad(loss_fn, x_k)
        rgrad_norm = rgrad.norm().detach()

        self.direction = rgrad
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
    def __init__(self, params, rank, max_lr, weight_decay = 0, momentum = 0.9, dampening = 0):
        super().__init__(params, rank, max_lr)

        self.b = None
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.dampening = dampening

    def fit(self, loss_fn, model: SFTuckerRiemannian):
        x_k = SFTucker(model.W.data, [model.R.data], num_shared_factors=2, shared_factor=model.E.data)
        rgrad, self.loss = SFTuckerRiemannian.grad(loss_fn, x_k)
        rgrad_norm = rgrad.norm().detach()

        if self.b is not None:
            self.b = SFTuckerRiemannian.project(x_k, self.b)
            self.b = self.momentum * self.b + (1 - self.momentum) * rgrad
        else:
            self.b = rgrad

        self.direction = self.b
        self.b = self.b.construct()

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


class SFTuckerAdam(RGD):
    def __init__(self, params, rank, max_lr, betas=(0.9, 0.999), eps=1e-8, step_velocity=1):
        super().__init__(params, rank, max_lr)
        self.betas = betas
        self.eps = eps
        self.step_velocity = step_velocity
        
        self.momentum = None
        self.second_momentum = torch.zeros(1, device="cuda")
        
        self.step_t = 1

    def fit(self, loss_fn, model, normalize_grad = 1.):
        x_k = SFTucker(model.W.data, [model.R.data], num_shared_factors=2, shared_factor=model.E.data)
        rgrad, self.loss = SFTuckerRiemannian.grad(loss_fn, x_k)
        rgrad_norm = rgrad.norm().detach()
        if self.momentum is not None:
            self.momentum = SFTuckerRiemannian.project(x_k, self.momentum.construct())
            self.momentum = self.betas[0] * self.momentum + (1 - self.betas[0]) * rgrad
        else:
            self.momentum = (1 - self.betas[0]) * rgrad
        self.second_momentum = self.betas[1] * self.second_momentum + (1 - self.betas[1]) * rgrad_norm ** 2
        second_momentum_corrected = self.second_momentum / (1 - self.betas[1] ** (self.step_t // self.step_velocity + 1))
        bias_correction_ratio = (1 - self.betas[0] ** (self.step_t // self.step_velocity + 1)) * torch.sqrt(
            second_momentum_corrected
        ) + self.eps
        self.direction = (1 / bias_correction_ratio) * self.momentum
        return rgrad_norm