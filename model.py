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


class SGD(Optimizer):
    def __init__(self, params, rank, lr, momentum = 0, dampening = 0, weight_decay = 0):

        self.rank = rank
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.dampening = dampening

        self.direction = None
        self.loss = None
        self.b = None

        defaults = dict(rank=rank, lr=self.lr)
        super().__init__(params, defaults)


    def fit(self, loss_fn, model: SFTuckerRiemannian):
        x_k = SFTucker(model.W.data, [model.R.data], num_shared_factors=2, shared_factor=model.E.data)
        rgrad, self.loss = SFTuckerRiemannian.grad(loss_fn, x_k)

        if self.weight_decay != 0:
            rgrad.delta_core += self.weight_decay * x_k.core
            rgrad.delta_regular_factors[0]+= self.weight_decay * x_k.factors[0]
            rgrad.delta_shared_factor += self.weight_decay * x_k.shared_factor

        if self.momentum != 0:
            if self.b is not None:
                self.b.point = x_k
                self.b = SFTuckerRiemannian.project(x_k, self.b.construct())
                self.b = self.momentum * self.b + (1 - self.dumpening) * rgrad
            else:
                self.b = rgrad

            self.direction = self.b

        else:
            self.direction = rgrad
        

    @torch.no_grad()
    def step(self):
        W, E, R = self.param_groups[0]["params"]

        x_k = self.direction.point
        x_k = (-self.param_groups[0]["lr"]) * self.direction + SFTuckerRiemannian.TangentVector(x_k)
        x_k = x_k.construct().round(self.rank)

        W.data.add_(x_k.core - W)
        R.data.add_(x_k.regular_factors[0] - R)
        E.data.add_(x_k.shared_factor - E)