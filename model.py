import torch
import numpy as np
from torch import nn
from torch.nn.init import xavier_normal_
from tucker_riemopt import Tucker

from torch.optim import Optimizer
from tucker_riemopt import TuckerRiemannian

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class TuckER:
    def __init__(self, d, d1, d2, **kwargs):
        self.rank = (d2, d1, d1)
        self.S = torch.rand((len(d.entities), d1), device=device)
        self.R = torch.rand((len(d.relations), d2), device=device)
        self.O = torch.rand((len(d.entities), d1), device=device)
        self.W = torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)), dtype=torch.float, device=device)
        self.criterion = torch.nn.BCELoss()

    def parameters(self):
        return nn.ParameterList([self.W, self.S, self.R, self.O])

    def init(self):
        xavier_normal_(self.S.data)
        xavier_normal_(self.R.data)
        xavier_normal_(self.O.data)

    def forward(self, e_idx, r_idx):    
        relations = self.R[r_idx, :]
        subjects = self.S[e_idx, :]
        preds = torch.einsum("abc,da->dbc", self.W, relations)
        preds = torch.bmm(subjects.view(-1, 1, subjects.shape[1]), preds).view(-1, subjects.shape[1])
        preds = preds @ self.O.T
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


    def fit(self, loss_fn, model: TuckerRiemannian):
        x_k = Tucker(model.W.data, [model.S.data, model.R.data, model.O.data])
        rgrad, self.loss = TuckerRiemannian.grad(loss_fn, x_k)

        if self.momentum != 0:
            if self.b is not None:
                self.b.point = x_k
                self.b = TuckerRiemannian.project(x_k, self.b.construct())
                self.b = self.momentum * self.b + (1 - self.dumpening) * rgrad
            else:
                self.b = rgrad

            self.direction = self.b

        else:
            self.direction = rgrad
        

    @torch.no_grad()
    def step(self):
        W, S, R, O = self.param_groups[0]["params"]

        x_k = self.direction.point
        x_k = (-self.param_groups[0]["lr"]) * self.direction + TuckerRiemannian.TangentVector(x_k)
        x_k = x_k.construct().round(self.rank)

        W.data.add_(x_k.core - W)
        S.data.add_(x_k.factors[0] - S)
        R.data.add_(x_k.factors[1] - R)
        O.data.add_(x_k.factors[2] - O)


class Adam(Optimizer):
    def __init__(self, params, rank, max_lr, betas=(0.9, 0.99), eps=1e-8):
        self.rank = rank
        self.lr = max_lr
        self.betas = betas
        self.eps = eps
        
        self.b = None
        self.v = 0
        self.k = 0
        
        self.step_t = 1
        
        defaults = dict(rank=rank, lr=self.lr)
        super().__init__(params, defaults)

    def fit(self, loss_fn, model: TuckerRiemannian):
        x_k = Tucker(model.W.data, [model.S.data, model.R.data, model.O.data])
        rgrad, self.loss = TuckerRiemannian.grad(loss_fn, x_k)
        rgrad_norm = rgrad.norm()
        
        self.k = np.sqrt(1 - self.betas[1]**self.step_t)/(1 - self.betas[0]**self.step_t)
        self.v = self.betas[1]*self.v + (1 - self.betas[1])*rgrad_norm**2
        
        if self.b is not None:
            self.b.point = x_k
            self.b = TuckerRiemannian.project(x_k, self.b.construct())
            self.b = self.betas[0]*self.b + (1 - self.betas[0])*rgrad
        else:
            self.b = (1 - self.betas[0])*rgrad

        self.direction = (self.k/(torch.sqrt(self.v + self.eps))) * self.b
        

    @torch.no_grad()
    def step(self, closure=None):
        W, S, R, O = self.param_groups[0]["params"]

        x_k = self.direction.point
        x_k = (-self.param_groups[0]["lr"]) * self.direction + TuckerRiemannian.TangentVector(x_k)
        x_k = x_k.construct().round(self.rank)

        W.data.add_(x_k.core - W)
        S.data.add_(x_k.factors[0] - S)
        R.data.add_(x_k.factors[1] - R)
        O.data.add_(x_k.factors[2] - O)
        
        self.step_t += 1