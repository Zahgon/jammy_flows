import numpy as np
import torch
#import flows

from .cnf_mf_base import Manifold
from .utils import EPS, sindiv, divsin

"""
Code implementation of "Neural manifold ordinary differential equations" (https://arxiv.org/abs/2006.10254), 
mostly copied from https://github.com/CUAI/Neural-Manifold-Ordinary-Differential-Equations
"""


class FirstJacobianScalar(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        y = x * torch.acos(x) / (1 - x.pow(2)).pow(1.5) - 1 / (1 - x.pow(2))
        y_limit = -torch.ones_like(x) / 3
        ctx.save_for_backward(x)
        return torch.where(x > 1 - EPS[x.dtype], y_limit, y)

    @staticmethod
    def backward(ctx, g):
        pass


firstjacscalar = FirstJacobianScalar.apply


class Sphere(Manifold):

    def __init__(self):
        super(Sphere, self).__init__()

    def zero(self, *shape, out=None):
        x = torch.zeros(*shape, out=out)
        x[..., 0] = -1
        return x

    def zero_vec(self, *shape, out=None):
        pass

    def zero_like(self, x):
        pass

    def zero_vec_like(self, x):
        pass

    def inner(self, x, u, v, keepdim=False):
        return (u * v).sum(dim=-1, keepdim=keepdim)

    def proju(self, x, u, inplace=False):
        return u.addcmul(-self.inner(None, x, u, keepdim=True), x)

    def projx(self, x, inplace=False):
        return x.div(self.norm(None, x, keepdim=True))

    def exp(self, x, u):
        norm_u = u.norm(dim=-1, keepdim=True)
        return x * torch.cos(norm_u) + u * sindiv(norm_u)

    def retr(self, x, u):
        return self.projx(x + u)

    def log(self, x, y):
        xy = (x * y).sum(dim=-1, keepdim=True)
        xy.data.clamp_(min=-1 + 1e-6, max=1 - 1e-6)
        val = torch.acos(xy)
        return divsin(val) * (y - xy * x)


    def jacoblog(self, x, y):
        z = (x * y).sum(dim=-1, keepdim=True)
        z.data.clamp_(min=-1 + 1e-4, max=1 - 1e-4)

        firstterm = firstjacscalar(z.unsqueeze(-1)) * (y - z * x).unsqueeze(-1) * x.unsqueeze(-2)
        secondterm = divsin(torch.acos(z).unsqueeze(-1)) * (torch.eye(x.shape[-1]).to(x).unsqueeze(0) - x.unsqueeze(-1) * x.unsqueeze(-2))
        return firstterm + secondterm


    def dist(self, x, y, squared=False, keepdim=False):
        pass

    def rand(self, *shape, out=None, ir=1e-2):
        x = self.zero(*shape, out=out)
        u = self.randvec(x, norm=ir)
        return self.retr(x, u)

    def rand_uniform(self, *shape, out=None):
        pass

    def rand_ball(self, *shape, out=None):
        pass

    def randvec(self, x, norm=1):
        u = torch.randn(x.shape, out=torch.empty_like(x))
        u = self.proju(x, u, inplace=True)  # "transport" ``u`` to ``x``
        u.div_(u.norm(dim=-1, keepdim=True)).mul_(norm)  # normalize
        return u

    def transp(self, x, y, u):
        pass

    def __str__(self):
        return "Sphere"
    
    def sh_to_dim(self, sh):
        pass

    def dim_to_sh(self, dim):
        pass

    def squeeze_tangent(self, x):
        pass

    def unsqueeze_tangent(self, x):
        pass

    def logdetexp(self, x, u):
        norm_u = u.norm(dim=-1)
        val = torch.abs(sindiv(norm_u)).log()
        return (u.shape[-1]-2) * val
