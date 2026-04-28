import abc
import torch
import torch.autograd.functional as AF
import numpy as np

from .utils import EPS


"""
Code implementation of "Neural manifold ordinary differential equations" (https://arxiv.org/abs/2006.10254), 
mostly copied from https://github.com/CUAI/Neural-Manifold-Ordinary-Differential-Equations
"""


class Manifold(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def zero(self, *shape):
        pass

    @abc.abstractmethod
    def zero_like(self, x):
        pass

    @abc.abstractmethod
    def zero_vec(self, *shape):
        pass

    @abc.abstractmethod
    def zero_vec_like(self, x):
        pass

    @abc.abstractmethod
    def inner(self, x, u, v, keepdim=False):
        pass

    def norm(self, x, u, squared=False, keepdim=False):
        norm_sq = self.inner(x, u, u, keepdim)
        norm_sq.data.clamp_(EPS[u.dtype])
        return norm_sq if squared else norm_sq.sqrt()

    @abc.abstractmethod
    def proju(self, x, u):
        pass

    def proju0(self, u):
        pass

    @abc.abstractmethod
    def projx(self, x):
        pass

    def egrad2rgrad(self, x, u):
        pass

    @abc.abstractmethod
    def exp(self, x, u):
        pass

    def exp0(self, u):
        pass

    @abc.abstractmethod
    def log(self, x, y):
        pass

    def log0(self, y):
        pass
        
    def dist(self, x, y, squared=False, keepdim=False):
        pass

    def pdist(self, x, squared=False):
        pass

    def transp(self, x, y, u):
        pass

    def transpfrom0(self, x, u):
        pass
    
    def transpto0(self, x, u):
        pass

    def mobius_addition(self, x, y):
        pass

    @abc.abstractmethod
    def sh_to_dim(self, shape):
        pass

    @abc.abstractmethod
    def dim_to_sh(self, dim):
        pass

    @abc.abstractmethod
    def squeeze_tangent(self, x):
        pass

    @abc.abstractmethod
    def unsqueeze_tangent(self, x):
        pass

    @abc.abstractmethod
    def rand(self, *shape):
        pass

    @abc.abstractmethod
    def randvec(self, x, norm=1):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass

    def logdetexp(self, x, u):
        #very expensive rip
        if len(u.shape) == 1:
            return torch.det(AF.jacobian(lambda v: self.exp(x, v), u))
        else:
            jacobians = [AF.jacobian(lambda v: self.exp(x[i], v), u[i]) for
                    i in range(u.shape[0])]
            return torch.det(torch.stack(jacobians))

    def logdetlog(self, x, y):
        pass

    def logdetexp0(self, u):
        pass
    
    def logdetlog0(self, y):
        pass
