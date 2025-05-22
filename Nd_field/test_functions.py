import torch
import numpy as np

def get_grad(x, fn, train=True):
    pts = x.clone()
    pts.requires_grad = True
    out = fn(pts) 
    out = out.sum() 
    out.backward()
    return pts.grad

def get_normalizing_constant(fn, dim, device):
    n_points = 10000
    x = torch.rand(n_points, dim, device=device)
    grad = get_grad(x, fn)
    c = torch.linalg.vector_norm(grad, dim=-1).mean().item()
    return c

class Field():
    def __init__(self, dim):
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed) 
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        self.dim = dim
        self.c = 0

    def eval_potential(self, x, train=True):
        return

    def __call__(self, x, train=True):
        pts = x.clone()
        pts.requires_grad = True
        out = self.eval_potential(pts, train) 
        out = out.sum() 
        out.backward()
        return pts.grad


# x has shape (batch, dim)
class PolyMax(Field):
    def __init__(self, dim, device):
        super().__init__(dim)

        idx = torch.arange(dim).view(-1,1)+1
        alpha = (idx + idx.T - 2) / (2*dim - 2)
        S = (2 + np.sin(4*np.pi * alpha))/(np.log(dim)*(1 + torch.abs(idx - idx.T)))
        P = (1 + 2*alpha)/(np.log(dim)*(1 + torch.abs(idx - idx.T)))
        Q = (3 - 2*alpha)/(np.log(dim)*(1 + torch.abs(idx - idx.T)))
        
        assert torch.linalg.eigvalsh(S).amin() > 0
        assert torch.linalg.eigvalsh(P).amin() > 0
        assert torch.linalg.eigvalsh(Q).amin() > 0
        self.A = torch.stack([S, P, Q], dim=0).to(device)
        def temp(x):
            x = x - 0.5
            xAx = torch.einsum('ni, mij, nj-> nm', x, self.A, x)
            y = torch.amax(xAx, dim=-1, keepdim=True)
            return y
        self.c = get_normalizing_constant(temp, dim, device)

    def eval_potential(self, x, train=True):
        x = x - 0.5
        xAx = torch.einsum('ni, mij, nj-> nm', x, self.A, x)
        y = torch.amax(xAx, dim=-1, keepdim=True)
        if not train:
            assert (torch.sum(y == xAx, dim=-1, keepdim=True) == torch.ones_like(y)).all()
        return y / self.c