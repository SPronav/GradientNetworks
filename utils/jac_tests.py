import torch 
from torch.autograd.functional import jacobian 
import numpy as np 
torch.manual_seed(42)

def is_symmetric(matrix):
    return torch.allclose(matrix, matrix.T, atol=1e-5)

def is_psd(matrix):
    eigvals = torch.linalg.eigvalsh(matrix)
    return eigvals.amin() > -1e-5

def check_jac_sym(model, domain, in_dim, device, num_samples=1000):

    for i in range(num_samples):
        x = (domain[1] - domain[0]) * torch.rand(size=(1,in_dim), device=device) + domain[0]
        J = torch.autograd.functional.jacobian(model, x)
        J = J.squeeze()

        if not is_symmetric(J):
            print('Jacobian symmetric test failed')
            print(J)
            return 

    print('jacobian symmetric test passed')
    return 

def check_jac_psd(model, domain, in_dim, device, num_samples=1000):

    for i in range(num_samples):
        x = (domain[1] - domain[0]) * torch.rand(size=(1,in_dim), device=device) + domain[0]
        J = torch.autograd.functional.jacobian(model, x)
        J = J.squeeze()

        if not is_symmetric(J) or not is_psd(J):
            print('Jacobian PSD Test Failed')
            return 

    print('jacobian PSD Test Passed')
    return 

