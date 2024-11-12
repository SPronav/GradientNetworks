import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

class mGradNet_C(nn.Module):
    def __init__(self, num_layers, in_dim, embed_dim, activation):
        super().__init__()
       
        self.num_layers = num_layers
        self.nonlinearity = nn.ModuleList([activation() for i in range(num_layers)])
        self.W = nn.Parameter(init_W(embed_dim, in_dim), requires_grad=True)

        self.bias = nn.ParameterList([nn.Parameter(init_b(embed_dim, embed_dim), requires_grad=True) for i in range(num_layers+1)])
        self.bias[0] = nn.Parameter(init_b(embed_dim, in_dim), requires_grad=True)
        self.bias[-1] = nn.Parameter(init_b(in_dim, embed_dim), requires_grad=True)

        self.beta = nn.ParameterList([nn.Parameter(torch.rand(embed_dim,), requires_grad=True) for i in range(num_layers)])
        self.alpha = nn.ParameterList([nn.Parameter(torch.rand(embed_dim,), requires_grad=True) for i in range(num_layers)])

    def forward(self, x):
        z = F.softplus(self.beta[0]).view(1,-1) * F.linear(x, self.W, self.bias[0])
        for i in range(self.num_layers - 1):
            skip = F.softplus(self.beta[i+1]).view(1,-1) * F.linear(x, self.W, self.bias[i+1])
            z = skip + F.softplus(self.alpha[i]).view(1,-1) * self.nonlinearity[i](z)

        z = F.softplus(self.alpha[-1]).view(1,-1) * self.nonlinearity[-1](z)
        z = F.linear(z, self.W.T, self.bias[-1])

        return z
    

class GradNet_C(nn.Module):
    def __init__(self, num_layers, in_dim, embed_dim, activation):
        super().__init__()

        self.num_layers = num_layers
        self.nonlinearity = nn.ModuleList([activation() for i in range(num_layers)])
        
        self.W = nn.Parameter(init_W(embed_dim, in_dim), requires_grad=True)
        self.bias = nn.ParameterList([nn.Parameter(init_b(embed_dim, embed_dim), requires_grad=True) for i in range(num_layers+1)])
        self.bias[0] = nn.Parameter(init_b(embed_dim, in_dim), requires_grad=True)
        self.bias[-1] = nn.Parameter(init_b(in_dim, embed_dim), requires_grad=True)
        
        self.beta = nn.ParameterList([nn.Parameter(torch.rand(embed_dim)-0.5, requires_grad=True) for i in range(num_layers)])
        self.alpha = nn.ParameterList([nn.Parameter(torch.rand(embed_dim)-0.5, requires_grad=True) for i in range(num_layers)])
        
    def forward(self, x):
        z = self.beta[0].view(1,-1) * F.linear(x, self.W, self.bias[0])
        for i in range(self.num_layers - 1):
            skip = self.beta[i+1].view(1,-1) * F.linear(x, self.W, self.bias[i+1])
            z = skip + self.alpha[i].view(1,-1) * self.nonlinearity[i](z)

        z = self.alpha[-1].view(1,-1) * self.nonlinearity[i](z)
        z = F.linear(z, self.W.T, self.bias[-1])

        return z
    


class GNM_Module(nn.Module):
    def __init__(self, in_dim, embed_dim, activation):
        super().__init__()

        self.beta = nn.Parameter(torch.rand(1), requires_grad=True)
        self.W = nn.Parameter(init_W(embed_dim, in_dim), requires_grad=True)
        self.b = nn.Parameter(init_b(embed_dim, in_dim), requires_grad=True)
        self.act = activation()

    def forward(self, x):

        z = F.linear(x, weight = self.W, bias=self.b)
        z = self.act(z * F.softplus(self.beta))
        z = F.linear(z, weight=self.W.T)

        return z
    
  
class mGradNet_M(nn.Module):
    def __init__(self, num_modules, in_dim, embed_dim, activation):
        super().__init__()

        self.num_modules = num_modules
        self.mmgn_modules = nn.ModuleList([GNM_Module(in_dim, embed_dim, activation) for i in range(num_modules)])
        self.alpha = nn.Parameter(torch.rand(num_modules,), requires_grad=True)
        self.bias = nn.Parameter(init_b(in_dim, embed_dim), requires_grad=True)

    def forward(self, x):

        z = 0
        for i in range(self.num_modules):
            out = self.mmgn_modules[i](x)
            z += F.softplus(self.alpha[i]) * out 
        z += self.bias
        return z
    
 
class GradNet_M(nn.Module):
    def __init__(self, num_modules, in_dim, embed_dim, activation):
        super().__init__()

        self.num_modules = num_modules
        self.mmgn_modules = nn.ModuleList([GNM_Module(in_dim, embed_dim, activation) for i in range(num_modules)])
        self.alpha = nn.Parameter(torch.randn(num_modules,), requires_grad=True)
        self.bias = nn.Parameter(init_b(in_dim, embed_dim), requires_grad=True)

    def forward(self, x):
        out = 0
        for i in range(self.num_modules):
            out += self.alpha[i] * self.mmgn_modules[i](x)
        out+= self.bias 
        return out 