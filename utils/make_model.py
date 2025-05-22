# Add the top-level directory to sys.path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))

from utils.models import WTanh, WSoftmax
from utils.models import GradNet_C, mGradNet_C, GradNet_M, mGradNet_M



def make_model(model_name, dim):

    ## Load model
    if model_name == 'mGN-C':
        h = {2:93, 32:629, 256:948, 1024:1003}
        model = mGradNet_C(num_layers=4, in_dim=dim, embed_dim=h[dim], activation=lambda: WTanh(h[dim], monotone=True))

    elif model_name == 'GN-C':
        h = {2:93, 32:629, 256:948, 1024:1003}
        model = GradNet_C(num_layers=4, in_dim=dim, embed_dim=h[dim], activation=lambda: WTanh(h[dim], monotone=False))

    elif model_name == 'mGN-M':  
        h = {2:169, 32:247, 256:254, 1024:255}
        model = mGradNet_M(num_modules=4, in_dim=dim, embed_dim=h[dim], activation=lambda: WSoftmax(h[dim], monotone=True))

    elif model_name == 'GN-M':
        h = {2:169, 32:247, 256:254, 1024:255}
        model = GradNet_M(num_modules=4, in_dim=dim, embed_dim=h[dim], activation=lambda: WSoftmax(h[dim], monotone=False))

    return model

def compute_hidden(fn, num_params):
    
    x = 1
    model = fn(x)
    param_count = sum(p.size().numel() for p in model.parameters() if p.requires_grad)
    while  param_count <= num_params:
        x+=1
        model = fn(x)
        param_count = sum(p.size().numel() for p in model.parameters() if p.requires_grad)
    return x-1


if __name__=="__main__":

    # Prints number of parameters 
    dim = 256
    print('Input Dim: ', dim)
    num_params = 1024*dim

    mGNC_hidden = compute_hidden(lambda x: mGradNet_C(num_layers=4, in_dim=dim, embed_dim=x, activation=lambda: WTanh(x, monotone=True)), num_params=num_params)
    print('mGN-C Embed Dim: ', mGNC_hidden)

    GNC_hidden = compute_hidden(lambda x: GradNet_C(num_layers=4, in_dim=dim, embed_dim=x, activation=lambda: WTanh(x, monotone=False)), num_params=num_params)
    print('GN-C Embed Dim: ', GNC_hidden)

    mGNM_hidden = compute_hidden(lambda x: mGradNet_M(num_modules=4, in_dim=dim, embed_dim=x, activation=lambda: WSoftmax(x, monotone=True)), num_params=num_params)
    print('mGN-M Embed Dim: ', mGNM_hidden)

    GNM_hidden = compute_hidden(lambda x: GradNet_M(num_modules=4, in_dim=dim, embed_dim=x, activation=lambda: WSoftmax(x, monotone=False)), num_params=num_params)
    print('GN-M Embed Dim: ', GNM_hidden)