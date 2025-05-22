import sys
import os
import torch 
import torch.nn as nn 
import numpy as np 
from tqdm import tqdm
import argparse 

# Add the top-level directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))

from utils.make_model import *
from test_functions import PolyMax

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='Get the model name from the command line.')
    parser.add_argument('--model', type=str, required=True, help='Specify model')
    parser.add_argument('--fn', type=str, required=True, help='Specify the test function')
    parser.add_argument('--dim', type=int, required=True, help='Specify the dimension')
    parser.add_argument('--gpu', type=int, required=True, help='Specify the gpu')
    parser.add_argument('--lr', type=float, default=1e-3, help='Specify the learning rate')
    args = parser.parse_args()
    model_name = args.model
    gpu = args.gpu
    fn = args.fn
    dim = args.dim
    lr = args.lr

    # training parameters
    iters = 10000
    batch_size = 100
    criterion = nn.MSELoss()

    if torch.cuda.is_available():
        device = torch.device("cuda:"+str(gpu))
        print("GPU available. Using : " + torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("GPU not available. Using CPU.")
    
    ## Load function
    field = PolyMax(dim, device)

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model = make_model(model_name, dim).to(device)
    print("Number of parameters", sum(p.size().numel() for p in model.parameters() if p.requires_grad))


    ## Train
    pbar = tqdm(range(iters), leave=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for i in range(iters):
        x = torch.rand(batch_size, dim).to(device)
        y = field(x)
        
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y) 
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Loss: {loss.item():.2E}")
        pbar.update(1)
                            

    # Check if the directory exists, create it if not
    save_path = 'trained_models/' + str(fn) + '/dim' + str(dim) + '/' + model_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Directory '{save_path}' created.")
    else:
        print(f"Directory '{save_path}' already exists.")

    model.cpu()
    torch.save(model.state_dict(), save_path+'/model'+'-lr'+str(lr)+'.pt')

    print('Saved Model.')