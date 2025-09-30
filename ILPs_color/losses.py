import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

def get_han_loss(x,y,perc=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]):

    sortedInds = (0.5-x).abs().sort(descending=True)[1]
    han_dises = []
    for p in perc:
        inds = sortedInds[0:int(p*len(sortedInds))]
        hans_dis = (x[inds].round() != y[inds].round()).sum()
        han_dises.append(hans_dis.item())

    return han_dises


def labelOpt(X_hat,X,lr=0.001,lamda=25,ITRS = 10000,device='cuda:0'):
    '''
    X_hat：[1xqxq]
    X: [1xqxq]
    '''
    size = X.shape[-1]
    P = torch.zeros(X.shape[0],size,size).to(device)
    X_hat_ = X_hat.squeeze(dim=0).detach().cpu().numpy()
    X_t = X.squeeze(dim=0).t().cpu().numpy()

    r_i,c_i = linear_sum_assignment(X_t@X_hat_,maximize=True)
    P[:,r_i,c_i] = 1

    X_bar = X @ P
    return X_bar

def lexOpt(_,X,lr=0.001,lamda=25,ITRS = 10000,device='cuda:0'):

    X_bar = X
    newX = X_bar.detach()
    X_bar = X.cpu().numpy().astype(int).astype(str)

    n,nr,nc = X_bar.shape

    for i in range(n):

        Y = X_bar[i]
        Y = list(Y.transpose())
        Y = [ ''.join(list(y)) for y in Y]
        Y = np.array(Y)
        inds = np.argsort(Y)[::-1]
        newY = X[i][:,list(inds)]
        newX[i] = newY

    return newX


def sinkhorn(C, itrs=100, device='cuda:0'):

    eps = torch.eye(C.shape[-1]).to(device) * 0.001
    C = C + eps[None, :, :]
    P = torch.zeros_like(C)
    C = C.cpu().numpy()
    for n in range(C.shape[0]):
        for t in range(C.shape[1]):
            x = C[n]
            max_index = np.unravel_index(np.argmax(x, axis=None), x.shape)
            P[n,max_index[0],max_index[1]] = 1
            C[:,max_index[0],:] = -1
            C[:, :,max_index[1]] = -1
    return P



if __name__ == '__main__':

    X = torch.Tensor(
        [
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ]
    )

    X_hat = torch.Tensor(
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
    )

    X_bar = labelOpt(X_hat[None,:,:].clone(),X[None,:,:].clone(),device='cpu')

    print('done')