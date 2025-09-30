from model import *
import torch 
import gzip
import pickle
import os
import random
import argparse
from alive_progress import alive_bar
import math
from LP_coloring import *





def handle_color_emb(fnm, d, v, c, edge_index):
    k=2#k染色层
    #toy graph
    num_v = v.shape[0]
    num_c = c.shape[0] #variable的数量，constraints的数量

    col_L, col_R = global_khop_coloring(edge_index, num_v, num_c, k=k)
    color_v = []
    color_c = []
    # ccc = torch.cat([col_L, col_R])
    # num_classes = ccc.unique().numel()
    for _ in range(d):
        v_permuted = col_L
        c_permuted = col_R
        color_v.append(v_permuted)
        color_c.append(c_permuted)
    color_vf = torch.stack(color_v, dim=0).t()
    color_cf = torch.stack(color_c, dim=0).t()

    color_cf = PE_matrix(color_cf).to(c.device)#concat to variable nodes input
    color_vf = PE_matrix(color_vf).to(v.device)#concat to constraints nodes input
    v = torch.cat((v, color_vf), dim = 1)
    c = torch.cat((c, color_cf), dim = 1)
    return v, c

def handle_gauss_emb(fnm, e_size, v, c, edge_index):

    dim = e_size
    device = v.device
    dtype = v.dtype
    # -- v
    v_add = torch.randn(v.shape[0], dim, dtype=dtype).to(device)
    v = torch.cat((v, v_add), dim = 1)
    # -- c
    c_add = torch.randn(c.shape[0], dim, dtype=dtype).to(device)
    c = torch.cat((c, c_add), dim = 1)

    return v,c

def handle_zero_emb(fnm, e_size, v, c, edge_index):
    v = torch.cat((v, torch.zeros(v.shape[0], e_size, dtype=v.dtype, device=v.device)), dim = 1)
    c = torch.cat((c, torch.zeros(c.shape[0], e_size, dtype=c.dtype, device=c.device)), dim = 1)
    return v,c

def handle_sinusoidal_emb(fnm, e_size, v, c, edge_index):

    dim = e_size
    device = v.device
    dtype = v.dtype
    # -- v
    eps = 0.5
    v_add = torch.zeros(v.shape[0], dim, dtype=dtype)
    for i in range(v.shape[0]):
        for j in range(dim // 2):
            v_add[i, 2 * j] = math.sin(eps * i / math.pow(10000, 2 * j / dim))
            v_add[i, 2 * j + 1] = math.cos(eps * i / math.pow(10000, 2 * j / dim))
    v_add = v_add.to(device)
    v = torch.cat((v, v_add), dim = 1)
    # -- c
    c_add = torch.zeros(c.shape[0], dim, dtype=dtype)
    for i in range(c.shape[0]):
        for j in range(dim // 2):
            c_add[i, 2 * j] = math.sin(eps * i / math.pow(10000, 2 * j / dim))
            c_add[i, 2 * j + 1] = math.cos(eps * i / math.pow(10000, 2 * j / dim))
    c_add = c_add.to(device)
    c = torch.cat((c, c_add), dim = 1)
    return v,c





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b","--embsize", type=int, help="n dimension of embeding", default=32)
    parser.add_argument("-d","--hidden", type=str, help="dimensions of hiden layers, splitted by ,", default="64,64,64")
    args = parser.parse_args()

    # file locations
    flist = os.listdir('../data/test')
    # obtain sizes
    f = gzip.open(f'../data/test/{flist[0]}','rb')
    pkl =  pickle.load(f)
    x_size = pkl['var_feat'].shape[-1] + args.embsize
    y_size = pkl['con_feat'].shape[-1] + args.embsize
    f.close()
    for ident in ['color','sin','gauss','vanilla']:
        # set up env
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feat_sizes = args.hidden.split(',')
        feat_sizes = [int(x) for x in feat_sizes]
        mdl = pdhg_net(x_size,y_size,feat_sizes).to(device)
        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(mdl.parameters(), lr=1e-4)
        model_name = f'best_{ident}.mdl'
        # handle embedding function
        emb_func = handle_zero_emb
        if ident == 'sin':
            emb_func = handle_sinusoidal_emb
        if ident == 'color':
            emb_func = handle_color_emb
        if ident == 'gauss':
            emb_func = handle_gauss_emb
        # check if will start from ckp
        if os.path.exists(f"../model/{model_name}"):
            checkpoint = torch.load(f"../model/{model_name}")
            mdl.load_state_dict(checkpoint['model'])
            if 'nepoch' in checkpoint:
                last_epoch=checkpoint['nepoch']
            best_loss=checkpoint['best_loss']
            print(f'Last best val loss gen:  {best_loss}')
            print('Model Loaded')
        else:
            print('No Model Found, cant predict')
            quit()

        # test
        flog = open(f'../logs/test/{ident}.log','w')
        avg_loss_x=0.0
        avg_loss_y=0.0
        with alive_bar(len(flist),  title=f'Testing {ident}') as bar:
            for fnm in flist:
                # train
                f = gzip.open(f'../data/test/{fnm}','rb')
                pkl =  pickle.load(f)
                A_idx = pkl['edge_index']
                A_val = pkl['edge_weight']
                b = pkl['b']
                c = pkl['c']
                x = pkl['var_feat']
                y = pkl['con_feat']
                sol = pkl['label']
                dual = pkl['dual']
                AT = torch.sparse_coo_tensor(A_idx,A_val).to(device)
                A = AT.T
                x = torch.as_tensor(x,dtype=torch.float32).to(device)
                y = torch.as_tensor(y,dtype=torch.float32).to(device)
                # embedding
                x,y = emb_func(f'../data/test/{fnm}', args.embsize, x, y, A_idx)
                b = torch.as_tensor(b,dtype=torch.float32).to(device)
                c = torch.as_tensor(c,dtype=torch.float32).to(device)
                x_gt = torch.as_tensor(sol,dtype=torch.float32).to(device)
                y_gt = torch.as_tensor(dual,dtype=torch.float32).to(device)
                f.close()

                #  apply gradient 
                x,y = mdl(x,y,A,AT,c,b)
                loss_x = loss_func(x, x_gt)
                loss_y = loss_func(y, y_gt) 
                loss = loss_x + loss_y
                avg_loss_x += loss_x.item()
                avg_loss_y += loss_y.item()
                print(f'File {fnm}    primal loss:{loss_x.item()}    dual loss:{loss_y.item()}')
                st = f'File {fnm}    primal loss:{loss_x.item()}    dual loss:{loss_y.item()}\n'
                flog.write(st)
                bar()
        avg_loss_x /= round(len(flist),2)
        avg_loss_y /= round(len(flist),2)
        print(f'{ident} avg:::: primal loss:{avg_loss_x}    dual loss:{avg_loss_y}')
        st = f'{ident} avg:::: primal loss:{avg_loss_x}    dual loss:{avg_loss_y}\n'
        flog.write(st)
        flog.close()