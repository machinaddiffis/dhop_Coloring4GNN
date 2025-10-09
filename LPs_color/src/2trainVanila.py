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






sin_history = {}

def color_permute(dim, v, c, ncolor):
    tarv = v
    tarc = c

    keys = list(range(ncolor))
    values = keys[:]  
    random.shuffle(values)
    mapping = dict(zip(keys, values))
    v = torch.tensor(np.vectorize(mapping.get)(v))
    c = torch.tensor(np.vectorize(mapping.get)(c))
    return v,c


def handle_color_emb(fnm, d, v, c, edge_index):
    if fnm in sin_history:
        col_L = sin_history[fnm][0]
        col_R = sin_history[fnm][1]
    else:
        k=2#k染色层
        #toy graph
        num_v = v.shape[0]
        num_c = c.shape[0] #variable的数量，constraints的数量

        col_L, col_R = global_khop_coloring(edge_index, num_v, num_c, k=k)
        sin_history[fnm] = [None,None]
        sin_history[fnm][0] = col_L
        sin_history[fnm][1] = col_R
    # permute 
    ccc = torch.cat([col_L, col_R])
    num_classes = ccc.unique().numel()
    col_L, col_R = color_permute(d, col_L, col_R, ncolor=num_classes)
    color_v = []
    color_c = []
    for _ in range(d):
        color_v.append(col_L)
        color_c.append(col_R)
    color_vf = torch.stack(color_v, dim=0).t()
    color_cf = torch.stack(color_c, dim=0).t()

    color_cf = PE_matrix(color_cf).to(c.device)#concat to variable nodes input
    color_vf = PE_matrix(color_vf).to(v.device)#concat to constraints nodes input
    v = torch.cat((v, color_vf), dim = 1)
    c = torch.cat((c, color_cf), dim = 1)

    return v, c

def handle_uniform_emb(fnm, e_size, v, c, edge_index):
    if fnm in sin_history:
        return sin_history[fnm][0],sin_history[fnm][1]

    dim = e_size
    device = v.device
    dtype = v.dtype
    # -- v
    v_add = torch.randn(v.shape[0], dim, dtype=dtype).to(device)
    v = torch.cat((v, v_add), dim = 1)
    # -- c
    c_add = torch.randn(c.shape[0], dim, dtype=dtype).to(device)
    c = torch.cat((c, c_add), dim = 1)

    sin_history[fnm] = []
    sin_history[fnm].append(v)
    sin_history[fnm].append(c)
    return v,c

def handle_zero_emb(fnm, e_size, v, c, edge_index):
    v = torch.cat((v, torch.zeros(v.shape[0], e_size, dtype=v.dtype, device=v.device)), dim = 1)
    c = torch.cat((c, torch.zeros(c.shape[0], e_size, dtype=c.dtype, device=c.device)), dim = 1)
    return v,c

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e","--maxepoch", type=int, help="number of max epoch", default=5000)
    parser.add_argument("-b","--embsize", type=int, help="n dimension of embeding", default=32)
    parser.add_argument("-d","--hidden", type=str, help="dimensions of hiden layers, splitted by ,", default="64,64,64")
    parser.add_argument("-i","--ident", type=str, help="identification of model", default="vanilla")
    parser.add_argument("-p","--problem", type=str, help="identification of problem", default="pagerank")
    parser.add_argument("--contin",action="store_true",help="Enable continue from last checkpoint")
    args = parser.parse_args()

    # file locations
    flist_train = os.listdir(f'../data/{args.problem}/train')
    flist_valid = os.listdir(f'../data/{args.problem}/valid')
    # obtain sizes
    f = gzip.open(f'../data/{args.problem}/train/{flist_train[0]}','rb')
    pkl =  pickle.load(f)
    x_size = pkl['var_feat'].shape[-1] + args.embsize
    y_size = pkl['con_feat'].shape[-1] + args.embsize
    f.close()

    # set up env
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_sizes = args.hidden.split(',')
    feat_sizes = [int(x) for x in feat_sizes]
    mdl = pdhg_net(x_size,y_size,feat_sizes).to(device)
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(mdl.parameters(), lr=1e-4)
    max_epoch = args.maxepoch
    best_loss = 1e+20   
    model_name = f'best_{args.ident}.mdl'
    # handle embedding function
    emb_func = handle_zero_emb
    if args.ident == 'color':
        emb_func = handle_color_emb
        if args.contin and os.path.exists(f'../pkl/{args.problem}/hist_{args.ident}.pkl'):
            ff = gzip.open(f'../pkl/{args.problem}/hist_{args.ident}.pkl','rb')
            sin_history = pickle.load(ff)
            ff.close()
            print(f'Sin history loaded')
    if args.ident == 'uniform':
        emb_func = handle_uniform_emb
        if args.contin and os.path.exists(f'../pkl/{args.problem}/hist_{args.ident}.pkl'):
            ff = gzip.open(f'../pkl/{args.problem}/hist_{args.ident}.pkl','rb')
            sin_history = pickle.load(ff)
            ff.close()
            print(f'Sin history loaded')


    # check if will start from ckp
    
    if args.contin and os.path.exists(f"../model/{args.problem}/{model_name}"):
        checkpoint = torch.load(f"../model/{args.problem}/{model_name}")
        mdl.load_state_dict(checkpoint['model'])
        if 'nepoch' in checkpoint:
            last_epoch=checkpoint['nepoch']
        best_loss=checkpoint['best_loss']
        print(f'Last best val loss gen:  {best_loss}')
        print('Model Loaded')
    if not os.path.exists(f"../model/{args.problem}"):
        os.makedirs(f"../model/{args.problem}")
    if not os.path.exists(f'../pkl/{args.problem}'):
        os.makedirs(f'../pkl/{args.problem}')
    if not os.path.exists(f'../logs/{args.problem}'):
        os.makedirs(f'../logs/{args.problem}')
    # train
    log_mode = 'w'
    if args.contin:
        log_mode = 'a'
    flog = open(f'../logs/{args.problem}/train_log_{args.ident}.log',log_mode)
    updated_hist = False
    for epoch in range(max_epoch):
        avg_loss_x=0.0
        avg_loss_y=0.0
        random.shuffle(flist_train)
        with alive_bar(len(flist_train),  title='Training') as bar:
            for fnm in flist_train:
                # train
                f = gzip.open(f'../data/{args.problem}/train/{fnm}','rb')
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
                x,y = emb_func(f'../data/{args.problem}/train/{fnm}', args.embsize, x, y, A_idx)
                b = torch.as_tensor(b,dtype=torch.float32).to(device)
                c = torch.as_tensor(c,dtype=torch.float32).to(device)
                x_gt = torch.as_tensor(sol,dtype=torch.float32).to(device)
                y_gt = torch.as_tensor(dual,dtype=torch.float32).to(device)
                f.close()

                #  apply gradient 
                optimizer.zero_grad()
                x,y = mdl(x,y,A,AT,c,b)
                loss_x = loss_func(x, x_gt)
                loss_y = loss_func(y, y_gt) 
                loss = loss_x + loss_y
                avg_loss_x += loss_x.item()
                avg_loss_y += loss_y.item()
                loss.backward()
                optimizer.step()
                bar()
        avg_loss_x /= round(len(flist_train),2)
        avg_loss_y /= round(len(flist_train),2)
        print(f'Epoch {epoch} Train:::: primal loss:{avg_loss_x}    dual loss:{avg_loss_y}')
        st = f'epoch{epoch}train: {avg_loss_x} {avg_loss_y}\n'
        flog.write(st)



        avg_loss_x=0.0
        avg_loss_y=0.0
        with alive_bar(len(flist_valid),  title='Valid') as bar:
            for fnm in flist_valid:
                # valid
                #  reading
                f = gzip.open(f'../data/{args.problem}/valid/{fnm}','rb')
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
                x,y = emb_func(f'../data/{args.problem}/valid/{fnm}',args.embsize, x, y, A_idx)
                b = torch.as_tensor(b,dtype=torch.float32).to(device)
                c = torch.as_tensor(c,dtype=torch.float32).to(device)
                x_gt = torch.as_tensor(sol,dtype=torch.float32).to(device)
                y_gt = torch.as_tensor(dual,dtype=torch.float32).to(device)
                f.close()
                #  obtain loss
                x,y = mdl(x,y,A,AT,c,b)
                loss_x = loss_func(x, x_gt)
                loss_y = loss_func(y, y_gt) 
                loss = loss_x + loss_y
                avg_loss_x += loss_x.item()
                avg_loss_y += loss_y.item()
                bar()
        avg_loss_x /= round(len(flist_valid),2)
        avg_loss_y /= round(len(flist_valid),2)
        print(f'Epoch {epoch} Valid:::: primal loss:{avg_loss_x}    dual loss:{avg_loss_y}')
        st = f'epoch{epoch}valid: {avg_loss_x} {avg_loss_y}\n'
        flog.write(st)


        if best_loss > avg_loss_x+avg_loss_y:
            best_loss = avg_loss_x+avg_loss_y
            state={'model':mdl.state_dict(),'optimizer':optimizer.state_dict(),'best_loss':best_loss,'nepoch':epoch}
            torch.save(state,f"../model/{args.problem}/{model_name}")
            print(f'Saving new best model with valid loss: {best_loss}')

        flog.flush()

        if not updated_hist:
            ff = gzip.open(f'../pkl/{args.problem}/hist_{args.ident}.pkl','wb')
            pickle.dump(sin_history, ff)
            ff.close()
            updated_hist = True
            print(f'Updated sin_history')
    flog.close()
    

